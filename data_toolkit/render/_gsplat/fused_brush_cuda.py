"""Fused Apache-2.0 CUDA backend for Brush-aligned pinhole 3DGS rendering.

Accelerates perspective rendering in ``GaussianScene`` by replacing the pure-PyTorch
per-Gaussian projection and un-culled 64-bit tile sort with:
  1. Dual-bucket tight-voxel frustum culling (0.50m voxels for regular splats +
     individual bounding spheres for large splats; strictly lossless).
  2. Single-pass warp-ballot Brush-aligned projection with FlashGS-style exact
     conic-ellipse tile intersection.
  3. Vectorized 128-bit FP16 SH3 storage and evaluation (~50% lower SH3 VRAM & BW).
  4. 32-bit packed (tile_id, monotonic float-depth) CUB radix sort.
  5. Front-batch-first HiGS-style bounded-batch split-tile rasterizer writing
     uint8 RGB, metric radial depth, and alpha directly on GPU with pinned D2H.
  6. Optional 4-byte-per-splat zone PVS bitmask for indoor map caching.
"""

from __future__ import annotations

import math
import os
import shutil
import threading
import warnings
from typing import Optional

import numpy as np
import torch

from .gsplat_brush_common import ALPHA_CUTOFF, SH_INRIA_BIAS, c2w_to_w2c
from .gsplat_pinhole_from_viewer import PinholeIntrinsics, _camera_params

_EXT_LOCK = threading.Lock()
_EXT_MODULE = None
_EXT_FAILED = False

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <math.h>
#include <cstdint>

__device__ __forceinline__ float dev_length_robust(float x, float y, float z) {
    float ax = fabsf(x), ay = fabsf(y), az = fabsf(z);
    float sc = fmaxf(ax, fmaxf(ay, az));
    if (sc <= 0.0f) return 0.0f;
    float inv = 1.0f / sc;
    float sx = x * inv, sy = y * inv, sz = z * inv;
    return sc * sqrtf(sx * sx + sy * sy + sz * sz);
}

__device__ __forceinline__ bool ellipse_intersects_tile(
    float mx, float my, float c0, float c1, float c2, float max_q,
    int tx, int ty)
{
    float dx0 = (float)(tx * 16) + 0.5f - mx;
    float dx1 = (float)(tx * 16 + 15) + 0.5f - mx;
    float dy0 = (float)(ty * 16) + 0.5f - my;
    float dy1 = (float)(ty * 16 + 15) + 0.5f - my;
    if (dx0 <= 0.0f && dx1 >= 0.0f && dy0 <= 0.0f && dy1 >= 0.0f) return true;
    float min_q = 1e30f;
    { float y = fminf(fmaxf(-c1 * dx0 / c2, dy0), dy1); min_q = fminf(min_q, c0*dx0*dx0 + 2.0f*c1*dx0*y + c2*y*y); }
    { float y = fminf(fmaxf(-c1 * dx1 / c2, dy0), dy1); min_q = fminf(min_q, c0*dx1*dx1 + 2.0f*c1*dx1*y + c2*y*y); }
    { float x = fminf(fmaxf(-c1 * dy0 / c0, dx0), dx1); min_q = fminf(min_q, c0*x*x + 2.0f*c1*x*dy0 + c2*dy0*dy0); }
    { float x = fminf(fmaxf(-c1 * dy1 / c0, dx0), dx1); min_q = fminf(min_q, c0*x*x + 2.0f*c1*x*dy1 + c2*dy1*dy1); }
    return min_q <= max_q + 1e-4f;
}

__global__ void select_voxels_pvs_kernel(
    int V, uint32_t zone_bit,
    const float* __restrict__ v_centers,
    const float* __restrict__ v_radii,
    const int* __restrict__ v_counts,
    const uint32_t* __restrict__ v_zmask,
    const float* __restrict__ view,
    int width, int height, float fx, float fy, float cx, float cy,
    int* __restrict__ d_n_vis,
    int* __restrict__ vis_v_ids,
    int* __restrict__ vis_v_cnt)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    bool keep = false;
    int cnt = 0;
    if (vid < V) {
        if (zone_bit == 0u || (v_zmask[vid] & zone_bit) != 0u) {
            float px = v_centers[vid * 3 + 0];
            float py = v_centers[vid * 3 + 1];
            float pz = v_centers[vid * 3 + 2];
            float r  = v_radii[vid];
            float x = view[0]*px + view[1]*py + view[2]*pz + view[3];
            float y = view[4]*px + view[5]*py + view[6]*pz + view[7];
            float z = view[8]*px + view[9]*py + view[10]*pz + view[11];
            if (z + r >= 0.01f) {
                float qxmin = -cx / fx, qxmax = ((float)width - cx) / fx;
                float qymin = -cy / fy, qymax = ((float)height - cy) / fy;
                float n0 = sqrtf(1.0f + qxmin * qxmin);
                float n1 = sqrtf(1.0f + qxmax * qxmax);
                float n2 = sqrtf(1.0f + qymin * qymin);
                float n3 = sqrtf(1.0f + qymax * qymax);
                if ((x - qxmin * z + r * n0 >= 0.0f) &&
                    (-x + qxmax * z + r * n1 >= 0.0f) &&
                    (y - qymin * z + r * n2 >= 0.0f) &&
                    (-y + qymax * z + r * n3 >= 0.0f)) {
                    keep = true;
                    cnt = v_counts[vid];
                }
            }
        }
    }
    unsigned mask = __ballot_sync(0xffffffff, keep);
    int lane = threadIdx.x & 31;
    int wcnt = __popc(mask);
    int wbase = 0;
    if (lane == 0 && wcnt > 0) wbase = atomicAdd(d_n_vis, wcnt);
    wbase = __shfl_sync(0xffffffff, wbase, 0);
    if (keep) {
        int out_i = wbase + __popc(mask & ((1u << lane) - 1u));
        vis_v_ids[out_i] = vid;
        vis_v_cnt[out_i] = cnt;
    }
}

__global__ void expand_voxels_kernel(
    const int* __restrict__ vis_v_ids,
    const int* __restrict__ v_starts,
    const int* __restrict__ v_counts,
    const int* __restrict__ v_out_offs,
    int n_vis_v,
    int* __restrict__ cand_idx)
{
    int bi = blockIdx.x;
    if (bi >= n_vis_v) return;
    int vid = vis_v_ids[bi];
    int s = v_starts[vid];
    int cnt = v_counts[vid];
    int out_base = v_out_offs[bi];
    for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
        cand_idx[out_base + i] = s + i;
    }
}

__global__ void brush_project_v5_kernel(
    const int* __restrict__ cand_idx, int K, uint32_t zone_bit,
    const uint32_t* __restrict__ g_zmask,
    const float* __restrict__ means,
    const float* __restrict__ quats,
    const float* __restrict__ scales,
    const float* __restrict__ opacities,
    const __half* __restrict__ sh_fp16,
    const float* __restrict__ view,
    const float* __restrict__ cam_pos,
    int width, int height, int tw, int th,
    float fx, float fy, float cx, float cy, float skew,
    float sh_bias, float min_rad_clip,
    int* __restrict__ d_M,
    float4* __restrict__ out_xy_op_maxq,
    float4* __restrict__ out_conic_dep,
    float4* __restrict__ out_col_rng,
    uint32_t* __restrict__ out_tile_bbox,
    int* __restrict__ out_tiles_touched)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool keep = false;
    float mx = 0, my = 0, dep = 0, rng = 0, c0 = 0, c1 = 0, c2 = 0, op_out = 0, max_q = 0;
    int tx0 = 0, ty0 = 0, tx1 = 0, ty1 = 0, nt = 0;
    int gid = 0;

    if (tid < K) {
        gid = cand_idx[tid];
        bool pvs_ok = (zone_bit == 0u) || ((g_zmask[gid] & zone_bit) != 0u);
        if (pvs_ok) {
            float op = opacities[gid];
            if ((op >= (1.0f / 255.0f)) && isfinite(op)) {
                float px = means[gid * 3 + 0];
                float py = means[gid * 3 + 1];
                float pz = means[gid * 3 + 2];

                float mcx = view[0]*px + view[1]*py + view[2]*pz + view[3];
                float mcy = view[4]*px + view[5]*py + view[6]*pz + view[7];
                float mcz = view[8]*px + view[9]*py + view[10]*pz + view[11];

                if (isfinite(mcx) && isfinite(mcy) && isfinite(mcz) && mcz <= 1e10f) {
                    rng = dev_length_robust(mcx, mcy, mcz);
                    if (rng >= 0.01f) {
                        float qw = quats[gid * 4 + 0];
                        float qx_r = quats[gid * 4 + 1];
                        float qy_r = quats[gid * 4 + 2];
                        float qz_r = quats[gid * 4 + 3];
                        float qnsq = qw*qw + qx_r*qx_r + qy_r*qy_r + qz_r*qz_r;
                        float sx = scales[gid * 3 + 0];
                        float sy = scales[gid * 3 + 1];
                        float sz = scales[gid * 3 + 2];
                        if (isfinite(qnsq) && qnsq >= 1e-6f && isfinite(sx) && isfinite(sy) && isfinite(sz)) {
                            float inv_qn = rsqrtf(qnsq);
                            qw *= inv_qn; qx_r *= inv_qn; qy_r *= inv_qn; qz_r *= inv_qn;

                            float r00 = 1.0f - 2.0f*(qy_r*qy_r + qz_r*qz_r), r01 = 2.0f*(qx_r*qy_r - qw*qz_r),     r02 = 2.0f*(qx_r*qz_r + qw*qy_r);
                            float r10 = 2.0f*(qx_r*qy_r + qw*qz_r),          r11 = 1.0f - 2.0f*(qx_r*qx_r + qz_r*qz_r), r12 = 2.0f*(qy_r*qz_r - qw*qx_r);
                            float r20 = 2.0f*(qx_r*qz_r - qw*qy_r),          r21 = 2.0f*(qy_r*qz_r + qw*qx_r),     r22 = 1.0f - 2.0f*(qx_r*qx_r + qy_r*qy_r);

                            float b00 = (view[0]*r00 + view[1]*r10 + view[2]*r20) * sx;
                            float b01 = (view[0]*r01 + view[1]*r11 + view[2]*r21) * sy;
                            float b02 = (view[0]*r02 + view[1]*r12 + view[2]*r22) * sz;
                            float b10 = (view[4]*r00 + view[5]*r10 + view[6]*r20) * sx;
                            float b11 = (view[4]*r01 + view[5]*r11 + view[6]*r21) * sy;
                            float b12 = (view[4]*r02 + view[5]*r12 + view[6]*r22) * sz;
                            float b20 = (view[8]*r00 + view[9]*r10 + view[10]*r20) * sx;
                            float b21 = (view[8]*r01 + view[9]*r11 + view[10]*r21) * sy;
                            float b22 = (view[8]*r02 + view[9]*r12 + view[10]*r22) * sz;

                            float qxmin = -cx / fx, qxmax = ((float)width - cx) / fx;
                            float qymin = -cy / fy, qymax = ((float)height - cy) / fy;
                            float sk_fx = (skew != 0.0f) ? (skew / fx) : 0.0f;

                            float pl[5][3] = {
                                { 1.0f,  sk_fx, -qxmin},
                                {-1.0f, -sk_fx,  qxmax},
                                { 0.0f,  1.0f,  -qymin},
                                { 0.0f, -1.0f,   qymax},
                                { 0.0f,  0.0f,   1.0f }
                            };
                            float offs[5] = {0.0f, 0.0f, 0.0f, 0.0f, -0.01f};
                            float supp = logf(op * 255.0f);
                            float rad = sqrtf(fmaxf(2.0f * supp, 0.0f));

                            float cn[5][3], var[5];
                            bool reaches = true;
                            #pragma unroll
                            for (int p = 0; p < 5; ++p) {
                                float nx = pl[p][0], ny = pl[p][1], nz = pl[p][2];
                                float lx = b00*nx + b10*ny + b20*nz;
                                float ly = b01*nx + b11*ny + b21*nz;
                                float lz = b02*nx + b12*ny + b22*nz;
                                cn[p][0] = b00*lx + b01*ly + b02*lz;
                                cn[p][1] = b10*lx + b11*ly + b12*lz;
                                cn[p][2] = b20*lx + b21*ly + b22*lz;
                                var[p] = cn[p][0]*nx + cn[p][1]*ny + cn[p][2]*nz;
                                float d = mcx*nx + mcy*ny + mcz*nz + offs[p] + rad * sqrtf(fmaxf(var[p], 0.0f));
                                if (!(d >= 0.0f)) reaches = false;
                            }

                            float iz0 = 1.0f / fmaxf(mcz, 0.01f);
                            float qy = mcy * iz0;
                            float qx = mcx * iz0 + sk_fx * qy;
                            bool inside = (mcz > 0.01f) && (qx >= qxmin) && (qx < qxmax) && (qy >= qymin) && (qy < qymax);
                            float depth_std = sqrtf(fmaxf(var[4], 0.0f));
                            float extent = rad * depth_std;
                            bool front = (mcz - extent) > 0.01f;

                            float ax = mcx, ay = mcy, az = mcz;
                            if (!inside) {
                                #pragma unroll
                                for (int it = 0; it < 2; ++it) {
                                    #pragma unroll
                                    for (int p = 0; p < 4; ++p) {
                                        float dist = ax*pl[p][0] + ay*pl[p][1] + az*pl[p][2];
                                        if (dist < 0.0f) {
                                            float step = -dist / fmaxf(var[p], 1e-12f);
                                            ax += cn[p][0] * step;
                                            ay += cn[p][1] * step;
                                            az += cn[p][2] * step;
                                        }
                                    }
                                }
                            }

                            float tol = 1e-5f * fmaxf(dev_length_robust(ax, ay, az), 1.0f);
                            bool anchor_inside = true;
                            #pragma unroll
                            for (int p = 0; p < 4; ++p) {
                                if (!((ax*pl[p][0] + ay*pl[p][1] + az*pl[p][2]) >= -tol)) anchor_inside = false;
                            }

                            float az_c = fmaxf(az, 0.01f);
                            bool stable = (az > 0.01f) && ((az - extent) > 0.01f) && ((extent / az_c) <= 0.33f);

                            float ddx = ax - mcx, ddy = ay - mcy, ddz = az - mcz;
                            float isx = 1.0f / fmaxf(sx, 1e-7f), isy = 1.0f / fmaxf(sy, 1e-7f), isz = 1.0f / fmaxf(sz, 1e-7f);
                            float u0 = (b00*ddx + b10*ddy + b20*ddz) * (isx * isx);
                            float u1 = (b01*ddx + b11*ddy + b21*ddz) * (isy * isy);
                            float u2 = (b02*ddx + b12*ddy + b22*ddz) * (isz * isz);
                            float dist_sq = u0*u0 + u1*u1 + u2*u2;
                            float tail = expf(-0.5f * dist_sq);

                            bool rescue_base = reaches && anchor_inside && (dist_sq <= 2.0f * supp);
                            bool stable_rescue = rescue_base && front && stable;
                            bool dual_rescue = rescue_base && (!front) && (!stable);

                            float center_excess = fmaxf(3.0f * depth_std / az_c / 0.33f - 1.0f, 0.0f);
                            float center_gain = expf(-0.1f * center_excess);
                            float gain = inside ? center_gain : (dual_rescue ? (tail * 0.45f) : tail);
                            bool valid = (inside || stable_rescue || dual_rescue) && isfinite(ax) && isfinite(ay) && isfinite(az) && isfinite(gain);

                            if (valid) {
                                float iz = 1.0f / az;
                                mx = fx * ax * iz + cx + ((skew != 0.0f) ? (skew * ay * iz) : 0.0f);
                                my = fy * ay * iz + cy;

                                float fw = (float)width, fh = (float)height;
                                float x_lo = (-0.15f * fw - cx) / fx, x_hi = (1.15f * fw - cx) / fx;
                                float y_lo = (-0.15f * fh - cy) / fy, y_hi = (1.15f * fh - cy) / fy;
                                float xz = fminf(fmaxf(ax * iz, x_lo), x_hi);
                                float yz = fminf(fmaxf(ay * iz, y_lo), y_hi);

                                float j00 = fx * iz, j01 = 0.0f, j02 = -fx * iz * xz;
                                if (skew != 0.0f) {
                                    float img_qx = (ax + sk_fx * ay) * iz;
                                    float cl = fminf(fmaxf(img_qx, x_lo), x_hi);
                                    j01 = skew * iz;
                                    j02 = -fx * iz * cl;
                                }
                                float j11 = fy * iz, j12 = -fy * iz * yz;

                                float v00 = j00*b00 + j01*b10 + j02*b20;
                                float v01 = j00*b01 + j01*b11 + j02*b21;
                                float v02 = j00*b02 + j01*b12 + j02*b22;
                                float v10 = j11*b10 + j12*b20;
                                float v11 = j11*b11 + j12*b21;
                                float v12 = j11*b12 + j12*b22;

                                float ca = v00*v00 + v01*v01 + v02*v02;
                                float cb = v00*v10 + v01*v11 + v02*v12;
                                float cc = v10*v10 + v11*v11 + v12*v12;
                                float max_abs = fmaxf(fabsf(ca), fmaxf(fabsf(cb), fabsf(cc)));
                                if (max_abs > 1e18f) {
                                    float sc_cov = 1e18f / max_abs;
                                    ca *= sc_cov; cb *= sc_cov; cc *= sc_cov;
                                }
                                ca += 0.3f;
                                cc += 0.3f;
                                float det = ca * cc - cb * cb;
                                op_out = op * gain;
                                if (isfinite(ca) && isfinite(cb) && isfinite(cc) && det > 0.0f && (op_out >= (1.0f / 255.0f))) {
                                    float inv_det = 1.0f / det;
                                    c0 = cc * inv_det; c1 = -cb * inv_det; c2 = ca * inv_det;
                                    float det_conic = c0 * c2 - c1 * c1;
                                    if (det_conic > 0.0f) {
                                        float inv_cd = 1.0f / det_conic;
                                        float pwr = logf(op_out * 255.0f);
                                        max_q = 2.0f * pwr;
                                        float ex = sqrtf(max_q * c2 * inv_cd);
                                        float ey = sqrtf(max_q * c0 * inv_cd);
                                        if (isfinite(ex) && isfinite(ey) && (ex >= min_rad_clip || ey >= min_rad_clip) &&
                                            isfinite(mx) && isfinite(my) &&
                                            (mx + ex > 0.0f) && (mx - ex < fw) && (my + ey > 0.0f) && (my - ey < fh))
                                        {
                                            tx0 = max(0, min(tw, (int)floorf((mx - ex) / 16.0f)));
                                            tx1 = max(0, min(tw, (int)ceilf((mx + ex) / 16.0f)));
                                            ty0 = max(0, min(th, (int)floorf((my - ey) / 16.0f)));
                                            ty1 = max(0, min(th, (int)ceilf((my + ey) / 16.0f)));
                                            nt = 0;
                                            for (int ty = ty0; ty < ty1; ++ty) {
                                                for (int tx = tx0; tx < tx1; ++tx) {
                                                    if (ellipse_intersects_tile(mx, my, c0, c1, c2, max_q, tx, ty)) nt++;
                                                }
                                            }
                                            if (nt > 0) {
                                                dep = dev_length_robust(ax, ay, az);
                                                keep = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    unsigned mask = __ballot_sync(0xffffffff, keep);
    int lane = threadIdx.x & 31;
    int warp_cnt = __popc(mask);
    int warp_base = 0;
    if (lane == 0 && warp_cnt > 0) warp_base = atomicAdd(d_M, warp_cnt);
    warp_base = __shfl_sync(0xffffffff, warp_base, 0);

    if (keep) {
        int out_i = warp_base + __popc(mask & ((1u << lane) - 1u));
        out_xy_op_maxq[out_i] = make_float4(mx, my, op_out, max_q);
        out_conic_dep[out_i]  = make_float4(c0, c1, c2, dep);
        out_tile_bbox[out_i]  = ((uint32_t)tx0) | ((uint32_t)ty0 << 8) | ((uint32_t)tx1 << 16) | ((uint32_t)ty1 << 24);
        out_tiles_touched[out_i] = nt;

        float dx = means[gid * 3 + 0] - cam_pos[0];
        float dy = means[gid * 3 + 1] - cam_pos[1];
        float dz = means[gid * 3 + 2] - cam_pos[2];
        float inv_n = rsqrtf(fmaxf(dx*dx + dy*dy + dz*dz, 1e-16f));
        float x = dx * inv_n, y = dy * inv_n, z = dz * inv_n;

        const uint4* s_vec = reinterpret_cast<const uint4*>(sh_fp16 + (size_t)gid * 48);
        __half sh_loc[48];
        #pragma unroll
        for (int v = 0; v < 6; ++v) {
            reinterpret_cast<uint4*>(sh_loc)[v] = s_vec[v];
        }

        const float cc0 = 0.2820948f, f0a = 0.4886025f;
        float z2 = z * z;
        float f0b = -1.0925485f * z, f1a = 0.54627424f;
        float fc1 = x * x - y * y, fs1 = 2.0f * x * y;
        float f0c = -2.285229f * z2 + 0.4570458f, f1b = 1.4453057f * z, f2a = -0.5900436f;
        float fc2 = x * fc1 - y * fs1, fs2 = x * fs1 + y * fc1;

        float w_sh[16] = {
            cc0, -f0a*y, f0a*z, -f0a*x,
            f1a*fs1, f0b*y, (0.9461747f*z2 - 0.31539157f), f0b*x, f1a*fc1,
            f2a*fs2, f1b*fs1, f0c*y, z*(1.8658817f*z2 - 1.119529f), f0c*x, f1b*fc1, f2a*fc2
        };
        float rgb[3];
        #pragma unroll
        for (int ch = 0; ch < 3; ++ch) {
            float acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < 16; ++k) acc += __half2float(sh_loc[k * 3 + ch]) * w_sh[k];
            acc = fminf(fmaxf(acc, -100.0f), 100.0f) + sh_bias;
            if (!isfinite(acc)) acc = 0.0f;
            rgb[ch] = fminf(fmaxf(acc, 0.0f), 100.0f);
        }
        out_col_rng[out_i] = make_float4(rgb[0], rgb[1], rgb[2], rng);
    }
}

__global__ void emit_tile_keys32_kernel(
    int M, int tw, int depth_bits,
    const float4* __restrict__ xy_op_maxq,
    const float4* __restrict__ conic_dep,
    const uint32_t* __restrict__ tile_bbox,
    const int* __restrict__ offsets,
    uint32_t* __restrict__ keys_out,
    int* __restrict__ vals_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    float4 xom = xy_op_maxq[i];
    float4 cd  = conic_dep[i];
    uint32_t bb = tile_bbox[i];
    int tx0 = (int)(bb & 0xffu);
    int ty0 = (int)((bb >> 8) & 0xffu);
    int tx1 = (int)((bb >> 16) & 0xffu);
    int ty1 = (int)((bb >> 24) & 0xffu);

    uint32_t dep_q = (__float_as_uint(cd.w) >> (31 - depth_bits)) & ((1u << depth_bits) - 1u);
    int pos = offsets[i];
    for (int ty = ty0; ty < ty1; ++ty) {
        for (int tx = tx0; tx < tx1; ++tx) {
            if (ellipse_intersects_tile(xom.x, xom.y, cd.x, cd.y, cd.z, xom.w, tx, ty)) {
                uint32_t tile_id = (uint32_t)(ty * tw + tx);
                keys_out[pos] = (tile_id << depth_bits) | dep_q;
                vals_out[pos] = i;
                pos++;
            }
        }
    }
}

__global__ void identify_tile_ranges32_kernel(
    int L, int depth_bits, const uint32_t* __restrict__ keys_sorted, int2* __restrict__ ranges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L) return;
    uint32_t cur_tile = keys_sorted[idx] >> depth_bits;
    if (idx == 0) ranges[cur_tile].x = 0;
    else {
        uint32_t prev_tile = keys_sorted[idx - 1] >> depth_bits;
        if (cur_tile != prev_tile) {
            ranges[prev_tile].y = idx;
            ranges[cur_tile].x = idx;
        }
    }
    if (idx == L - 1) ranges[cur_tile].y = L;
}

__global__ void __launch_bounds__(256) rasterize_front_batch_kernel(
    int width, int height, int tw, int front_bs, int tail_bs,
    const int2* __restrict__ ranges,
    const int* __restrict__ vals_sorted,
    const float4* __restrict__ xy_op_maxq,
    const float4* __restrict__ conic_dep,
    const float4* __restrict__ col_rng,
    int* __restrict__ tile_rem_batches,
    float4* __restrict__ front_cr,
    float* __restrict__ front_T,
    uint8_t* __restrict__ out_rgb_u8,
    float* __restrict__ out_dist,
    float* __restrict__ out_alpha)
{
    int tx = blockIdx.x, ty = blockIdx.y;
    int tile_id = ty * tw + tx;
    int px_i = tx * 16 + threadIdx.x;
    int py_i = ty * 16 + threadIdx.y;
    int t_local = threadIdx.y * 16 + threadIdx.x;
    bool inside = (px_i < width) && (py_i < height);
    bool done = !inside;
    float px = (float)px_i + 0.5f, py = (float)py_i + 0.5f;

    int2 range = ranges[tile_id];
    int start = range.x;
    int full_end = range.y;
    int end = min(full_end, start + front_bs);

    __shared__ float4 s_xom[256];
    __shared__ float4 s_con[256];
    __shared__ float4 s_col[256];

    float T = 1.0f;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f, acc_rng = 0.0f;

    int n_sub = (end - start + 255) >> 8;
    for (int b = 0; b < n_sub; ++b) {
        if (__syncthreads_and(done)) break;
        int load_idx = start + (b << 8) + t_local;
        if (load_idx < end) {
            int sid = vals_sorted[load_idx];
            s_xom[t_local] = xy_op_maxq[sid];
            s_con[t_local] = conic_dep[sid];
            s_col[t_local] = col_rng[sid];
        }
        __syncthreads();
        if (!done) {
            int batch_sz = min(256, end - start - (b << 8));
            for (int j = 0; j < batch_sz; ++j) {
                float4 xom = s_xom[j];
                float dx = xom.x - px, dy = xom.y - py;
                float4 con = s_con[j];
                float sigma = 0.5f * (con.x * dx * dx + con.z * dy * dy) + con.y * dx * dy;
                if (sigma < 0.0f) continue;
                float alpha = fminf(0.999f, xom.z * __expf(-sigma));
                if (alpha < (1.0f / 255.0f)) continue;
                float next_T = T * (1.0f - alpha);
                if (next_T <= 1e-4f) { done = true; break; }
                float vis = alpha * T;
                float4 cr = s_col[j];
                acc_r += vis * cr.x; acc_g += vis * cr.y; acc_b += vis * cr.z; acc_rng += vis * cr.w;
                T = next_T;
            }
        }
    }

    bool all_done = __syncthreads_and(done);
    int rem = all_done ? 0 : max(0, full_end - end);
    int rem_b = (rem > 0) ? ((rem + tail_bs - 1) / tail_bs) : 0;
    if (t_local == 0) tile_rem_batches[tile_id] = rem_b;

    if (inside) {
        int pix = py_i * width + px_i;
        if (rem_b == 0) {
            float alp = 1.0f - T;
            out_alpha[pix] = alp;
            out_dist[pix] = (alp > 1e-8f) ? (acc_rng / alp) : INFINITY;
            out_rgb_u8[pix * 3 + 0] = (uint8_t)lrintf(fminf(fmaxf(acc_r, 0.0f), 1.0f) * 255.0f);
            out_rgb_u8[pix * 3 + 1] = (uint8_t)lrintf(fminf(fmaxf(acc_g, 0.0f), 1.0f) * 255.0f);
            out_rgb_u8[pix * 3 + 2] = (uint8_t)lrintf(fminf(fmaxf(acc_b, 0.0f), 1.0f) * 255.0f);
        } else {
            front_cr[pix] = make_float4(acc_r, acc_g, acc_b, acc_rng);
            front_T[pix] = T;
        }
    }
}

__global__ void build_tail_desc_kernel(
    int n_tiles, int front_bs, int tail_bs,
    const int2* __restrict__ ranges,
    const int* __restrict__ tile_rem_batches,
    const int* __restrict__ tile_rem_offs,
    int4* __restrict__ batch_desc)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tiles) return;
    int nb = tile_rem_batches[t];
    if (nb == 0) return;
    int base = tile_rem_offs[t];
    int s0 = ranges[t].x + front_bs;
    int e0 = ranges[t].y;
    for (int k = 0; k < nb; ++k) {
        int s = s0 + k * tail_bs;
        int e = min(e0, s + tail_bs);
        batch_desc[base + k] = make_int4(t, s, e, nb);
    }
}

__global__ void __launch_bounds__(256) rasterize_tail_batch_kernel(
    int total_batches, int width, int height, int tw,
    const int4* __restrict__ batch_desc,
    const int* __restrict__ vals_sorted,
    const float4* __restrict__ xy_op_maxq,
    const float4* __restrict__ conic_dep,
    const float4* __restrict__ col_rng,
    const float* __restrict__ front_T,
    float4* __restrict__ batch_cr,
    float* __restrict__ batch_T)
{
    int bid = blockIdx.x;
    if (bid >= total_batches) return;
    int4 desc = batch_desc[bid];
    int tile_id = desc.x, start = desc.y, end = desc.z;
    int ty = tile_id / tw, tx = tile_id - ty * tw;
    int px_i = tx * 16 + threadIdx.x, py_i = ty * 16 + threadIdx.y;
    int t_local = threadIdx.y * 16 + threadIdx.x;
    bool inside = (px_i < width) && (py_i < height);
    bool done = (!inside) || (front_T[py_i * width + px_i] <= 1e-4f);
    float px = (float)px_i + 0.5f, py = (float)py_i + 0.5f;

    __shared__ float4 s_xom[256];
    __shared__ float4 s_con[256];
    __shared__ float4 s_col[256];

    float T = 1.0f;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f, acc_rng = 0.0f;

    int n_sub = (end - start + 255) >> 8;
    for (int b = 0; b < n_sub; ++b) {
        if (__syncthreads_and(done)) break;
        int load_idx = start + (b << 8) + t_local;
        if (load_idx < end) {
            int sid = vals_sorted[load_idx];
            s_xom[t_local] = xy_op_maxq[sid];
            s_con[t_local] = conic_dep[sid];
            s_col[t_local] = col_rng[sid];
        }
        __syncthreads();
        if (!done) {
            int batch_sz = min(256, end - start - (b << 8));
            for (int j = 0; j < batch_sz; ++j) {
                float4 xom = s_xom[j];
                float dx = xom.x - px, dy = xom.y - py;
                float4 con = s_con[j];
                float sigma = 0.5f * (con.x * dx * dx + con.z * dy * dy) + con.y * dx * dy;
                if (sigma < 0.0f) continue;
                float alpha = fminf(0.999f, xom.z * __expf(-sigma));
                if (alpha < (1.0f / 255.0f)) continue;
                float next_T = T * (1.0f - alpha);
                if (next_T <= 1e-4f) { done = true; break; }
                float vis = alpha * T;
                float4 cr = s_col[j];
                acc_r += vis * cr.x; acc_g += vis * cr.y; acc_b += vis * cr.z; acc_rng += vis * cr.w;
                T = next_T;
            }
        }
    }
    if (inside) {
        size_t off = (size_t)bid * 256 + t_local;
        batch_cr[off] = make_float4(acc_r, acc_g, acc_b, acc_rng);
        batch_T[off]  = T;
    }
}

__global__ void __launch_bounds__(256) composite_tail_tiles_kernel(
    int width, int height, int tw,
    const int* __restrict__ tile_rem_batches,
    const int* __restrict__ tile_rem_offs,
    const float4* __restrict__ front_cr,
    const float* __restrict__ front_T,
    const float4* __restrict__ batch_cr,
    const float* __restrict__ batch_T,
    uint8_t* __restrict__ out_rgb_u8,
    float* __restrict__ out_dist,
    float* __restrict__ out_alpha)
{
    int tx = blockIdx.x, ty = blockIdx.y;
    int tile_id = ty * tw + tx;
    int nb = tile_rem_batches[tile_id];
    if (nb == 0) return;

    int px_i = tx * 16 + threadIdx.x, py_i = ty * 16 + threadIdx.y;
    if (px_i >= width || py_i >= height) return;
    int t_local = threadIdx.y * 16 + threadIdx.x;
    int pix = py_i * width + px_i;

    float4 fc = front_cr[pix];
    float T = front_T[pix];
    float acc_r = fc.x, acc_g = fc.y, acc_b = fc.z, acc_rng = fc.w;
    if (T > 1e-4f) {
        int base = tile_rem_offs[tile_id];
        for (int k = 0; k < nb; ++k) {
            size_t off = (size_t)(base + k) * 256 + t_local;
            float4 cr = batch_cr[off];
            float bk_T = batch_T[off];
            acc_r += T * cr.x; acc_g += T * cr.y; acc_b += T * cr.z; acc_rng += T * cr.w;
            T *= bk_T;
            if (T <= 1e-4f) break;
        }
    }
    float alp = 1.0f - T;
    out_alpha[pix] = alp;
    out_dist[pix] = (alp > 1e-8f) ? (acc_rng / alp) : INFINITY;
    out_rgb_u8[pix * 3 + 0] = (uint8_t)lrintf(fminf(fmaxf(acc_r, 0.0f), 1.0f) * 255.0f);
    out_rgb_u8[pix * 3 + 1] = (uint8_t)lrintf(fminf(fmaxf(acc_g, 0.0f), 1.0f) * 255.0f);
    out_rgb_u8[pix * 3 + 2] = (uint8_t)lrintf(fminf(fmaxf(acc_b, 0.0f), 1.0f) * 255.0f);
}

struct WorkspaceV5 {
    int cap_V = 0, cap_K = 0, cap_M = 0, cap_L = 0, cap_tiles = 0, cap_pix = 0, cap_batches = 0;
    size_t scan_v_bytes = 0, scan_m_bytes = 0, sort_bytes = 0, scan_t_bytes = 0;
    int *d_n_vis = nullptr, *d_vis_ids = nullptr, *d_vis_cnt = nullptr, *d_vis_off = nullptr;
    int *d_cand = nullptr, *d_M = nullptr;
    float4 *d_xom = nullptr, *d_con = nullptr, *d_col = nullptr;
    uint32_t *d_bbox = nullptr;
    int *d_nt = nullptr, *d_off = nullptr;
    uint32_t *d_keys_in = nullptr, *d_keys_out = nullptr;
    int *d_vals_in = nullptr, *d_vals_out = nullptr;
    int2 *d_ranges = nullptr;
    int *d_tile_rem = nullptr, *d_tile_roff = nullptr;
    float4 *d_front_cr = nullptr;
    float *d_front_T = nullptr;
    int4 *d_bdesc = nullptr;
    float4 *d_bcr = nullptr;
    float *d_bT = nullptr;
    void *d_scan_v_tmp = nullptr, *d_scan_m_tmp = nullptr, *d_sort_tmp = nullptr, *d_scan_t_tmp = nullptr;

    void ensure_V(int V) {
        if (V > cap_V) {
            if (d_vis_ids) { cudaFree(d_vis_ids); cudaFree(d_vis_cnt); cudaFree(d_vis_off); }
            cap_V = max(V, (int)(V*1.1f)+1024);
            cudaMalloc(&d_vis_ids, cap_V*sizeof(int)); cudaMalloc(&d_vis_cnt, (cap_V+1)*sizeof(int)); cudaMalloc(&d_vis_off, (cap_V+1)*sizeof(int));
            size_t need = 0; cub::DeviceScan::ExclusiveSum(nullptr, need, d_vis_cnt, d_vis_off, cap_V+1);
            if (need > scan_v_bytes) { if (d_scan_v_tmp) cudaFree(d_scan_v_tmp); scan_v_bytes = need; cudaMalloc(&d_scan_v_tmp, scan_v_bytes); }
        }
        if (!d_n_vis) cudaMalloc(&d_n_vis, sizeof(int));
        if (!d_M) cudaMalloc(&d_M, sizeof(int));
    }
    void ensure_K(int K) {
        if (K > cap_K) { if (d_cand) cudaFree(d_cand); cap_K = max(K, (int)(K*1.25f)+1024); cudaMalloc(&d_cand, cap_K*sizeof(int)); }
    }
    void ensure_M(int M) {
        if (M > cap_M) {
            if (d_xom) { cudaFree(d_xom); cudaFree(d_con); cudaFree(d_col); cudaFree(d_bbox); cudaFree(d_nt); cudaFree(d_off); }
            cap_M = max(M, (int)(M*1.25f)+1024);
            cudaMalloc(&d_xom, cap_M*sizeof(float4)); cudaMalloc(&d_con, cap_M*sizeof(float4)); cudaMalloc(&d_col, cap_M*sizeof(float4));
            cudaMalloc(&d_bbox, cap_M*sizeof(uint32_t)); cudaMalloc(&d_nt, (cap_M+1)*sizeof(int)); cudaMalloc(&d_off, (cap_M+1)*sizeof(int));
            size_t need = 0; cub::DeviceScan::ExclusiveSum(nullptr, need, d_nt, d_off, cap_M+1);
            if (need > scan_m_bytes) { if (d_scan_m_tmp) cudaFree(d_scan_m_tmp); scan_m_bytes = need; cudaMalloc(&d_scan_m_tmp, scan_m_bytes); }
        }
    }
    void ensure_L(int L) {
        if (L > cap_L) {
            if (d_keys_in) { cudaFree(d_keys_in); cudaFree(d_keys_out); cudaFree(d_vals_in); cudaFree(d_vals_out); }
            cap_L = max(L, (int)(L*1.25f)+4096);
            cudaMalloc(&d_keys_in, cap_L*sizeof(uint32_t)); cudaMalloc(&d_keys_out, cap_L*sizeof(uint32_t));
            cudaMalloc(&d_vals_in, cap_L*sizeof(int)); cudaMalloc(&d_vals_out, cap_L*sizeof(int));
            size_t need = 0; cub::DeviceRadixSort::SortPairs(nullptr, need, d_keys_in, d_keys_out, d_vals_in, d_vals_out, cap_L, 0, 32);
            if (need > sort_bytes) { if (d_sort_tmp) cudaFree(d_sort_tmp); sort_bytes = need; cudaMalloc(&d_sort_tmp, sort_bytes); }
        }
    }
    void ensure_frame(int n_tiles, int W, int H) {
        if (n_tiles > cap_tiles) {
            if (d_ranges) { cudaFree(d_ranges); cudaFree(d_tile_rem); cudaFree(d_tile_roff); }
            cap_tiles = n_tiles;
            cudaMalloc(&d_ranges, cap_tiles*sizeof(int2));
            cudaMalloc(&d_tile_rem, (cap_tiles+1)*sizeof(int));
            cudaMalloc(&d_tile_roff, (cap_tiles+1)*sizeof(int));
            size_t need = 0; cub::DeviceScan::ExclusiveSum(nullptr, need, d_tile_rem, d_tile_roff, cap_tiles+1);
            if (need > scan_t_bytes) { if (d_scan_t_tmp) cudaFree(d_scan_t_tmp); scan_t_bytes = need; cudaMalloc(&d_scan_t_tmp, scan_t_bytes); }
        }
        int np = W * H;
        if (np > cap_pix) {
            if (d_front_cr) { cudaFree(d_front_cr); cudaFree(d_front_T); }
            cap_pix = np;
            cudaMalloc(&d_front_cr, cap_pix*sizeof(float4));
            cudaMalloc(&d_front_T, cap_pix*sizeof(float));
        }
    }
    void ensure_batches(int B) {
        if (B > cap_batches) {
            if (d_bdesc) { cudaFree(d_bdesc); cudaFree(d_bcr); cudaFree(d_bT); }
            cap_batches = max(B, (int)(B*1.25f)+256);
            cudaMalloc(&d_bdesc, cap_batches*sizeof(int4));
            cudaMalloc(&d_bcr, (size_t)cap_batches*256*sizeof(float4));
            cudaMalloc(&d_bT,  (size_t)cap_batches*256*sizeof(float));
        }
    }
};
static thread_local WorkspaceV5 g_v5;

std::vector<int> fused_brush_render(
    int zone_id,
    torch::Tensor v_centers, torch::Tensor v_radii, torch::Tensor v_starts, torch::Tensor v_counts, torch::Tensor v_zmask,
    torch::Tensor g_zmask,
    torch::Tensor means, torch::Tensor quats, torch::Tensor scales, torch::Tensor opacities, torch::Tensor sh_fp16,
    torch::Tensor view_3x4, torch::Tensor cam_pos,
    int width, int height, double fx, double fy, double cx, double cy, double skew,
    double sh_bias, double min_rad_clip, int front_bs, int tail_bs,
    torch::Tensor out_rgb_u8, torch::Tensor out_dist, torch::Tensor out_alpha)
{
    int V = v_centers.size(0);
    int tw = (width + 15) / 16, th = (height + 15) / 16, n_tiles = tw * th;
    g_v5.ensure_V(V);
    g_v5.ensure_frame(n_tiles, width, height);

    uint32_t zone_bit = (zone_id >= 0) ? (1u << zone_id) : 0u;
    cudaMemsetAsync(g_v5.d_n_vis, 0, sizeof(int));
    select_voxels_pvs_kernel<<<(V + 255) / 256, 256>>>(
        V, zone_bit, v_centers.data_ptr<float>(), v_radii.data_ptr<float>(), v_counts.data_ptr<int>(),
        reinterpret_cast<const uint32_t*>(v_zmask.data_ptr<int32_t>()),
        view_3x4.data_ptr<float>(), width, height, (float)fx, (float)fy, (float)cx, (float)cy,
        g_v5.d_n_vis, g_v5.d_vis_ids, g_v5.d_vis_cnt);

    int n_vis_v = 0; cudaMemcpy(&n_vis_v, g_v5.d_n_vis, sizeof(int), cudaMemcpyDeviceToHost);
    if (n_vis_v == 0) { out_rgb_u8.zero_(); out_dist.fill_(INFINITY); out_alpha.zero_(); return {0, 0, 0, 0}; }

    cudaMemsetAsync(g_v5.d_vis_cnt + n_vis_v, 0, sizeof(int));
    size_t svb = g_v5.scan_v_bytes;
    cub::DeviceScan::ExclusiveSum(g_v5.d_scan_v_tmp, svb, g_v5.d_vis_cnt, g_v5.d_vis_off, n_vis_v + 1);
    int K = 0; cudaMemcpy(&K, g_v5.d_vis_off + n_vis_v, sizeof(int), cudaMemcpyDeviceToHost);
    if (K == 0) { out_rgb_u8.zero_(); out_dist.fill_(INFINITY); out_alpha.zero_(); return {0, 0, 0, 0}; }

    g_v5.ensure_K(K); g_v5.ensure_M(K);
    expand_voxels_kernel<<<n_vis_v, 256>>>(g_v5.d_vis_ids, v_starts.data_ptr<int>(), v_counts.data_ptr<int>(), g_v5.d_vis_off, n_vis_v, g_v5.d_cand);

    cudaMemsetAsync(g_v5.d_M, 0, sizeof(int));
    brush_project_v5_kernel<<<(K + 255) / 256, 256>>>(
        g_v5.d_cand, K, zone_bit, reinterpret_cast<const uint32_t*>(g_zmask.data_ptr<int32_t>()),
        means.data_ptr<float>(), quats.data_ptr<float>(), scales.data_ptr<float>(), opacities.data_ptr<float>(),
        reinterpret_cast<const __half*>(sh_fp16.data_ptr<at::Half>()),
        view_3x4.data_ptr<float>(), cam_pos.data_ptr<float>(),
        width, height, tw, th, (float)fx, (float)fy, (float)cx, (float)cy, (float)skew,
        (float)sh_bias, (float)min_rad_clip,
        g_v5.d_M, g_v5.d_xom, g_v5.d_con, g_v5.d_col, g_v5.d_bbox, g_v5.d_nt);

    int M = 0; cudaMemcpy(&M, g_v5.d_M, sizeof(int), cudaMemcpyDeviceToHost);
    if (M == 0) { out_rgb_u8.zero_(); out_dist.fill_(INFINITY); out_alpha.zero_(); return {K, 0, 0, 0}; }

    cudaMemsetAsync(g_v5.d_nt + M, 0, sizeof(int));
    size_t smb = g_v5.scan_m_bytes;
    cub::DeviceScan::ExclusiveSum(g_v5.d_scan_m_tmp, smb, g_v5.d_nt, g_v5.d_off, M + 1);
    int L = 0; cudaMemcpy(&L, g_v5.d_off + M, sizeof(int), cudaMemcpyDeviceToHost);
    if (L == 0) { out_rgb_u8.zero_(); out_dist.fill_(INFINITY); out_alpha.zero_(); return {K, M, 0, 0}; }

    int tile_bits = 1; while ((1 << tile_bits) < n_tiles) tile_bits++;
    int depth_bits = 32 - tile_bits;
    g_v5.ensure_L(L);
    emit_tile_keys32_kernel<<<(M + 255) / 256, 256>>>(
        M, tw, depth_bits, g_v5.d_xom, g_v5.d_con, g_v5.d_bbox, g_v5.d_off, g_v5.d_keys_in, g_v5.d_vals_in);

    size_t sort_b = g_v5.sort_bytes;
    cub::DeviceRadixSort::SortPairs(
        g_v5.d_sort_tmp, sort_b, g_v5.d_keys_in, g_v5.d_keys_out, g_v5.d_vals_in, g_v5.d_vals_out, L, 0, 32);

    cudaMemsetAsync(g_v5.d_ranges, 0, n_tiles * sizeof(int2));
    identify_tile_ranges32_kernel<<<(L + 255) / 256, 256>>>(L, depth_bits, g_v5.d_keys_out, g_v5.d_ranges);

    rasterize_front_batch_kernel<<<dim3(tw, th), dim3(16, 16)>>>(
        width, height, tw, front_bs, tail_bs, g_v5.d_ranges, g_v5.d_vals_out,
        g_v5.d_xom, g_v5.d_con, g_v5.d_col, g_v5.d_tile_rem, g_v5.d_front_cr, g_v5.d_front_T,
        out_rgb_u8.data_ptr<uint8_t>(), out_dist.data_ptr<float>(), out_alpha.data_ptr<float>());

    cudaMemsetAsync(g_v5.d_tile_rem + n_tiles, 0, sizeof(int));
    size_t stb = g_v5.scan_t_bytes;
    cub::DeviceScan::ExclusiveSum(g_v5.d_scan_t_tmp, stb, g_v5.d_tile_rem, g_v5.d_tile_roff, n_tiles + 1);
    int tail_batches = 0; cudaMemcpy(&tail_batches, g_v5.d_tile_roff + n_tiles, sizeof(int), cudaMemcpyDeviceToHost);

    if (tail_batches > 0) {
        g_v5.ensure_batches(tail_batches);
        build_tail_desc_kernel<<<(n_tiles + 255) / 256, 256>>>(n_tiles, front_bs, tail_bs, g_v5.d_ranges, g_v5.d_tile_rem, g_v5.d_tile_roff, g_v5.d_bdesc);
        rasterize_tail_batch_kernel<<<tail_batches, dim3(16, 16)>>>(
            tail_batches, width, height, tw, g_v5.d_bdesc, g_v5.d_vals_out,
            g_v5.d_xom, g_v5.d_con, g_v5.d_col, g_v5.d_front_T, g_v5.d_bcr, g_v5.d_bT);
        composite_tail_tiles_kernel<<<dim3(tw, th), dim3(16, 16)>>>(
            width, height, tw, g_v5.d_tile_rem, g_v5.d_tile_roff, g_v5.d_front_cr, g_v5.d_front_T,
            g_v5.d_bcr, g_v5.d_bT, out_rgb_u8.data_ptr<uint8_t>(), out_dist.data_ptr<float>(), out_alpha.data_ptr<float>());
    }
    return {K, M, L, tail_batches};
}
"""

_CPP_SRC = (
    "std::vector<int> fused_brush_render("
    "int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
    "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
    "torch::Tensor, torch::Tensor, torch::Tensor, int, int, double, double, double, "
    "double, double, double, double, int, int, torch::Tensor, torch::Tensor, torch::Tensor);"
)


def _get_extension():
    global _EXT_MODULE, _EXT_FAILED
    if _EXT_MODULE is not None:
        return _EXT_MODULE
    if _EXT_FAILED:
        return None
    with _EXT_LOCK:
        if _EXT_MODULE is not None:
            return _EXT_MODULE
        if _EXT_FAILED:
            return None
        if not torch.cuda.is_available():
            _EXT_FAILED = True
            return None
        if shutil.which("ninja") is None:
            _EXT_FAILED = True
            return None
        try:
            from torch.utils.cpp_extension import load_inline

            _EXT_MODULE = load_inline(
                name="lingbot_fused_brush_v5",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["fused_brush_render"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
            )
            return _EXT_MODULE
        except Exception as exc:
            _EXT_FAILED = True
            warnings.warn(
                f"Fused Brush CUDA backend unavailable ({exc}); falling back to gsplat.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None


def is_available() -> bool:
    """Return True if the fused Brush CUDA extension can be loaded."""
    return _get_extension() is not None


class FusedBrushScene:
    """GPU-resident voxel-indexed Gaussian scene for fused Brush-aligned rendering."""

    def __init__(
        self,
        xyz: np.ndarray,
        quats: np.ndarray,
        scales: np.ndarray,
        opacity: np.ndarray,
        sh_colors: np.ndarray,
        device: str = "cuda:0",
        voxel_size: float = 0.50,
        small_radius_threshold: float = 0.35,
    ) -> None:
        ext = _get_extension()
        if ext is None:
            raise RuntimeError("Fused Brush CUDA extension is not available.")
        self._ext = ext
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)

        keep = opacity >= ALPHA_CUTOFF
        self._keep_mask = keep
        xyz_t = torch.as_tensor(xyz[keep], device=self.device, dtype=torch.float32).contiguous()
        quats_t = torch.as_tensor(quats[keep], device=self.device, dtype=torch.float32).contiguous()
        scales_t = torch.as_tensor(scales[keep], device=self.device, dtype=torch.float32).contiguous()
        opac_t = torch.as_tensor(opacity[keep], device=self.device, dtype=torch.float32).contiguous()
        sh_fp16_t = torch.as_tensor(sh_colors[keep], device=self.device, dtype=torch.float16).contiguous()

        with torch.no_grad():
            supp_pow = torch.log(opac_t * 255.0).clamp_min(0.0)
            r_brush = torch.sqrt(2.0 * supp_pow) * scales_t.amax(dim=-1) + 0.02
            is_small = r_brush <= small_radius_threshold
            coords = torch.floor(xyz_t / voxel_size).to(torch.int64)
            c = coords - coords.min(dim=0).values
            raw_keys = c[:, 0] * 1000000 + c[:, 1] * 1000 + c[:, 2]
            keys = torch.where(
                is_small,
                raw_keys,
                raw_keys.max() + 1 + torch.arange(len(xyz_t), device=self.device, dtype=torch.int64),
            )
            order = torch.argsort(keys)
            self._order_cpu = order.cpu()

            self.xyz = xyz_t[order].contiguous()
            self.quats = quats_t[order].contiguous()
            self.scales = scales_t[order].contiguous()
            self.opac = opac_t[order].contiguous()
            self.sh_fp16 = sh_fp16_t[order].contiguous()
            r_ord = r_brush[order]

            uniq_k, counts = torch.unique_consecutive(keys[order], return_counts=True)
            v_num = len(uniq_k)
            self.v_counts = counts.to(torch.int32).contiguous()
            v_ends = torch.cumsum(self.v_counts, dim=0, dtype=torch.int32)
            self.v_starts = (v_ends - self.v_counts).contiguous()
            g_vid = torch.repeat_interleave(
                torch.arange(v_num, device=self.device, dtype=torch.int32), counts
            )
            self.v_centers = torch.zeros((v_num, 3), dtype=torch.float32, device=self.device)
            self.v_centers.index_add_(0, g_vid, self.xyz)
            self.v_centers /= counts[:, None].float()
            dist_each = (self.xyz - self.v_centers[g_vid]).norm(dim=-1) + r_ord
            self.v_radii = torch.zeros(v_num, dtype=torch.float32, device=self.device)
            self.v_radii.scatter_reduce_(0, g_vid.long(), dist_each, reduce="amax", include_self=False)
            self._g_vid_cpu = g_vid.cpu()

        self._zone_centers: Optional[np.ndarray] = None
        self._g_zmask = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._v_zmask = torch.zeros(1, dtype=torch.int32, device=self.device)

        self._buf_shape: Optional[tuple[int, int]] = None
        self._out_rgb = None
        self._out_dist = None
        self._out_alpha = None
        self._pin_rgb = None
        self._pin_dist = None
        self._pin_alpha = None

    def set_pvs_mask(
        self,
        zone_centers: np.ndarray,
        zone_max: torch.Tensor,
        tau: float = 0.002,
    ) -> None:
        """Attach an optional lightweight 32-bit zone PVS bitmask (<=32 zones)."""
        n_zones = int(zone_centers.shape[0])
        if n_zones > 32:
            raise ValueError("Bitmask PVS supports at most 32 zones.")
        zm = zone_max[:, self._keep_mask][:, self._order_cpu]
        g_mask_np = np.zeros(len(self.xyz), dtype=np.uint32)
        v_mask_np = np.zeros(len(self.v_centers), dtype=np.uint32)
        g_vid_np = self._g_vid_cpu.numpy()
        for zi in range(n_zones):
            m_z = (zm[zi] >= tau).numpy()
            g_mask_np[m_z] |= np.uint32(1) << zi
            v_mask_np[np.unique(g_vid_np[m_z])] |= np.uint32(1) << zi
        self._zone_centers = np.asarray(zone_centers, dtype=np.float32)
        self._g_zmask = torch.from_numpy(g_mask_np.view(np.int32)).to(self.device).contiguous()
        self._v_zmask = torch.from_numpy(v_mask_np.view(np.int32)).to(self.device).contiguous()

    def clear_pvs_mask(self) -> None:
        self._zone_centers = None
        self._g_zmask = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._v_zmask = torch.zeros(1, dtype=torch.int32, device=self.device)

    def _ensure_buffers(self, width: int, height: int) -> None:
        if self._buf_shape == (height, width):
            return
        self._buf_shape = (height, width)
        self._out_rgb = torch.empty((height, width, 3), dtype=torch.uint8, device=self.device)
        self._out_dist = torch.empty((height, width), dtype=torch.float32, device=self.device)
        self._out_alpha = torch.empty((height, width), dtype=torch.float32, device=self.device)
        self._pin_rgb = torch.empty((height, width, 3), dtype=torch.uint8, pin_memory=True)
        self._pin_dist = torch.empty((height, width), dtype=torch.float32, pin_memory=True)
        self._pin_alpha = torch.empty((height, width), dtype=torch.float32, pin_memory=True)

    @torch.inference_mode()
    def render(
        self,
        camera_to_world: np.ndarray,
        k: np.ndarray,
        width: int,
        height: int,
        *,
        min_rad_clip: float = 0.0,
        front_bs: int = 1024,
        tail_bs: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render Brush-aligned RGB (uint8), radial distance (float32), and alpha (float32)."""
        torch.cuda.set_device(self.device)
        width, height = int(width), int(height)
        self._ensure_buffers(width, height)

        k_arr = np.asarray(k, dtype=np.float64)
        intr = PinholeIntrinsics(
            fov_x=2.0 * math.atan(width * 0.5 / k_arr[0, 0]),
            fov_y=2.0 * math.atan(height * 0.5 / k_arr[1, 1]),
            center_uv=(float(k_arr[0, 2]) / width, float(k_arr[1, 2]) / height),
            skew=float(k_arr[0, 1]),
        )
        fx, fy, cx, cy = _camera_params(width, height, intr)

        c2w = np.asarray(camera_to_world, dtype=np.float64)
        w2c = c2w_to_w2c(c2w)
        view_3x4 = torch.tensor(np.ascontiguousarray(w2c[:3, :]), device=self.device, dtype=torch.float32)
        cam_pos = torch.tensor(np.ascontiguousarray(c2w[:3, 3]), device=self.device, dtype=torch.float32)

        zone_id = -1
        if self._zone_centers is not None:
            xy = c2w[:2, 3].astype(np.float32)
            zone_id = int(np.argmin(np.linalg.norm(self._zone_centers - xy[None, :], axis=-1)))

        self._ext.fused_brush_render(
            zone_id,
            self.v_centers,
            self.v_radii,
            self.v_starts,
            self.v_counts,
            self._v_zmask,
            self._g_zmask,
            self.xyz,
            self.quats,
            self.scales,
            self.opac,
            self.sh_fp16,
            view_3x4,
            cam_pos,
            width,
            height,
            float(fx),
            float(fy),
            float(cx),
            float(cy),
            float(intr.skew),
            float(SH_INRIA_BIAS),
            float(min_rad_clip),
            int(front_bs),
            int(tail_bs),
            self._out_rgb,
            self._out_dist,
            self._out_alpha,
        )
        self._pin_rgb.copy_(self._out_rgb, non_blocking=True)
        self._pin_dist.copy_(self._out_dist, non_blocking=True)
        self._pin_alpha.copy_(self._out_alpha, non_blocking=True)
        torch.cuda.synchronize(self.device)
        return (
            self._pin_rgb.numpy().copy(),
            self._pin_dist.numpy().copy(),
            self._pin_alpha.numpy().copy(),
        )

    def close(self) -> None:
        self.xyz = None
        self.quats = None
        self.scales = None
        self.opac = None
        self.sh_fp16 = None
        self._out_rgb = None
        self._out_dist = None
        self._out_alpha = None
        self._buf_shape = None
