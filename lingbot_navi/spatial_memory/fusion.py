from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .observations import ObservationBatch

_QUALIFYING, _FALLBACK = 0, 1


@dataclass(frozen=True)
class MemoryResult:
    features: np.ndarray
    positions: np.ndarray
    rgb: np.ndarray
    counts: np.ndarray
    depths: np.ndarray | None = None
    frame_names: list[str] | None = None
    frame_poses: np.ndarray | None = None
    qualifying_voxels: int | None = None


class BestObservationAccumulator:
    """Keep one real observation per voxel with two priority reservoirs."""

    def __init__(
        self,
        *,
        voxel_size: float,
        feature_dim: int,
        max_voxels: int,
        preferred_min_depth: float,
        preferred_max_depth: float,
        generator: torch.Generator,
    ):
        self.voxel_size = float(voxel_size)
        self.feature_dim = int(feature_dim)
        self.max_voxels = int(max_voxels)
        self.preferred_min_depth = float(preferred_min_depth)
        self.preferred_max_depth = float(preferred_max_depth)
        self.generator = generator
        self._features = torch.zeros(2, max_voxels, feature_dim)
        self._positions = torch.zeros(2, max_voxels, 3)
        self._rgb = torch.zeros(2, max_voxels, 3)
        self._depths = torch.zeros(2, max_voxels)
        self._priorities = torch.full((2, max_voxels), float("inf"))
        self._counts = torch.zeros(max_voxels, 2, dtype=torch.long)
        self._has = torch.zeros(max_voxels, 2, dtype=torch.bool)
        self._frame_poses = torch.full((2, max_voxels, 7), float("nan"))
        self._frame_names: list[list[str | None]] = [
            [None] * max_voxels,
            [None] * max_voxels,
        ]
        self._hash_table: dict[tuple[int, int, int], int] = {}
        self._num_voxels = 0

    @property
    def num_voxels(self) -> int:
        return self._num_voxels

    @staticmethod
    def _winner(
        absolute_indices: torch.Tensor,
        stream_mask: torch.Tensor,
        priorities: torch.Tensor,
    ) -> tuple[int, torch.Tensor] | None:
        candidates = torch.nonzero(stream_mask, as_tuple=False).squeeze(-1)
        if candidates.numel() == 0:
            return None
        candidate_priorities = priorities[candidates]
        winner_position = candidates[int(torch.argmin(candidate_priorities).item())]
        return (
            int(absolute_indices[int(winner_position.item())].item()),
            priorities[int(winner_position.item())],
        )

    def _store_winner(
        self,
        stream: int,
        slot: int,
        observation_index: int,
        priority: torch.Tensor,
        observations: ObservationBatch,
        frame_name: str,
        frame_pose: torch.Tensor | None,
    ) -> None:
        self._features[stream, slot] = observations.features[observation_index]
        self._positions[stream, slot] = observations.positions[observation_index]
        if observations.rgb is not None:
            self._rgb[stream, slot] = observations.rgb[observation_index]
        self._depths[stream, slot] = observations.depths[observation_index]
        self._priorities[stream, slot] = priority
        self._has[slot, stream] = True
        self._frame_names[stream][slot] = frame_name
        if frame_pose is not None:
            self._frame_poses[stream, slot] = frame_pose

    @torch.no_grad()
    def update(
        self,
        observations: ObservationBatch,
        *,
        frame_name: str = "",
        frame_pose: torch.Tensor | None = None,
    ) -> None:
        observation_count = observations.features.shape[0]
        if observation_count == 0:
            return
        if frame_pose is not None:
            frame_pose = torch.as_tensor(frame_pose, dtype=torch.float32)
            if frame_pose.shape != (7,):
                raise ValueError(
                    f"frame_pose must have shape (7,), got {tuple(frame_pose.shape)}"
                )
        priorities = torch.rand(observation_count, generator=self.generator)
        voxel_indices = torch.floor(observations.positions / self.voxel_size).long()
        unique_indices, inverse = torch.unique(
            voxel_indices, dim=0, return_inverse=True
        )
        buckets: list[list[int]] = [[] for _ in range(unique_indices.shape[0])]
        for observation_index, bucket_index in enumerate(inverse.tolist()):
            buckets[int(bucket_index)].append(observation_index)

        for bucket_index, members in enumerate(buckets):
            indices = torch.tensor(members, dtype=torch.long)
            depths = observations.depths[indices]
            qualifying = (depths > self.preferred_min_depth) & (
                depths < self.preferred_max_depth
            )
            key = tuple(int(value) for value in unique_indices[bucket_index].tolist())
            slot = self._hash_table.get(key)
            if slot is None:
                if self._num_voxels >= self.max_voxels:
                    continue
                slot = self._num_voxels
                self._hash_table[key] = slot
                self._num_voxels += 1

            for stream, stream_mask in (
                (_QUALIFYING, qualifying),
                (_FALLBACK, ~qualifying),
            ):
                stream_count = int(stream_mask.sum().item())
                if stream_count == 0:
                    continue
                self._counts[slot, stream] += stream_count
                winner = self._winner(indices, stream_mask, priorities[indices])
                assert winner is not None
                observation_index, priority = winner
                if (not bool(self._has[slot, stream])) or bool(
                    priority < self._priorities[stream, slot]
                ):
                    self._store_winner(
                        stream,
                        slot,
                        observation_index,
                        priority,
                        observations,
                        frame_name,
                        frame_pose,
                    )

    def finalize(self) -> MemoryResult:
        count = self._num_voxels
        rows = torch.arange(count)
        streams = torch.where(self._has[:count, _QUALIFYING], _QUALIFYING, _FALLBACK)
        names = [
            self._frame_names[int(streams[index])][index] or ""
            for index in range(count)
        ]
        selected_poses = self._frame_poses[streams, rows]
        frame_poses = (
            selected_poses.numpy()
            if count > 0 and bool(torch.isfinite(selected_poses).all())
            else None
        )
        return MemoryResult(
            features=self._features[streams, rows].numpy(),
            positions=self._positions[streams, rows].numpy(),
            rgb=np.clip(self._rgb[streams, rows].numpy(), 0.0, 1.0),
            counts=self._counts[:count].sum(dim=1).numpy(),
            depths=self._depths[streams, rows].numpy(),
            frame_names=names,
            frame_poses=frame_poses,
            qualifying_voxels=int(self._has[:count, _QUALIFYING].sum().item()),
        )
