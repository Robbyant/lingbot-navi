"""Source mesh axial depth; RGB remains in data_toolkit.render.GaussianScene."""
import numpy as np

class MeshDepth:
    def __init__(self, path, device='cuda:0'):
        import torch
        import nvdiffrast.torch as dr
        import trimesh
        self.torch,self.dr,self.device=torch,dr,device
        mesh=trimesh.load(path,process=False,force='mesh')
        vertices=np.asarray(mesh.vertices,dtype=np.float32)
        self.mesh_h=torch.tensor(np.column_stack([vertices,np.ones(len(vertices),np.float32)]),device=device)
        self.faces=torch.tensor(np.asarray(mesh.faces,dtype=np.int32),device=device)
        self.ctx=dr.RasterizeCudaContext(device=device)

    def render(self, camera, c2w, near=.01, far=100.):
        torch,dr=self.torch,self.dr
        height,width=camera.height,camera.width
        focal=float(camera.K[0,0])
        if not np.allclose(camera.K,[[focal,0,width/2],[0,focal,height/2],[0,0,1]]):
            raise ValueError('Mesh depth requires centered square-pixel intrinsics')
        view_t=torch.tensor(np.linalg.inv(c2w).astype(np.float32),device=self.device)
        with torch.inference_mode():
            camera=self.mesh_h@view_t.T
            # OpenCV Y down -> OpenGL clip Y up; flip raster output back to top-left.
            proj=torch.tensor([[2*focal/width,0,0,0],[0,-2*focal/height,0,0],[0,0,(far+near)/(far-near),-2*far*near/(far-near)],[0,0,1,0]],device=self.device,dtype=torch.float32)
            clip=(camera@proj.T)[None].contiguous()
            # Conservative six-plane culling, then compact vertices. Large original
            # scenes can exceed CUDA rasterizer work-buffer limits without compaction.
            faces=self.faces
            for axis,sign in ((0,1),(0,-1),(1,1),(1,-1),(2,1),(2,-1)):
                plane=clip[0,:,3]+sign*clip[0,:,axis]
                faces=faces[(plane[faces.long()]>=0).any(-1)]
                if not len(faces):return np.zeros((height,width),np.float32)
            nearest=torch.full((height,width),float('inf'),device=self.device)
            for face_chunk in faces.split(1000000):
                unique,inverse=torch.unique(face_chunk.long().reshape(-1),sorted=True,return_inverse=True)
                packed_faces=inverse.reshape(-1,3).to(torch.int32).contiguous()
                packed_clip=clip[:,unique,:].contiguous()
                rast,_=dr.rasterize(self.ctx,packed_clip,packed_faces,resolution=[height,width])
                z,_=dr.interpolate(camera[unique,2:3][None].contiguous(),rast,packed_faces)
                z=torch.where(rast[...,3:4]>0,z,torch.full_like(z,float('inf')))[0,:,:,0]
                nearest=torch.minimum(nearest,z)
            result=torch.where(torch.isfinite(nearest),nearest,torch.zeros_like(nearest))
            return torch.flip(result,dims=(0,)).cpu().numpy().astype(np.float32)
