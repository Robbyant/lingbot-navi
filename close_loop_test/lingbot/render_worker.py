"""Use existing private renderer IPC and data_toolkit's native 3DGS RGB."""
import socket,sys,traceback
import numpy as np
from data_toolkit.hm3d_render.worker import send,receive
from data_toolkit.render import GaussianScene,CameraSpec
from data_toolkit.render.mesh_depth import MeshDepth
from close_loop_test.common.memory_camera import habitat_camera_pose_to_zup,_quat_to_matrix_xyzw

def main():
    sock=socket.socket(fileno=int(sys.argv[1]));rgb=None
    try:
        cfg=receive(sock);scene=cfg['worker_options'];device='cuda:'+str(cfg['gpu'])
        rgb=GaussianScene(scene['splat_path'],device=device)
        depth=MeshDepth(scene['source_mesh'],device=device)
        send(sock,dict(ok=True))
        while True:
            request=receive(sock)
            if request is None:break
            frames={}
            for name,spec in request.items():
                camera=CameraSpec.from_hfov(spec['width'],spec['height'],spec['hfov'])
                xyz,q=habitat_camera_pose_to_zup(spec['position'],spec['rotation'])
                pose=np.eye(4);pose[:3,:3]=_quat_to_matrix_xyzw(*q);pose[:3,3]=xyz
                frames[name]=(depth.render(camera,pose,spec['near'],spec['far']) if spec['depth']
                              else np.asarray(rgb.render_camera(camera,pose).rgb))
            send(sock,dict(ok=True,observations=frames))
    except BaseException:
        try:send(sock,dict(ok=False,error=traceback.format_exc()))
        except (OSError,EOFError):pass
        raise
    finally:
        if rgb is not None:rgb.close()
        sock.close()

if __name__=='__main__':main()
