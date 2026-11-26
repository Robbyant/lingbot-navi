"""Run from the repository root: python -m examples.render_3dgs --ply scene.ply"""

import argparse

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from data_toolkit.render import CameraSpec, GaussianScene
from data_toolkit.render.geometry import FLU_FROM_CV


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True, help="Path to a 3D Gaussian Splatting PLY file")
    parser.add_argument(
        "--pose", type=float, nargs=6, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        default=[0, 0, 1.5, 0, 0, 0],
        help="World XYZ in meters and roll/pitch/yaw in degrees; zero angles look +X, Z is up",
    )
    args = parser.parse_args()

    camera = CameraSpec.from_hfov(width=640, height=480, hfov_deg=90)
    pose = np.eye(4)
    pose[:3, 3] = args.pose[:3]
    pose[:3, :3] = Rotation.from_euler("xyz", args.pose[3:], degrees=True).as_matrix() @ FLU_FROM_CV
    with GaussianScene(args.ply) as scene:
        rgb = np.asarray(scene.render_camera(camera, pose).rgb)

    cv2.imshow("3DGS", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
