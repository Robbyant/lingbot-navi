"""Build Spatial Memory by rendering a saved camera trajectory in a 3DGS scene."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gs-asset", required=True, type=Path)
    parser.add_argument(
        "--trajectory",
        default=ROOT / "examples/sample_data/trajectory.json",
        type=Path,
        help="JSON with camera poses, intrinsics and block frame indices",
    )
    parser.add_argument("--map-weights", default=ROOT / "ckpts/lingbot-map/lingbot-map.pt", type=Path)
    parser.add_argument("--base-model", default=ROOT / "ckpts/lingbot-navi/navigation_transformer", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for path in (args.gs_asset, args.trajectory, args.map_weights):
        if not path.is_file():
            parser.error(f"Missing input file: {path}")
    if not args.base_model.is_dir():
        parser.error(f"Missing navigation checkpoint directory: {args.base_model}")
    from examples.memory_builder import (
        TrajectoryAssets,
        LingBotSettings,
        FeatureSettings,
        produce_v1_blocks,
    )

    product = produce_v1_blocks(
        TrajectoryAssets(args.gs_asset.resolve(), args.trajectory.resolve()),
        args.output_dir,
        backend=LingBotSettings(args.map_weights.resolve(), device=args.device),
        features=FeatureSettings(args.base_model.resolve(), device=args.device),
    )
    print(f"Spatial Memory: {product.prepared_memory}")
    print(f"Rendering scene: {product.output_dir / 'scene.json'}")
    print(f"Capture trajectories: {product.output_dir / 'trajectory_bev.png'}")


if __name__ == "__main__":
    main()
