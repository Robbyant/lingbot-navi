"""Install the shared inference, memory-building and rendering dependencies."""

import argparse
import os
import re
import tempfile
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
HABITAT_COMMIT = "57ee4941dc4765240f0f91f70b2c97a919bf9038"


def pip(*args):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *map(str, args)],
        cwd=ROOT,
        check=True,
    )



def install_habitat(wheel=None):
    cache = ROOT / "third_party" / ".habitat-build"
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key, directory in (("TMPDIR", "tmp"), ("CCACHE_DIR", "ccache"),
                           ("XDG_CACHE_HOME", "cache")):
        env.setdefault(key, str(cache / directory))
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    env.update(
        HABITAT_BUILD_GUI_VIEWERS="OFF", HABITAT_WITH_BULLET="OFF",
        HABITAT_WITH_CUDA="OFF", HABITAT_WITH_AUDIO="OFF",
        HABITAT_BUILD_TEST="OFF", CMAKE_POLICY_VERSION_MINIMUM="3.5",
    )
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "4")

    def run(command, cwd=ROOT):
        subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)

    with tempfile.TemporaryDirectory(dir=cache) as directory:
        work = Path(directory)
        constraints = work / "constraints.txt"
        pins = [line for line in (ROOT / "requirements.txt").read_text().splitlines()
                if re.fullmatch(r"[A-Za-z0-9_.-]+==[^ ;]+", line.strip())]
        constraints.write_text("\n".join(pins) + "\n")
        if wheel is None:
            run([sys.executable, "-m", "pip", "install",
                 "scikit-build-core>=0.10,<0.12", "pybind11>=2.10,<3",
                 "cmake>=3.27,<4", "ninja"])
            source = work / "source"
            run(["git", "init", source])
            run(["git", "remote", "add", "origin",
                 "https://github.com/facebookresearch/habitat-sim.git"], source)
            run(["git", "fetch", "--depth", "1", "origin", HABITAT_COMMIT], source)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], source)
            run(["git", "submodule", "update", "--init", "--recursive",
                 "--depth", "1", "--jobs", "4"], source)
            wheels = work / "wheels"
            run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                 "--no-build-isolation", "--wheel-dir", wheels, source])
            wheel, = wheels.glob("habitat_sim-*.whl")
        run([sys.executable, "-m", "pip", "install", "-c", constraints, wheel])
        run([sys.executable, "-m", "pip", "check"])
        run([sys.executable, "-c",
             "import habitat_sim; print('Habitat-Sim:', habitat_sim.__version__)"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    habitat = parser.add_mutually_exclusive_group()
    habitat.add_argument("--habitat-sim", action="store_true",
                         help="Build and install pinned headless Habitat-Sim for evaluation")
    habitat.add_argument(
        "--habitat-sim-wheel", type=Path,
        help="Install a local Habitat-Sim wheel without changing the pinned dependencies",
    )
    args = parser.parse_args()
    if args.habitat_sim_wheel is not None:
        args.habitat_sim_wheel = args.habitat_sim_wheel.resolve()
        if not args.habitat_sim_wheel.is_file() or args.habitat_sim_wheel.suffix != ".whl":
            parser.error("--habitat-sim-wheel must point to an existing .whl file")
    pip(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
    )
    # nvdiffrast builds against the installed PyTorch/CUDA toolchain.
    pip("setuptools>=61", "wheel", "ninja")
    pip("--no-build-isolation", "-r", ROOT / "requirements.txt")
    if args.habitat_sim or args.habitat_sim_wheel is not None:
        install_habitat(args.habitat_sim_wheel)
    # Compile before evaluation workers can race to initialize the same cache.
    print("Initializing gsplat CUDA extension (first install may take several minutes)...", flush=True)
    subprocess.run(
        [sys.executable, "-c",
         "from gsplat.cuda._backend import _C\nif _C is None:\n    raise RuntimeError('gsplat CUDA extension unavailable; check CUDA Toolkit installation')\nprint('gsplat CUDA extension ready:', _C.__file__)\n"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
