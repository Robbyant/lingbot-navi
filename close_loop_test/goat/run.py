"""Run exactly one episode using official GOAT tasks and measurements."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tempfile import TemporaryDirectory
# Bound per-process CPU parallelism for small synchronous batches.
for _name, _value in [('OMP_NUM_THREADS', '4'), ('MKL_NUM_THREADS', '4'), ('OPENBLAS_NUM_THREADS', '1')]:
    os.environ.setdefault(_name, _value)
import numpy as np
from close_loop_test.common.observations import jsonable, save_observation
from .goals import validate_episode, task_target
from close_loop_test.common.client import ModelClient


def build_environment(args):
    goat = args.goat_root.resolve()
    if not (goat / 'goat_bench').is_dir():
        raise FileNotFoundError('Clone GOAT-Bench into third_party/goat-bench; see README.md')
    sys.path.insert(0, str(goat))
    os.environ["LINGBOT_GOAT_EVAL_ONLY"] = "1"
    import goat_bench  # official registration
    import habitat
    from .compat import register_simulator
    renderer_options = json.loads(args.renderer_config.read_text()) if args.renderer_config else {}
    renderer_options.setdefault("python", str(args.model_python or sys.executable))
    register_simulator(renderer_options)
    from habitat.config import read_write
    from habitat.config.default_structured_configs import HabitatSimDepthSensorConfig
    cfg = habitat.get_config(str(goat / 'config/tasks/goat_stretch_hm3d.yaml'))
    with read_write(cfg):
        cfg.habitat.dataset.data_path = str(args.data_root.resolve() / args.split / (args.split + '.json.gz'))
        cfg.habitat.dataset.scenes_dir = str(args.scene_root.resolve())
        cfg.habitat.dataset.split = args.split
        if args.scene:
            cfg.habitat.dataset.content_scenes = [args.scene]
        cfg.habitat.environment.iterator_options.shuffle = False
        cfg.habitat.environment.max_episode_steps = args.budget * 20 + 1 + (args.init_explore_steps + 2 if args.independent_initial_exploration else 0)
        cfg.habitat.seed = args.seed
        # LingBot uses raw text/image goals; do not instantiate CLIP cache sensors.
        cfg.habitat.task.lab_sensors = {}
        cfg.habitat.task.actions.look_up.type = 'LingBotNativeLookUp'
        cfg.habitat.task.actions.look_down.type = 'LingBotNativeLookDown'
        cfg.habitat.task.measurements.pop('goat_distance_to_goal_reward', None)
        sim = cfg.habitat.simulator
        sim.type = "LingBotGoatSim-v0"
        sim.habitat_sim_v0.gpu_device_id = args.gpu
        sim.scene_dataset = str(args.scene_root.resolve() / 'hm3d/hm3d_basis.scene_dataset_config.json')
        sim.forward_step_size = .25
        sim.navmesh_settings.agent_max_climb = .10
        sim.navmesh_settings.cell_height = .05
        sensors = sim.agents.main_agent.sim_sensors
        sensors.depth_sensor = HabitatSimDepthSensorConfig()
        for name in ('rgb_sensor', 'depth_sensor'):
            sensor = sensors[name]
            sensor.width, sensor.height, sensor.hfov = 640, 480, 90
            sensor.position = [0., 1.5, 0.]
        sensors.depth_sensor.normalize_depth = False
        sensors.depth_sensor.min_depth = 0.
        sensors.depth_sensor.max_depth = 10.
    dataset = habitat.make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    if not 0 <= args.episode_index < len(dataset.episodes):
        raise ValueError('episode-index outside loaded dataset: %d' % len(dataset.episodes))
    episode = dataset.episodes[args.episode_index]
    validate_episode(episode)
    dataset.episodes = [episode]
    env = habitat.Env(config=cfg, dataset=dataset)
    return env, cfg




def run(args):
    map1 = None
    if args.lingbot_map1:
        map1 = dict(python=str(args.map1_python or args.model_python),
                    checkpoint=str(args.map1_checkpoint.resolve()))
        for value in map1.values():
            if not Path(value).exists(): raise FileNotFoundError(value)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = TemporaryDirectory(prefix='.runtime-', dir=output)
    work = Path(runtime.name)
    env = client = session = None
    events = open(output / 'events.jsonl', 'w')
    started = time.monotonic()
    steps, current, substeps, plans = 0, -1, 0, 0
    result = dict(status='running', baseline='release_v3_lingbotnavi_goat_policy', renderer='mesh_cuda',
                  uses_simulator_depth_and_pose=not args.lingbot_map1, uses_navmesh_execution=True,
                  memory_geometry="lingbot_map1" if args.lingbot_map1 else "simulator",
                  uses_simulator_reference_for_map_scale=args.lingbot_map1,
                  policy_source_commit='999b6930889ff3abe829318c356cb0c5bca71b67',
                  retained_differences=['release_v3_center_model', 'release_numerical_runtime'],
                  arguments=jsonable(vars(args)))

    def event(kind, **values):
        row = dict(event=kind, step=steps, subtask=current, elapsed_seconds=time.monotonic() - started)
        row.update(values)
        events.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
        events.flush()

    try:
        env, _ = build_environment(args)
        result['renderer_options'] = env.sim._cuda_options
        if env.sim.config.sim_cfg.create_renderer:
            raise RuntimeError('Habitat native renderer must remain disabled')
        observations = env.reset()
        from .compat import apply_navigation_compatibility
        result["navigation_compatibility"] = apply_navigation_compatibility(env.sim)
        result.update(episode_id=env.current_episode.episode_id,
                      scene_id=env.current_episode.scene_id,
                      tasks=env.current_episode.tasks,
                      goat_commit=subprocess.check_output(
                          ['git', '-C', str(args.goat_root), 'rev-parse', 'HEAD'], text=True).strip())
        if args.environment_only:
            # Explicit diagnostic mode; never reported as a model evaluation.
            result['baseline'] = 'environment_only'
        else:
            client = ModelClient(args.model_python, output, dict(
                weights=str(args.weights.resolve()), base_model=str(args.base_model.resolve()),
                device=args.device, output=str(work), max_voxels=args.max_voxels, scene=Path(env.current_episode.scene_id).name.replace('.basis.glb', ''), map1=map1))
            result['model'] = client.identity
            if client.identity['goal_mode'] == 'viewpoint':
                result['baseline'] = 'release_v4_viewpoint_lingbotnavi_goat_policy'
                result['retained_differences'] = ['release_numerical_runtime']
        if args.environment_only:
            while not env.episode_over:
                observations = env.step({'action':'subtask_stop'})
                steps += 1
        else:
            from .session import Session
            session = Session(env, client, work, args.budget, event, save_observation, map1=args.lingbot_map1, init_explore_steps=args.init_explore_steps, independent_initial_exploration=args.independent_initial_exploration)
            while session._episode_active and not session.done:
                session.episode_step()
                steps, current, plans = session.step, session.subtask_idx, session.plans
        result.update(status='environment_checked' if args.environment_only else 'complete',
                      metrics=env.get_metrics(), steps=steps, plans=plans)
    except BaseException as error:
        if session is not None:
            steps, plans = session.step, session.plans
        result.update(status='failed', error=repr(error), steps=steps, plans=plans)
        if env is not None:
            result['partial_metrics'] = env.get_metrics()
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        (output / 'result.json').write_text(json.dumps(jsonable(result), indent=2, allow_nan=False))
        events.close()
        try:
            if client is not None:
                client.close()
            if env is not None:
                env.close()
        finally:
            runtime.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goat-root', type=Path, default=Path('third_party/goat-bench'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--scene-root', type=Path, required=True)
    parser.add_argument('--split', default='val_unseen', choices=['val_seen', 'val_seen_synonyms', 'val_unseen'])
    parser.add_argument('--scene')
    parser.add_argument('--episode-index', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--renderer-config', type=Path, help='CUDA renderer interpreter/dependency options JSON')
    parser.add_argument('--model-python', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--base-model', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--budget', type=int, default=500, help='Actions per subtask, including submission')
    parser.add_argument('--init-explore-steps', type=int, default=200)
    parser.add_argument('--independent-initial-exploration', action='store_true',
                        help='Keep initial exploration outside the per-subtask action budget')
    parser.add_argument('--max-voxels', type=int, default=500000)
    parser.add_argument('--lingbot-map1', action=argparse.BooleanOptionalAction, default=False,
                        help='Use streaming lingbot-map1.0 memory geometry (default: simulator)')
    parser.add_argument('--map1-python', type=Path, help='lingbot-map1.0 interpreter; defaults to model-python')
    parser.add_argument('--map1-checkpoint', type=Path, help='lingbot-map1.0 checkpoint')
    parser.add_argument('--environment-only', action='store_true')
    args = parser.parse_args()
    if args.base_model is None and args.weights is not None:
        weights = args.weights.expanduser().resolve()
        args.base_model = weights if weights.is_dir() else weights.parent
    for name in ('budget', 'max_voxels', 'init_explore_steps'):
        if getattr(args, name) <= 0:
            parser.error(name + ' must be positive')
    if not args.environment_only and any(getattr(args, name) is None for name in ('model_python', 'weights')):
        parser.error('model-python and weights are required for model evaluation')
    if args.lingbot_map1 and args.map1_checkpoint is None:
        parser.error('--lingbot-map1 requires --map1-checkpoint')
    if args.lingbot_map1 and args.environment_only:
        parser.error('--environment-only does not execute Map1')
    import signal
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Evaluation interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
