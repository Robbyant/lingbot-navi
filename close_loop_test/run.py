"""Run evaluation from an experiment configuration: python -m close_loop_test.run --help."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import signal
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATH_FIELDS = ('weights', 'base_model', 'map1_checkpoint', 'goat_root',
               'data_root', 'scene_root', 'dataset_release')


def reference_scenes(args):
    dataset = args.dataset
    if dataset == 'lingbot':
        from .lingbot.release import ReleaseV1
        release = ReleaseV1(args.dataset_release, args.manifest_sha256)
        contents = {path.stem: release.content(path) for path in release.task_files()}
    else:
        contents = {}
        for path in sorted(Path(args.data_root).glob(args.split + '/content/*.json.gz')):
            with gzip.open(path, 'rt') as stream:
                contents[path.name[:-len('.json.gz')]] = json.load(stream)
    references = json.loads(Path(__file__).with_name('reference_cases.json').read_text())[dataset]
    actual = {}
    for scene, content in contents.items():
        if not content['episodes']:
            raise ValueError('No episodes in scene ' + scene)
        episode = content['episodes'][0]
        tasks = json.dumps(episode['tasks'], ensure_ascii=False, separators=(',', ':')).encode()
        actual[scene] = dict(episode_id=str(episode['episode_id']),
                             tasks_sha256=hashlib.sha256(tasks).hexdigest())
    if actual != references:
        raise ValueError('Reference scene/episode/task list mismatch for ' + args.dataset)
    if args.scene:
        if args.scene not in references:
            raise ValueError('Scene is not in the reference experiment: ' + args.scene)
        return [args.scene]
    return sorted(references)


def build_commands(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path,
                        help='Experiment JSON under close_loop_test/config/')
    parser.add_argument('--scene', help='Run only this reference scene')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text())
        if config['dataset'] not in ('goat', 'lingbot') or not isinstance(config['map1'], bool):
            raise ValueError('dataset must be goat/lingbot and map1 must be boolean')
        for key in ('budget', 'init_explore_steps', 'max_voxels', 'parallelism'):
            if type(config[key]) is not int or config[key] <= 0:
                raise ValueError(key + ' must be a positive integer')
        if type(config['gpu']) is not int or config['gpu'] < 0:
            raise ValueError('gpu must be a nonnegative integer')
        if config['parallelism'] != 1:
            raise ValueError('This reproduction runner uses parallelism=1')
        if type(config['seed']) is not int:
            raise ValueError('seed must be an integer')
        required = ['weights'] + (['base_model'] if 'base_model' in config else [])
        required += ['goat_root', 'data_root', 'scene_root', 'split'] if config['dataset'] == 'goat' else ['dataset_release', 'manifest_sha256']
        if config['map1']:
            required.append('map1_checkpoint')
        for key in required:
            if not isinstance(config.get(key), str) or not config[key]:
                raise ValueError('Missing configuration field: ' + key)
        for key in PATH_FIELDS:
            if key in config:
                config[key] = str((args.config.resolve().parent / config[key]).resolve())
        if 'base_model' not in config:
            weights = Path(config['weights'])
            config['base_model'] = str(weights if weights.is_dir() else weights.parent)
        for key in ('dataset', 'map1', 'budget', 'seed', 'max_voxels', 'init_explore_steps',
                    'gpu', 'parallelism', 'split', 'manifest_sha256', *PATH_FIELDS):
            setattr(args, key, config.get(key))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.configuration = config
    dataset, map1, explore = args.dataset, args.map1, args.init_explore_steps
    label = dataset + ('-map1-' if map1 else '-no-map1-')
    output = args.output or ROOT / 'outputs' / (label + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if output.exists():
        parser.error('output directory already exists: ' + str(output))
    try:
        scenes = reference_scenes(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    cmd = [sys.executable, '-m', 'close_loop_test.' + dataset + '.run',
           '--episode-index', '0', '--model-python', sys.executable,
           '--weights', args.weights, '--base-model', args.base_model,
           '--init-explore-steps', str(explore), '--budget', str(args.budget),
           '--max-voxels', str(args.max_voxels), '--gpu', '0', '--device', 'cuda:0', '--seed', str(args.seed),
           '--lingbot-map1' if map1 else '--no-lingbot-map1']
    if dataset == 'goat':
        cmd += ['--goat-root', args.goat_root, '--data-root', args.data_root,
                '--scene-root', args.scene_root, '--split', args.split,
                '--independent-initial-exploration']
    else:
        cmd += ['--dataset-release', args.dataset_release, '--renderer-python', sys.executable,
                '--manifest-sha256', args.manifest_sha256]
    if map1:
        cmd += ['--map1-python', sys.executable, '--map1-checkpoint', args.map1_checkpoint]
    commands = [cmd + ['--scene', scene, '--output', str(output / scene / 'episode-0')]
                for scene in scenes]
    args.output = output
    return commands, args


def main():
    commands, args = build_commands()
    if args.dry_run:
        print(f'GPU: {args.gpu}; parallelism: {args.parallelism}')
        for cmd in commands:
            print(shlex.join(cmd))
        return
    from .summarize import summarize
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(config_file=str(args.config.resolve()), configuration=args.configuration, commands=commands)
    manifest['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest['source_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for folder in ('close_loop_test', 'lingbot_navi')
                                 for path in sorted((ROOT / folder).rglob('*.py'))}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    env.pop('PYTHONPATH', None)
    records = []
    try:
        for cmd in commands:
            scene = cmd[cmd.index('--scene') + 1]
            row = dict(scene=scene, output=cmd[cmd.index('--output') + 1], returncode=None)
            records.append(row)
            print('Evaluating', scene, flush=True)
            with (args.output / (scene + '.log')).open('w') as log:
                process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                try:
                    row['returncode'] = process.wait()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
    finally:
        (args.output / 'subset_status.json').write_text(json.dumps(records, indent=2) + '\n')
        paths = [Path(cmd[cmd.index('--output') + 1]) / 'result.json' for cmd in commands]
        try:
            summary = summarize([p for p in paths if p.exists()], len(commands))
        except (OSError, ValueError, KeyError, TypeError) as error:
            summary = dict(status='incomplete', error=str(error), metrics={})
        summary['missing_results'] = [str(p) for p in paths if not p.exists()]
        summary['failed_processes'] = [r['scene'] for r in records if r['returncode'] != 0]
        if summary['failed_processes'] or len(records) != len(commands):
            summary['status'] = 'incomplete'
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if summary['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
