"""Summarize GOAT/LingBot results with episode-equal weighting."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


# Execution placement and output paths do not change the evaluation protocol.
PLACEMENT_ARGUMENTS = {'scene', 'episode_index', 'output', 'gpu', 'device',
                       'model_python', 'renderer_python', 'map1_python'}
PROTOCOL_FIELDS = ('dataset', 'manifest_sha256', 'judge_protocol', 'renderer',
                   'depth_source', 'navigation_executor', 'memory_geometry',
                   'uses_navmesh_execution', 'uses_simulator_depth_and_pose',
                   'uses_simulator_reference_for_map_scale', 'model',
                   'navigation_compatibility', 'goat_commit', 'policy_source_commit')


def configuration_signature(row):
    protocol = {key: row.get(key) for key in PROTOCOL_FIELDS}
    protocol['arguments'] = {key: value for key, value in row['arguments'].items()
                             if key not in PLACEMENT_ARGUMENTS}
    return json.dumps(protocol, sort_keys=True, allow_nan=False)


def summarize(paths, expected_episodes=None):
    """Retain per-type means; never present a partial batch as verified complete.

    expected_episodes checks the count, not membership in a reference scene list.
    Callers reproducing a benchmark must also validate their selected cases.
    """
    if expected_episodes is not None and expected_episodes <= 0:
        raise ValueError('expected_episodes must be positive')
    buckets = defaultdict(list)
    excluded, seen, whole_episode_success = [], set(), []
    signature = None
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / 'result.json'
        row = json.loads(path.read_text())
        key = (row.get('scene_id', row['arguments'].get('scene')),
               str(row.get('episode_id', row['arguments'].get('episode_index'))))
        if key in seen:
            raise ValueError('Duplicate episode: ' + str(key))
        seen.add(key)
        if row['status'] != 'complete':
            excluded.append(dict(path=str(path), status=row['status']))
            continue
        current = configuration_signature(row)
        if signature is not None and signature != current:
            raise ValueError('Mixed evaluation configurations')
        signature = current
        tasks = row['tasks']
        success = row['metrics']['success']['subtask_success']
        spl = row['metrics']['spl']['spl_by_subtask']
        if not tasks or not len(tasks) == len(success) == len(spl):
            raise ValueError('Incomplete subtask metrics: ' + str(path))
        if any(value not in (0, 1) for value in success):
            raise ValueError('Invalid success value: ' + str(path))
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               or not 0 <= value <= 1 for value in spl):
            raise ValueError('Invalid SPL value: ' + str(path))
        if any(not isinstance(task, list) or len(task) < 2
               or task[1] not in ('object', 'description', 'image') for task in tasks):
            raise ValueError('Invalid task type: ' + str(path))
        whole_episode_success.append(all(success))
        for kind in ('overall', 'object', 'description', 'image'):
            ids = [i for i, task in enumerate(tasks) if kind == 'overall' or task[1] == kind]
            if ids:
                buckets[kind].append((len(ids), sum(success[i] for i in ids),
                                      statistics.mean(success[i] for i in ids),
                                      statistics.mean(spl[i] for i in ids)))
    completed = len(whole_episode_success)
    if not seen:
        raise ValueError('No episode results supplied')
    count_matches = expected_episodes is not None and len(seen) == expected_episodes
    status = 'complete' if count_matches and not excluded else 'incomplete'
    if expected_episodes is None and not excluded:
        status = 'unverified_count'
    return dict(status=status, expected_episodes=expected_episodes,
                supplied_episodes=len(seen), completed_episodes=completed,
                excluded=excluded, aggregation='episode_equal_weight',
                episode_success_rate=statistics.mean(whole_episode_success) if completed else None,
                metrics={kind: dict(episodes=len(values), subtasks=sum(x[0] for x in values),
                                    successes=sum(x[1] for x in values),
                                    SR=statistics.mean(x[2] for x in values),
                                    SPL=statistics.mean(x[3] for x in values))
                         for kind, values in buckets.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+', help='result.json files or single-episode directories')
    parser.add_argument('--expected-episodes', type=int, help='Expected batch size (e.g. 36); does not verify scene membership')
    args = parser.parse_args()
    try:
        result = summarize(args.results, args.expected_episodes)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    if result['status'] == 'incomplete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
