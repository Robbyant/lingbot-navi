"""Expose only task text or goal pixels to the agent; evaluator geometry stays private."""
from pathlib import Path
import numpy as np
from PIL import Image


def validate_episode(episode):
    if len(episode.tasks) != len(episode.goals):
        raise ValueError('GOAT tasks/goals length mismatch (upstream long-description filtering). '
                         'Use an aligned episode; do not silently change benchmark tasks.')
    for task in episode.tasks:
        if task[1] not in ('object', 'description', 'image'):
            raise ValueError('Unsupported GOAT task: ' + str(task[1]))


def task_target(env, index, output):
    episode = env.current_episode
    task = episode.tasks[index]
    kind = task[1]
    if kind == 'object':
        return dict(kind=kind, text=str(task[0]).replace('_', ' ').strip() or 'object')
    goals = episode.goals[index]
    goal = next(g for g in goals if str(g['object_id']) == str(task[2]))
    if kind == 'description':
        text=str(goal.get('lang_desc') or '').strip().rstrip('.')
        return dict(kind=kind, text=text or str(task[0]).replace('_', ' ').strip() or 'object')
    # Honor the image index chosen by this episode, not a random goal photograph.
    parameters = goal['image_goals'][int(task[-1])]
    image = render_goal_image(env.sim, parameters)
    path = Path(output) / ('goal_%02d.png' % index)
    Image.fromarray(image).save(path)
    return dict(kind='image', image=str(path))


def render_goal_image(sim, parameters):
    """Use LingBot-Navi's CUDA renderer at the dataset optical pose and HFOV."""
    dimensions = list(parameters.get('image_dimensions', parameters.get('dimensions', [512, 512])))
    if dimensions != [512, 512]:
        raise ValueError('Expected GOAT 512x512 target image')
    hfov = float(parameters['hfov'])
    if not 0 < hfov < 180:
        raise ValueError('Invalid target image HFOV')
    # Match the historical goal agent's Habitat/Magnum storage round trip.
    # Keep this conversion separate from the live navigation agent.
    from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum, quat_from_magnum
    rotation = quat_from_magnum(quat_to_magnum(quat_from_coeffs(parameters['rotation'])))
    position = np.asarray(parameters['position'], dtype=np.float32)
    camera = dict(position=position.tolist(), rotation=[rotation.x, rotation.y, rotation.z, rotation.w],
                  height=512, width=512, hfov=hfov, near=.01, far=1000., depth=False)
    return np.asarray(sim.render_cameras({'goal': camera})['goal'])[..., :3].copy()
