"""Make GOAT's training imports optional; leave tasks/dataset/measurements untouched."""
import argparse
from pathlib import Path


def prepare(root):
    path = Path(root) / 'goat_bench/__init__.py'
    utility = Path(root) / 'goat_bench/utils/utils.py'
    text = utility.read_text()
    import_line = 'from goat_bench.models.encoders.resnet_gn import ResNet'
    if '\n' + import_line + '\n' in text:
        if '    if isinstance(encoder.backbone, ResNet):' not in text:
            raise ValueError('Unrecognized GOAT encoder helper')
        text = text.replace('\n' + import_line + '\n', '\n')
        text = text.replace('    if isinstance(encoder.backbone, ResNet):',
                            '    ' + import_line + '\n    if isinstance(encoder.backbone, ResNet):')
        utility.write_text(text)
    original = path.read_text()
    marker = '# LingBot-Navi optional evaluation-only imports\n'
    if 'if os.environ.get("LINGBOT_GOAT_EVAL_ONLY") == "1":' in original:
        return
    required = ('from goat_bench import config', 'from goat_bench.models import',
                'from goat_bench.trainers import')
    if not all(text in original for text in required):
        raise ValueError('Unrecognized GOAT initializer; review compatibility before patching')
    # Normal imports retain upstream behavior. The evaluator opts into a minimal registry.
    lightweight = '''from goat_bench import config
from goat_bench.dataset import goat_dataset
from goat_bench.measurements import nav
from goat_bench.task import actions, goat_task, simulator
'''
    text = marker + 'import os\nif os.environ.get("LINGBOT_GOAT_EVAL_ONLY") == "1":\n'
    text += ''.join('    ' + line + '\n' for line in lightweight.splitlines())
    text += 'else:\n' + ''.join('    ' + line + '\n' for line in original.splitlines())
    path.write_text(text)
    print('Enabled optional evaluation-only GOAT registration:', path)



def prepare_modern(root):
    """Apply the Habitat 0.3 API/config bridge used by the validated runtime."""
    root = Path(root)
    init = root / 'goat_bench/__init__.py'
    text = init.read_text().replace(
        '    from goat_bench.task import actions, goat_task, simulator',
        '    from goat_bench.task import goat_task, simulator')
    init.write_text(text)
    config = root / 'config/tasks/goat_stretch_hm3d.yaml'
    text = config.read_text()
    text = text.replace(
        '  - /habitat/simulator/agents@habitat.simulator.agents.main_agent: rgb_agent',
        '  - /habitat/simulator/agents@habitat.simulator.agents.main_agent: agent_base\n'
        '  - /habitat/simulator/sim_sensors@habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor: rgb_sensor')
    if '      look_up:\n        tilt_angle: 30' not in text:
        text = text.replace('    actions:\n', '    actions:\n      look_up:\n        tilt_angle: 30\n      look_down:\n        tilt_angle: 30\n')
    text = '\n'.join(line for line in text.split('\n') if line != '    tilt_angle: 30').replace('    action_space_config: "v2-goat"\n', '')
    config.write_text(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default='third_party/goat-bench')
    parser.add_argument('--habitat-modern', action='store_true', help='Apply the validated Habitat 0.3 config bridge')
    args = parser.parse_args()
    prepare(args.root)
    if args.habitat_modern:
        prepare_modern(args.root)
