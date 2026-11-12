"""Check task/policy metadata consistency before native execution or upload."""
import ast
import json
from pathlib import Path
import re
import sys

root = Path(__file__).parent
spec = json.loads((root/'task.json').read_text())
pilot = json.loads((root/'pilot.json').read_text())
for field in ('title', 'category', 'instruction', 'setup', 'scoring', 'run_limit', 'fidelity', 'scores'):
    assert isinstance(spec.get(field), str) and spec[field].strip(), field
assert spec['category'] in ('Impact on jobs', 'Impact on AI timelines', 'Safety and alignment',
                            'Societal relatability', 'Other')
assert len(spec['fidelity']) <= 500
tree = ast.parse((root/'task.py').read_text())
instruction = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id=='INSTRUCTION' for t in n.targets))
assert instruction == spec['instruction'] == pilot['instruction']
assert pilot['max_steps'] == 440 and pilot['control_hz'] == 20
for name in ('geometry.json', 'build_scene.py', 'setup.png', 'LICENSE', 'pyproject.toml',
             'resist_isaac/runtime.py', 'resist_isaac/state.py', 'resist_isaac/scripted.py',
             'tools/run_trial.py'):
    assert (root/name).is_file(), name
if '--ready' in sys.argv:
    assert re.fullmatch(r'M-[A-Z0-9]+', spec.get('media', '')), 'Real uploaded media ID required'
    assert 'not yet' not in spec['scores'].lower(), 'Actual evaluated scores required'
print('Task instruction, scorer package, 440-step limit and metadata checks passed.')
if not spec.get('media'):
    print('Publication pending: upload real media before the --ready check.')
