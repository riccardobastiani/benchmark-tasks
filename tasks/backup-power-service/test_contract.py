"""Check submission fields, numeric budget and included task assets."""
import json
from pathlib import Path
import re
import sys

root=Path(__file__).parent
spec=json.loads((root/'task.json').read_text())
pilot=json.loads((root/'pilot.json').read_text())
for field in ('title','category','instruction','setup','scoring','run_limit','fidelity','scores'):
    assert isinstance(spec.get(field),str) and spec[field].strip(),field
assert spec['category'] in ('Impact on jobs','Impact on AI timelines','Safety and alignment',
                            'Societal relatability','Other')
assert len(spec['fidelity'])<=500
assert len(spec['scores'])<=300
assert spec['instruction']==pilot['instruction']
assert pilot['max_steps']==2400 and pilot['control_hz']==20
for name in ('task.py','geometry.py','rules.py','interaction.py','build_scene.py','setup.png',
             'LICENSE','pyproject.toml','resist_isaac/runtime.py','resist_isaac/usd_assets.py',
             'tools/run_recovery_pilot.py','ORIGIN.json'):
    assert (root/name).is_file(),name
assert list((root/'assets').glob('*.png'))
if '--ready' in sys.argv:
    assert re.fullmatch(r'M-[A-Z0-9]+',spec.get('media','')),'Actual media upload ID required'
    assert 'not run' not in spec['scores'].lower(),'Completed Dream-1 score required'
print('Instruction, budget, field lengths and task package checks passed.')
