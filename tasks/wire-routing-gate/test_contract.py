"""Check the fixed task's metadata and included source before submission."""
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
assert pilot['max_steps']==1200 and pilot['control_hz']==20
for name in ('task.py','geometry.py','rules.py','interaction.py','build_scene.py','setup.png',
             'LICENSE','pyproject.toml','resist_isaac/runtime.py','resist_isaac/usd_assets.py',
             'tools/run_recovery_pilot.py'):
    assert (root/name).is_file(),name
if '--ready' in sys.argv:
    assert re.fullmatch(r'M-[A-Z0-9]+',spec.get('media','')),'Real uploaded media ID required'
    assert 'not run' not in spec['scores'].lower(),'Actual evaluated scores required'
print('Instruction, 1200-control budget, fidelity length and package checks passed.')
if not spec.get('media'):print('Media upload and public PR publication remain pending.')
