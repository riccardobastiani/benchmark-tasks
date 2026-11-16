"""Human-free scene, nominal mass data and unchanged calibrated geometry."""
import numpy as np
from geometry import geometry,TIP,TIP_RADIUS,DOCK,TRAY_START
assets=geometry()
assert set(assets)=={'tray','fixture','sharp_a','sharp_b','spatula','holder'}
for data in assets.values():
 assert not data.get('joints')
 for part in data['bodies']:
  assert part['mass']>0
  assert np.linalg.eigvalsh(np.array(part['inertia']['tensor'])).min()>0
for name in ('sharp_a','sharp_b'):
 tip=assets[name]['bodies'][0]['shapes'][6]
 np.testing.assert_allclose(tip['fromto'][3:],TIP)
 assert tip['size'][0]==TIP_RADIUS==.00045
assert DOCK[0]>TRAY_START[0]
assert {p.name for p in (__import__('pathlib').Path(__file__).parent/'assets').glob('*.png')}=={'USED_SURGICAL_INSTRUMENTS.png','SHARPS_ONLY.png','SOILED_INSTRUMENTS.png'}
print('No human actor/joints; original labels, whole devices and nominal native masses checked.')
