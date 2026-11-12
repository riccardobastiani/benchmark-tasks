"""Check joint anchors and the scene's starting arrangement before native execution."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).parent
spec = importlib.util.spec_from_file_location('_wire_geometry', root/'geometry.py')
g = importlib.util.module_from_spec(spec); sys.modules[spec.name] = g; spec.loader.exec_module(g)
from yam_arena.urdf import quat_xyzw_to_matrix
assets = g.geometry(); wire = assets['wire']; nodes = g.wire_nodes()
assert len(wire['bodies']) == len(nodes)-1 and len(wire['joints']) == len(nodes)-2
for i, part in enumerate(wire['bodies']):
    w, x, y, z = part['orientation']; rotation = quat_xyzw_to_matrix(np.array([x, y, z, w]))
    centre = np.array(part['position']); half = part['shapes'][0]['size'][1]
    np.testing.assert_allclose(centre-rotation[:, 2]*half, nodes[i], atol=1e-12)
    np.testing.assert_allclose(centre+rotation[:, 2]*half, nodes[i+1], atol=1e-12)
    assert 0 < part['mass'] < .01
for name in ('upstream', 'target'):
    clip = assets[name]
    assert len(clip['joints']) == 2
    assert all(j['stiffness'] > 0 and j['damping'] > 0 for j in clip['joints'])
assert abs(nodes[0, 0]-g.UPSTREAM[0]) < 1e-8
assert np.linalg.norm(nodes[-1, :2]-g.TARGET[:2]) > .04
print(f'Geometry and native joint anchors passed: {len(wire["bodies"])} wire segments.')

assert abs(g.GATE_HIGH-g.GATE_LOW-.080)<1e-12
assert np.max(nodes[:,0])+g.WIRE_RADIUS<g.GATE_X
posts=[shape for shape in assets['fixture']['bodies'][0]['shapes'] if shape['type']=='cylinder']
assert len(posts)==2 and all(shape['size']==[g.POST_RADIUS,g.POST_HEIGHT/2] for shape in posts)
print('Two-post geometry and initially outside-gate wire checked.')

assert g.GATE_WIRE_HEIGHT==g.POST_HEIGHT-g.WIRE_RADIUS
