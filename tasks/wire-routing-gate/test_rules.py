"""Gate-route regressions: wrong side, over-post route, bights and free tails."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('_gate_rules',Path(__file__).with_name('rules.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
for routed in [False,True]:
 for seated in [False,True]:assert r.score(True,routed,seated,True,True)==0
assert r.score(False,False,False,True,False)==0
assert r.score(False,True,False,False,True)==.5
assert r.score(False,True,True,False,True)==.5
assert r.score(False,True,True,True,True)==1
seam=[(None,None,np.array([0,.012,.008]),np.array([0,.00005,.008])),
      (None,None,np.array([0,-.00005,.008]),np.array([0,-.012,.008]))]
assert r.clip_state(seam,np.zeros(3))==(True,True)
def state(points):
 p=[np.array(x,dtype=float) for x in points]
 return [((a+b)/2,np.zeros(6),a,b) for a,b in zip(p,p[1:])]
def route(points):
 return r.gate_route(state(points),np.array([0,1,0]),np.array([1,-1,0]),.5,-.2,.2,0,.004,.02)
z=.008
start=[(0,1.2,z),(0,1,z)];finish=[(1,-1,z),(1,-1.2,z)]
good=start+[(.5,0,z)]+finish
assert route(good)[0]
assert not route(start+[(.5,-.3,z)]+finish)[0]
assert not route(start+[(.5,0,.1)]+finish)[0]
# A correct-looking loose tail cannot rescue the wrong connected span.
assert not route(start+[(.5,-.3,z),(1,-1,z),(.5,0,z),(1,-1.2,z)])[0]
# One bight enters the gap, but the actual span also crosses outside it.
assert not route(start+[(.6,0,z),(.25,0,z),(.6,-.35,z)]+finish)[0]
# After the blue clip, a free tail may lie elsewhere without changing the route.
assert route(start+[(.5,0,z),(1,-1,z),(.5,-.6,z),(1,-1.2,z)])[0]
print('Connected-span gate, wrong-side, over-post, bight, free-tail and clip-seam checks passed.')
