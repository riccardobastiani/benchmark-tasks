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
# Routing earns partial credit independently of arm participation.
assert r.score(False,True,True,True,False)==.5
assert r.score(False,True,False,False,False)==.5
# Sequential handling, interrupted contact and different wire locations count.
use=r.GripperParticipation()
for contacts in [{'left':{2},'right':set()}, {'left':set(),'right':set()},
                 {'left':set(),'right':{40}}, {'left':{3},'right':set()},
                 {'left':set(),'right':{41}}]:
 use.record(contacts,.1)
assert use.complete
assert r.score(False,True,True,True,use.complete)==1
# Simultaneous contact can happen anywhere on the wire; no clip predicate.
use=r.GripperParticipation()
for _ in range(48):use.record({'left':{1},'right':{4}},1/240)
assert use.complete
assert r.score(False,True,True,True,use.complete)==1
assert r.score(True,True,True,True,use.complete)==0
# One arm, fleeting second-arm contact, and a fresh trial cannot earn full score.
use=r.GripperParticipation();use.record({'left':{3},'right':set()},1.)
use.record({'left':set(),'right':{20}},.1)
assert not use.complete and r.score(False,True,True,True,use.complete)==.5
assert not r.GripperParticipation().complete
for bad_dt in (float('nan'),float('inf'),0.,-.1):
 try:use.record({'left':set(),'right':set()},bad_dt)
 except ValueError:pass
 else:raise AssertionError('Invalid contact measurement accepted')
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
