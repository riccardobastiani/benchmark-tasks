"""Pure logical checks only; these do not establish physical feasibility."""
import numpy as np
from rules import Proof,Controller,opening_covered,shaft_point_speed
origin=np.zeros(3);axis=np.array([1.,0,0])
assert abs(shaft_point_speed((origin,axis),np.array([.001,0,0]),axis,.07,.01)-.1)<1e-12
assert shaft_point_speed((origin,axis),origin,np.array([np.cos(.1),np.sin(.1),0]),.07,.1)>.04
assert opening_covered(.007+np.arange(16)*.010+.050)
assert not opening_covered(.007+np.arange(16)*.010)
p=Proof();p.observe(.2,0.,.13,.080,True);assert p.fraction==0
p.observe(.2,0.,.13,.020,True);assert p.fraction>0
p.reset();assert p.fraction==0
c=Controller();b={n:False for n in ('mode','reset','start')}
c.step(.05,True,False,{**b,'mode':True},0.,0.);assert c.state=='SETUP'
c.step(.01,True,False,b,0.,0.)
for _ in range(12):c.step(.01,True,False,{**b,'mode':True},0.,0.)
assert c.state=='RESET_REQUIRED'
c.step(.01,True,False,b,0.,0.)
for _ in range(12):c.step(.01,True,False,{**b,'reset':True},0.,0.)
assert c.state=='ARMED' and c.target==0
c.step(.01,True,True,b,0.,0.)
for _ in range(12):c.step(.01,True,True,{**b,'start':True},0.,0.)
assert c.target==0 and c.state=='RESET_REQUIRED'
print('Pure coverage, excess-speed rejection, proof reset and reset/start separation checks passed.')
