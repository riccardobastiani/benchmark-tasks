"""Outcome/containment checks for the human-free clearance endpoint."""
import numpy as np
from rules import score,contained
assert score(True,[True,True],True,True)==1
assert score(True,[True,False],True,True)==.25
assert score(True,[True,True],False,True)==.5
assert score(True,[True,True],True,False)==.5
assert score(False,[True,True],True,True)==0
assert score(True,[False,False],True,True)==0
centre=np.zeros(3);half=np.array([.1,.1,.2])
assert contained([[0,0,.0078],[.02,.02,.18]],centre,half)
for point in ([0,0,.005],[.12,0,.02],[0,0,.22]):
 assert not contained([[0,0,.02],point],centre,half)
print('Whole-device containment, preservation, partial score and released placement endpoint checked.')
