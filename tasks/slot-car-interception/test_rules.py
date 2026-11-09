"""Small scoring/contact/reset check; native execution separately validates physics."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).parent
spec=importlib.util.spec_from_file_location('_interception_rules',root/'task.py',submodule_search_locations=[str(root)])
task=importlib.util.module_from_spec(spec);sys.modules[spec.name]=task;spec.loader.exec_module(task)
from _interception_rules.geometry import lane_pose,track_coordinates,LENGTH,geometry,INITIAL_S
for y,b,want in [(0,0,0),(1,0,.5),(0,1,.5),(1,1,1)]:
 assert task.score({'yellow':y,'blue':b},False)==want
 assert task.score({'yellow':y,'blue':b},True)==0
assert task.permitted_contact('yellow','/World/envs/env_0/LeftArm/tip_left')
assert task.permitted_contact('blue','/World/envs/env_0/RightArm/gripper')
for car,path in [('red','LeftArm/tip_left'),('yellow','RightArm/tip_right'),('blue','LeftArm/tip_left'),('yellow','LeftArm/link4'),('yellow','red/chassis')]:
 assert not task.permitted_contact(car,'/World/envs/env_0/'+path)
for s in np.linspace(0,LENGTH,100,endpoint=False):
 p,yaw=lane_pose(s); offset,actual=track_coordinates(p)
 assert abs(offset)<1e-8 and abs((actual-s+LENGTH/2)%LENGTH-LENGTH/2)<1e-8
assets,poses=geometry()
for name,s in INITIAL_S.items():
 p,yaw=lane_pose(s);guide=np.array(poses[name][:2])+.044*np.array([np.cos(yaw),np.sin(yaw)])
 np.testing.assert_allclose(guide,p)
print('Scoring, assigned-contact permissions and three-car lane resets passed.')
