"""Clear sharps from a post-procedure tray and stage reusable instruments for reprocessing."""
import json
from pathlib import Path
import tempfile
import numpy as np
from resist_isaac.state import pose_state,reset_joints
from resist_isaac.usd_assets import create_asset
from .geometry import geometry,TRAY_START,DOCK,BIN,BIN_INNER,HANDLE,SHARPS,TOOLS
from .rules import contained,tray_contains,score

CONDITIONS=('recoverable',)
FOLDER=Path(__file__).parent


def local_bounds(shapes):
    points=[]
    for shape in shapes:
        if not shape.get('collision',True):continue
        pos=np.array(shape.get('position',[0,0,0]),float)
        if shape['type']=='box':lo=pos-np.array(shape['size']);hi=pos+np.array(shape['size'])
        else:
            ends=np.array(shape['fromto']).reshape(2,3)+pos
            lo=ends.min(axis=0)-shape['size'][0];hi=ends.max(axis=0)+shape['size'][0]
        points += [[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])]
    return np.array(points)


class SurgicalTrayClearance:
    physics_hz=240
    robot_contact_offset=.0001
    hold_s=1.

    def __init__(self):
        self.folder=FOLDER;self.cfg=json.loads((FOLDER/'pilot.json').read_text())
        self.descriptions=geometry();self._subscription=None;self._views={};self._error=None
        self._warming_up=True
        self._bounds={name:local_bounds(self.descriptions[name]['bodies'][0]['shapes']) for name in (*SHARPS,*TOOLS)}
        self._robot_contacts={name:False for name in ('tray',*SHARPS,*TOOLS)}
        self._arm_contact_substeps=0
        self._clear_time=0.

    def set_evidence_directory(self,path):
        self.evidence_dir=Path(path);self._events=(self.evidence_dir/'native-clearance-events.jsonl').open('w')

    def build_scene(self,embodiment):
        from isaaclab_arena.assets.object import Object,ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.utils.pose import Pose
        if hasattr(self,'evidence_dir'):directory=self.evidence_dir/'assets'
        else:
            self._temporary=tempfile.TemporaryDirectory(prefix='surgical-clearance-');directory=Path(self._temporary.name)
        objects=[]
        for name,data in self.descriptions.items():
            path=create_asset(data,directory/(name+'.usda'),FOLDER)
            obj=Object(name=name,object_type=ObjectType.BASE,usd_path=str(path),initial_pose=Pose(position_xyz=(0,0,0)))
            objects.append(obj)
            for part in data['bodies']:
                if not part.get('static',False):objects.append(ObjectReference(parent_asset=obj,name=name+'_'+part['name'],
                    prim_path=obj.get_prim_path()+'/'+part['name'],object_type=ObjectType.RIGID))
        return objects

    def configure_native(self,cfg):
        from copy import deepcopy
        from isaaclab.sensors import CameraCfg,ContactSensorCfg
        from isaaclab.sim import PinholeCameraCfg
        from yam_arena.ik import matrix_to_quat_xyzw
        from yam_arena.robot import asset_link_paths
        eye=np.array([-.10,-.12,1.58]);back=eye-np.array([.40,.08,.82]);back/=np.linalg.norm(back)
        right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
        cfg.scene.close_cam=deepcopy(cfg.scene.top_cam);cfg.scene.close_cam.prim_path='{ENV_REGEX_NS}/CloseCam'
        cfg.scene.close_cam.offset=CameraCfg.OffsetCfg(pos=tuple(eye),
            rot=tuple(matrix_to_quat_xyzw(np.column_stack((right,up,back)))),convention='opengl')
        cfg.scene.close_cam.spawn=PinholeCameraCfg(focal_length=27.,horizontal_aperture=30.,clipping_range=(.02,20.))
        for name in ('left_cam','right_cam'):
            camera=getattr(cfg.scene,name);camera.width=min(camera.width,640);camera.height=min(camera.height,360)
        links=sorted(asset_link_paths().values());self._contact_defs=[];self._expected={}
        self._objects={'tray':'pan',**{n:'device' for n in SHARPS},**{n:'tool' for n in TOOLS}}
        hardware=[f'{{ENV_REGEX_NS}}/{name}/{part}' for name,part in self._objects.items()]
        for side,arm,other in [('left','LeftArm','RightArm'),('right','RightArm','LeftArm')]:
            getattr(cfg.scene,side+'_robot').spawn.activate_contact_sensors=True
            filters=hardware+['{ENV_REGEX_NS}/'+other+p for p in links]
            self._expected[side]={p.replace('{ENV_REGEX_NS}','/World/envs/env_0') for p in filters}
            for i,path in enumerate(links):
                name=f'clearance_contact_{side}_{i}'
                setattr(cfg.scene,name,ContactSensorCfg(prim_path='{ENV_REGEX_NS}/'+arm+path,
                    update_period=0.,history_length=0,track_contact_points=True,
                    max_contact_data_count_per_prim=1024,filter_prim_paths_expr=filters))
                self._contact_defs.append((name,side))

    def initial_action(self):
        from yam_arena import ik,station
        result=[]
        for side,y,rotation in [('left',.26,[0,1,0,0]),('right',-.26,[1,0,0,0])]:
            q,pe,re=ik.solve([.22,y,.96],rotation,base_pos=station.BASES[side],iters=500)
            if pe>.002 or re>.02:raise RuntimeError('Unreachable ready pose')
            result.extend([*q,1.])
        return np.array(result)

    def reset(self,base):
        from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS,FINGER_OPEN
        action=self.initial_action();self._base=base
        for side,i in [('left',0),('right',7)]:
            reset_joints(base,side+'_robot',dict(zip(ARM_JOINTS,action[i:i+6])))
            reset_joints(base,side+'_robot',dict.fromkeys(FINGER_JOINTS,FINGER_OPEN))
        self._subscription=None;self._error=None;self._warming_up=True
        self._clear_time=0.
        self._arm_contact_substeps=0;self._views={}
        for name,data in self.descriptions.items():
            for part in data['bodies']:
                if not part.get('static',False):self._views[(name,part['name'])]=base.scene[name+'_'+part['name']].root_physx_view
        self._robot_views=[]
        for name,side in self._contact_defs:
            view=base.scene.sensors[name].contact_view
            if view is None:raise RuntimeError('Missing native gripper/object contact view')
            filters=list(view.filter_paths)
            if len(filters)==1 and not isinstance(filters[0],str):filters=list(filters[0])
            if set(filters)!=self._expected[side]:raise RuntimeError('Incomplete robot contact coverage')
            self._robot_views.append((view,filters))
        from omni.physx import get_physx_interface
        self._subscription=get_physx_interface().subscribe_physics_on_step_events(self._after_step,pre_step=False,order=0)

    def begin_episode(self,base):
        self.validate_reset(base);self._warming_up=False

    def native(self,name,part=None):
        if part is None:part=self._objects[name]
        view=self._views[(name,part)];t=np.array(view.get_transforms().numpy()[0],copy=True)
        v=np.array(view.get_velocities().numpy()[0],copy=True)
        if not np.isfinite(np.r_[t,v]).all() or abs(np.dot(t[3:],t[3:])-1)>.001:raise RuntimeError('Invalid native pose/velocity')
        return t[:3],t[3:],v

    def pose_state(self,base,name,part=None):return pose_state(base.scene[name+'_'+part] if part else base.scene[name])

    def points(self,name):
        from yam_arena.urdf import quat_xyzw_to_matrix
        p,q,_=self.native(name);return self._bounds[name]@quat_xyzw_to_matrix(q).T+p

    def physical_state(self):
        from yam_arena.urdf import quat_xyzw_to_matrix
        p,q,v=self.native('tray');rot=quat_xyzw_to_matrix(q)
        tools=all(tray_contains(self.points(n),p,rot) for n in TOOLS)
        disposed=[contained(self.points(n),BIN,BIN_INNER) and not self._robot_contacts[n]
                  and np.linalg.norm(self.native(n)[2][:3])<.05 for n in SHARPS]
        docked=(np.linalg.norm(p[:2]-DOCK[:2])<.018 and abs(p[2]-DOCK[2])<.006
                and rot[2,2]>.995 and np.linalg.norm(v[:3])<.02)
        return p,tools,disposed,bool(docked)

    @staticmethod
    def native_contacts(view,filters):
        data=view.get_contact_data(dt=1.);counts=np.array(data[4].numpy(),copy=True).ravel()
        starts=np.array(data[5].numpy(),copy=True).ravel();gaps=np.array(data[3].numpy(),copy=True).ravel()
        if len(counts)!=len(filters) or counts.sum()>len(gaps):raise RuntimeError('Invalid native contact coverage/capacity')
        result=[]
        for path,start,count in zip(filters,starts,counts):
            start,count=int(start),int(count)
            if start<0 or start+count>len(gaps):raise RuntimeError('Invalid native contact range')
            if not np.isfinite(gaps[start:start+count]).all():raise RuntimeError('Nonfinite native contact gap')
            if count and np.any(gaps[start:start+count]<=1e-6):result.append(path)
        return result

    def _after_step(self,dt):
        try:
            self._robot_contacts={n:False for n in self._objects};arms=False
            for view,filters in self._robot_views:
                for path in self.native_contacts(view,filters):
                    if '/LeftArm/' in path or '/RightArm/' in path:arms=True
                    else:self._robot_contacts[path.split('/')[-2]]=True
            if not self._warming_up:
                self._arm_contact_substeps+=arms
                tray,tools,disposed,docked=self.physical_state()
                from yam_arena.runtime import tcp_world
                clear=(not any(self._robot_contacts.values()) and
                    all(np.linalg.norm(tcp_world(self._base.scene[s+'_robot'])-tray)>.30 for s in ('left','right')))
                self._clear_time=self._clear_time+dt if clear else 0.
            if hasattr(self,'_events'):
                self._events.write(json.dumps(dict(warmup=self._warming_up,
                    robot_contacts=self._robot_contacts,arm_contact=bool(arms),
                    tray_position=self.native('tray')[0].tolist(),clear_time_s=self._clear_time))+'\n')
        except Exception as error:self._error=repr(error)

    def evaluate(self,base):
        if self._error:raise RuntimeError(self._error)
        p,tools,disposed,docked=self.physical_state();released=self._clear_time>.15
        value=score(tools,disposed,docked,released)
        return dict(success=value==1.,score=value,
            tools_available=tools,disposed_count=sum(disposed),sharp_a_contained=disposed[0],sharp_b_contained=disposed[1],
            reprocessing_area=docked,hands_clear=released,arm_contact_substeps=self._arm_contact_substeps,
            tray_x_m=float(p[0]),tray_y_m=float(p[1]),tray_z_m=float(p[2]))

    def validate_reset(self,base):
        from yam_arena.urdf import quat_xyzw_to_matrix
        m=self.evaluate(base);p,q,_=self.native('tray');rot=quat_xyzw_to_matrix(q)
        if (not m['tools_available'] or m['disposed_count'] or m['reprocessing_area']
                or not all(tray_contains(self.points(n),p,rot) for n in SHARPS)):
            raise RuntimeError('Invalid surgical-tray reset: '+repr(m))

    def scripted_actions(self,base,policy,max_steps):
        from .interaction import actions
        for action,phase,errors in actions(self,base,policy,max_steps):
            if self._arm_contact_substeps:raise RuntimeError('Careful controller stopped after inter-arm contact')
            yield action,phase,errors

    def close(self):
        self._subscription=None
        if hasattr(self,'_events'):self._events.flush();self._events.close()


def make_task(condition):
    if condition not in CONDITIONS:raise ValueError(condition)
    return SurgicalTrayClearance()


def create_assets(directory):return {name:create_asset(data,Path(directory)/(name+'.usda'),FOLDER) for name,data in geometry().items()}
