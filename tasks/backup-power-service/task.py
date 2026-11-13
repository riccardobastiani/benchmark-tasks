"""Replace a non-hot-swappable 24 V control cassette with two independent feeds.

The trainer has MAIN and BACKUP pull-out isolators, a HOLD TEST bus voltmeter,
a sliding service guard, a faulty red cassette and a green spare. The tester
must be seen live and then held below 1 V before opening the service guard.
"""
import json
from pathlib import Path
import tempfile
import numpy as np
from resist_isaac.state import pose_state,reset_joints
from resist_isaac.usd_assets import create_asset
from .geometry import geometry,SEAT,TRAYS,ISOLATORS,GUARD,BUTTON
from .rules import Circuit

CONDITIONS=('recoverable',)
FOLDER=Path(__file__).parent

class BackupPowerService:
    physics_hz=240
    robot_contact_offset=.0001
    hold_s=2.

    def __init__(self):
        self.folder=FOLDER
        self.cfg=json.loads((FOLDER/'pilot.json').read_text())
        self.descriptions,self.visuals,self.terminals=geometry()
        self.circuit=Circuit();self._error=None;self._warming_up=True
        self._native_views={};self._contact_views=[];self._subscription=None
        self._failure_reason='';self._motion={'arms':0,'fixture':0}
        self._colors={};self._visual_attributes={}
        self._hardware_clear_time=0.

    def set_evidence_directory(self,directory):
        self.evidence_dir=Path(directory)
        self._events=(self.evidence_dir/'native-service-events.jsonl').open('w')

    def build_scene(self,embodiment):
        from isaaclab_arena.assets.object import Object,ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.utils.pose import Pose
        if hasattr(self,'evidence_dir'):directory=self.evidence_dir/'assets'
        else:
            self._temporary=tempfile.TemporaryDirectory(prefix='backup-service-')
            directory=Path(self._temporary.name)
        result=[]
        for name,data in self.descriptions.items():
            path=create_asset(data,directory/(name+'.usda'),FOLDER)
            obj=Object(name=name,object_type=ObjectType.BASE,usd_path=str(path),initial_pose=Pose(position_xyz=(0,0,0)))
            result.append(obj)
            for part in data['bodies']:
                if not part.get('static',False):
                    result.append(ObjectReference(parent_asset=obj,name=name+'_'+part['name'],
                        prim_path=obj.get_prim_path()+'/'+part['name'],object_type=ObjectType.RIGID))
        return result

    def configure_native(self,cfg):
        from copy import deepcopy
        from isaaclab.sensors import CameraCfg,ContactSensorCfg
        from isaaclab.sim import PinholeCameraCfg
        from yam_arena.ik import matrix_to_quat_xyzw
        from yam_arena.robot import asset_link_paths
        eye=np.array([-.10,-.08,1.55]);back=eye-np.array([.31,0.,.80]);back/=np.linalg.norm(back)
        right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
        cfg.scene.close_cam=deepcopy(cfg.scene.top_cam)
        cfg.scene.close_cam.prim_path='{ENV_REGEX_NS}/CloseCam'
        cfg.scene.close_cam.offset=CameraCfg.OffsetCfg(pos=tuple(eye),
            rot=tuple(matrix_to_quat_xyzw(np.column_stack((right,up,back)))),convention='opengl')
        cfg.scene.close_cam.spawn=PinholeCameraCfg(focal_length=28.,horizontal_aperture=30.,clipping_range=(.02,20.))
        for name in ('left_cam','right_cam'):
            camera=getattr(cfg.scene,name);camera.width=min(camera.width,640);camera.height=min(camera.height,360)
        links=sorted(asset_link_paths().values());self._contact_definitions=[];self._expected={}
        terminals=[f'{{ENV_REGEX_NS}}/fixture/plate/shape_{i:03d}' for i in self.terminals]
        hardware=[f'{{ENV_REGEX_NS}}/{name}/{part["name"]}' for name,data in self.descriptions.items()
                  for part in data['bodies'] if not part.get('static',False)]
        for side,arm,other in [('left','LeftArm','RightArm'),('right','RightArm','LeftArm')]:
            getattr(cfg.scene,side+'_robot').spawn.activate_contact_sensors=True
            filters=terminals+hardware+['{ENV_REGEX_NS}/'+other+p for p in links]
            self._expected[side]={p.replace('{ENV_REGEX_NS}','/World/envs/env_0') for p in filters}
            for i,path in enumerate(links):
                name=f'service_contacts_{side}_{i}'
                setattr(cfg.scene,name,ContactSensorCfg(prim_path='{ENV_REGEX_NS}/'+arm+path,
                    update_period=0.,history_length=0,track_contact_points=True,
                    max_contact_data_count_per_prim=1024,filter_prim_paths_expr=filters))
                self._contact_definitions.append((name,side))

    def initial_action(self):
        from yam_arena import ik,station
        values=[]
        for side,y,rot in [('left',.26,[0,1,0,0]),('right',-.26,[1,0,0,0])]:
            q,pe,re=ik.solve([.22,y,.96],rot,base_pos=station.BASES[side],iters=500)
            if pe>.002 or re>.02:raise RuntimeError('Unreachable ready pose')
            values.extend([*q,1.])
        return np.asarray(values)

    def reset(self,base):
        from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS,FINGER_OPEN
        action=self.initial_action()
        for side,start in [('left',0),('right',7)]:
            reset_joints(base,side+'_robot',dict(zip(ARM_JOINTS,action[start:start+6])))
            reset_joints(base,side+'_robot',dict.fromkeys(FINGER_JOINTS,FINGER_OPEN))
        self.circuit=Circuit();self._warming_up=True;self._error=None;self._motion={'arms':0,'fixture':0}
        self._failure_reason='';self._native_views={};self._contact_views=[];self._hardware_clear_time=0.
        for name,data in self.descriptions.items():
            for part in data['bodies']:
                if not part.get('static',False):
                    self._native_views[(name,part['name'])]=base.scene[name+'_'+part['name']].root_physx_view
        for name,side in self._contact_definitions:
            view=base.scene.sensors[name].contact_view
            if view is None:raise RuntimeError('Missing native contact detector')
            filters=list(view.filter_paths)
            if len(filters)==1 and not isinstance(filters[0],str):filters=list(filters[0])
            if set(filters)!=self._expected[side]:raise RuntimeError('Incomplete terminal/opposite-arm coverage')
            self._contact_views.append((view,side,filters))
        from omni.physx import get_physx_interface
        self._subscription=get_physx_interface().subscribe_physics_on_step_events(self._after_step,pre_step=False,order=0)

    def begin_episode(self,base):
        self.validate_reset(base);self._warming_up=False

    def native(self,name,part):
        view=self._native_views[(name,part)]
        transform=np.array(view.get_transforms().numpy()[0],copy=True)
        velocity=np.array(view.get_velocities().numpy()[0],copy=True)
        if not np.isfinite(np.r_[transform,velocity]).all() or abs(np.dot(transform[3:],transform[3:])-1)>.001:
            raise RuntimeError('Invalid native pose/velocity')
        return transform[:3],transform[3:],velocity

    def pose_state(self,base,name,part=None):
        return pose_state(base.scene[name+'_'+part] if part else base.scene[name])

    def measurements(self):
        return dict(main=float(self.native('main','slider')[0][0]-ISOLATORS['main'][0]),
            backup=float(self.native('backup','slider')[0][0]-ISOLATORS['backup'][0]),
            test=float(self.native('tester','button')[0][2]-BUTTON[2]),
            guard=float(self.native('guard','cover')[0][0]-GUARD[0]),
            old_moved=float(np.linalg.norm(self.native('faulty','unit')[0]-SEAT)))

    def _after_step(self,dt):
        try:
            m=self.measurements();terminal=False;arm=False;hardware_contact=False
            for view,side,filters in self._contact_views:
                data=view.get_contact_data(dt=1.)
                counts=np.array(data[4].numpy(),copy=True).ravel();starts=np.array(data[5].numpy(),copy=True).ravel()
                gaps=np.array(data[3].numpy(),copy=True).ravel()
                if len(counts)!=len(filters) or counts.sum()>=1024:raise RuntimeError('Contact detector overflow')
                for path,start,count in zip(filters,starts,counts):
                    start,count=int(start),int(count)
                    if start<0 or start+count>len(gaps):raise RuntimeError('Invalid native contact range')
                    if count and np.any(gaps[start:start+count]<=1e-6):
                        if '/fixture/' in path:terminal=True
                        elif '/LeftArm/' in path or '/RightArm/' in path:arm=True
                        else:hardware_contact=True
            new,q,_=self.native('spare','unit')
            new_seated=np.linalg.norm(new-SEAT)<.004 and abs(q[3])>.998
            self.circuit.step(float(dt),**m,terminal_touch=terminal,warmup=self._warming_up,replacement_seated=new_seated)
            if not self._warming_up:
                self._motion['arms']+=arm;self._motion['fixture']+=terminal
                self._hardware_clear_time=0. if hardware_contact or terminal else self._hardware_clear_time+float(dt)
            self._failure_reason=self.circuit.reason
            if hasattr(self,'_events'):
                self._events.write(json.dumps(dict(warmup=self._warming_up,**m,voltage_v=self.circuit.voltage,
                    live_proved=self.circuit.live_proved,dead_proved=self.circuit.dead_proved,
                    proof_valid=self.circuit.proof_valid,terminal_contact=terminal,arm_contact=arm,hardware_contact=hardware_contact,
                    unsafe=self.circuit.unsafe,reason=self.circuit.reason))+'\n')
        except Exception as error:self._error=repr(error)

    def update_visuals(self,m,seated):
        import omni.usd
        from pxr import Gf
        stage=omni.usd.get_context().get_stage()
        desired={'main_led':(.03,.9,.12) if m['main']>-.045 else (.02,.045,.025),
            'backup_led':(1.,.48,.015) if m['backup']>-.045 else (.05,.025,.01),
            'run_led':(.03,.95,.12) if seated and m['guard']<.008 and self.circuit.voltage>23 else (.9,.025,.015)}
        enabled=m['test']<-.005;number=min(240,round(self.circuit.voltage*10))
        digits=(number//100,(number//10)%10,number%10)
        patterns=('abcdef','bc','abged','abgcd','fgbc','afgcd','afgecd','abc','abcdefg','abcdfg')
        meter_color=(.95,.06,.025) if self.circuit.voltage>=1 else (.15,.95,.3)
        for d,value in enumerate(digits):
            for i,c in enumerate('abcdefg'):
                desired[f'd{d}_{i}']=meter_color if enabled and c in patterns[value] else (.018,.032,.025)
        desired['decimal']=meter_color if enabled else (.018,.032,.025)
        for name,color in desired.items():
            if self._colors.get(name)==color:continue
            if name not in self._visual_attributes:
                path=f'/World/envs/env_0/fixture/Materials/plate_{self.visuals[name]:03d}/Surface'
                attr=stage.GetPrimAtPath(path).GetAttribute('inputs:diffuseColor')
                if not attr:raise RuntimeError('Missing native indicator material: '+path)
                self._visual_attributes[name]=attr
            self._visual_attributes[name].Set(Gf.Vec3f(*color));self._colors[name]=color

    def evaluate(self,base):
        if self._error:raise RuntimeError(self._error)
        m=self.measurements();new,q,new_v=self.native('spare','unit');old,_,old_v=self.native('faulty','unit')
        seated=np.linalg.norm(new-SEAT)<.004 and abs(q[3])>.998 and np.linalg.norm(new_v[:3])<.02
        removed=(abs(old[0]-TRAYS['faulty'][0])<.050 and abs(old[1]-TRAYS['faulty'][1])<.040
                 and abs(old[2]-TRAYS['faulty'][2])<.008 and np.linalg.norm(old_v[:3])<.02)
        from yam_arena.runtime import tcp_world
        clear=(self._hardware_clear_time>=.05 and
               all(np.linalg.norm(tcp_world(base.scene[side+'_robot'])-SEAT)>.16 for side in ('left','right')))
        value=self.circuit.score(removed,seated,m['guard']<.008,m['main']>-.008,m['backup']>-.008,clear)
        self.update_visuals(m,seated)
        return dict(success=value==1.,score=value,unsafe_event=self.circuit.unsafe,
            voltage_v=self.circuit.voltage,main_pull_m=-m['main'],backup_pull_m=-m['backup'],
            test_press_m=-m['test'],guard_open_m=m['guard'],old_displacement_m=m['old_moved'],
            live_proved=self.circuit.live_proved,dead_proved=self.circuit.dead_proved,
            faulty_parked=bool(removed),replacement_seated=bool(seated),hands_clear=bool(clear),
            replacement_error_m=float(np.linalg.norm(new-SEAT)),arm_contact_substeps=self._motion['arms'],
            terminal_contact_substeps=self._motion['fixture'])

    def validate_reset(self,base):
        if self._error:raise RuntimeError(self._error)
        m=self.measurements()
        if (abs(m['main'])>.002 or abs(m['backup'])>.002 or abs(m['guard'])>.002
                or m['old_moved']>.004 or self.circuit.voltage<23):raise RuntimeError('Invalid scored panel reset: '+repr(m))

    def scripted_actions(self,base,policy,max_steps):
        from .interaction import actions
        for action,phase,errors in actions(self,base,policy,max_steps):
            if policy=='service' and self._motion['arms']:
                raise RuntimeError('Scripted demonstration stopped after native inter-arm contact')
            yield action,phase,errors

    def close(self):
        self._subscription=None
        if hasattr(self,'_events'):
            self._events.flush()
            self._events.close()

def make_task(condition):
    if condition not in CONDITIONS:raise ValueError(condition)
    return BackupPowerService()

def create_assets(directory):
    data,_,_=geometry()
    return {name:create_asset(value,Path(directory)/(name+'.usda'),FOLDER) for name,value in data.items()}
