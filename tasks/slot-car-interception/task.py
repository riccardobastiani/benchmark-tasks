"""Intercept yellow with the left arm and blue with the right; keep red running."""
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from resist_isaac.state import pose_state
from resist_isaac.usd_assets import create_asset
from .geometry import geometry,track_coordinates,lane_pose,INITIAL_S,PADS,ROAD_Z,CAR_Z,WHEEL_RADIUS,RADIUS,HALF_STRAIGHT

CONDITIONS=('recoverable',)
FOLDER=Path(__file__).parent
CARS=('red','yellow','blue')
PARTS=('chassis','wheel0','wheel1','wheel2','wheel3','guide')
ASSIGNED={'yellow':'LeftArm','blue':'RightArm'}

def assigned_release(name,robot_arms,gripper_arms):
    arm=ASSIGNED[name]
    return robot_arms=={arm} and arm in gripper_arms


def permitted_contact(name,other):
    owner=ASSIGNED.get(name)
    return bool(owner and '/'+owner+'/' in other
                and other.rsplit('/',1)[-1] in ('tip_left','tip_right','gripper'))


def score(reinserted,touched_running):
    """The user's entire scalar rule; no hidden lap/time/clearance bonus."""
    return 0. if touched_running else .5*sum(bool(reinserted[name]) for name in ('yellow','blue'))


class SlotCarInterception:
    physics_hz=240
    robot_contact_offset=.0001
    hold_s=1.0

    def __init__(self):
        self.folder=FOLDER
        self.cfg=json.loads((FOLDER/'pilot.json').read_text())
        self.descriptions,self.poses=geometry()
        self._unsafe=False;self._subscription=None;self._error=None
        self._running={name:True for name in CARS}
        self._inserted={'yellow':False,'blue':False}
        self._last_handler={name:False for name in ASSIGNED}
        self._released_by_assigned={name:False for name in ASSIGNED}
        self._distance={name:0. for name in CARS};self._last_s={}
        self._power={name:None for name in CARS}
        self._drives={};self._views={};self._contact_file=None
        self._contact_filters={};self._contact_views={}

    def build_scene(self,embodiment):
        from isaaclab_arena.assets.object import Object,ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.utils.pose import Pose
        from pxr import Usd,UsdPhysics,PhysxSchema
        self._temporary=tempfile.TemporaryDirectory(prefix='slot-car-')
        objects=[]
        for name,data in self.descriptions.items():
            path=create_asset(data,Path(self._temporary.name)/(name+'.usda'),FOLDER)
            if name in CARS:
                stage=Usd.Stage.Open(str(path))
                for part in PARTS:
                    prim=stage.GetPrimAtPath('/Car/'+part)
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
                # Low-grip front tires are a real slot-car preparation, unlike
                # steering the chassis with an invisible path-following force.
                for i in (0,1):
                    material=stage.GetPrimAtPath(f'/Car/Materials/wheel{i}_000')
                    PhysxSchema.PhysxMaterialAPI.Apply(material).CreateFrictionCombineModeAttr('min')
                for i in range(len(data['bodies'][-1]['shapes'])):
                    material=stage.GetPrimAtPath(f'/Car/Materials/guide_{i:03d}')
                    PhysxSchema.PhysxMaterialAPI.Apply(material).CreateFrictionCombineModeAttr('min')
                for i in (2,3):
                    drive=UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath('/Car/axle'+str(i)),'angular')
                    drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(0.)
                    drive.CreateDampingAttr(.0003/(180/math.pi))
                    drive.CreateMaxForceAttr(0.);drive.CreateTargetVelocityAttr(0.)
                stage.GetRootLayer().Save()
            rotation=(0,0,0,1)
            if name in CARS:
                _,yaw=lane_pose(INITIAL_S[name])
                rotation=(0.,0.,math.sin(yaw/2),math.cos(yaw/2))
            obj=Object(name=name,object_type=ObjectType.BASE,usd_path=str(path),
                initial_pose=Pose(position_xyz=tuple(self.poses[name]),
                    rotation_xyzw=rotation))
            objects.append(obj)
            if name in CARS:
                for part in PARTS:
                    objects.append(ObjectReference(parent_asset=obj,name=name+'_'+part,
                        prim_path=obj.get_prim_path()+'/'+part,object_type=ObjectType.RIGID))
        return objects

    def configure_native(self,cfg):
        from isaaclab.sensors import ContactSensorCfg,CameraCfg
        from isaaclab.sim import PinholeCameraCfg
        from copy import deepcopy
        from yam_arena.ik import matrix_to_quat_xyzw
        eye=np.array([.88,-.77,1.42]); back=eye-np.array([.40,0.,.78]);back/=np.linalg.norm(back)
        right=np.cross([0.,0.,1.],back);right/=np.linalg.norm(right);up=np.cross(back,right)
        cfg.scene.close_cam=deepcopy(cfg.scene.top_cam)
        cfg.scene.close_cam.prim_path='{ENV_REGEX_NS}/CloseCam'
        cfg.scene.close_cam.offset=CameraCfg.OffsetCfg(pos=tuple(eye),
            rot=tuple(matrix_to_quat_xyzw(np.column_stack((right,up,back)))),convention='opengl')
        cfg.scene.close_cam.spawn=PinholeCameraCfg(focal_length=25.,horizontal_aperture=30.,clipping_range=(.03,20.))
        from yam_arena.robot import asset_link_paths
        robots=['{ENV_REGEX_NS}/'+arm+path for arm in ('LeftArm','RightArm') for path in sorted(asset_link_paths().values())]
        for name in CARS:
            filters=robots+['{ENV_REGEX_NS}/'+other+'/'+p for other in CARS if other!=name for p in PARTS]
            self._contact_filters[name]=[p.replace('{ENV_REGEX_NS}','/World/envs/env_0') for p in filters]
            for part in PARTS:
                setattr(cfg.scene,f'contacts_{name}_{part}',ContactSensorCfg(
                    prim_path='{ENV_REGEX_NS}/'+name+'/'+part,update_period=0.,history_length=0,
                    track_contact_points=True,max_contact_data_count_per_prim=512,
                    filter_prim_paths_expr=filters))
        # Calibration evidence only: expose actual guide/road contact points,
        # which are deliberately excluded from the no-touch safety filters.
        self._road_filters=[f'{{ENV_REGEX_NS}}/fixture/track/shape_{i:03d}'
            for i,shape in enumerate(self.descriptions['fixture']['bodies'][0]['shapes'])
            if shape.get('collision',True)]+['{ENV_REGEX_NS}/table']
        cfg.scene.guide_road_contacts=ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/red/guide',update_period=0.,history_length=0,
            track_contact_points=True,
            max_contact_data_count_per_prim=1024,filter_prim_paths_expr=self._road_filters)

    def initial_action(self):
        from yam_arena import ik,station
        result=[]
        for side,y in (('left',.26),('right',-.26)):
            rotation=[0,1,0,0] if side=='left' else [1,0,0,0]
            q,pe,re=ik.solve([.22,y,.96],rotation,base_pos=station.BASES[side],iters=500)
            if pe>.002 or re>.02:raise ValueError(f'Initial {side} IK failed: {pe}, {re}')
            result.extend([*q,1.])
        return np.asarray(result)

    def reset(self,base):
        from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS,FINGER_OPEN
        from resist_isaac.state import reset_joints
        self._subscription=None
        self._unsafe=False;self._error=None
        self._running={name:True for name in CARS}
        self._inserted={'yellow':False,'blue':False}
        self._last_handler={name:False for name in ASSIGNED}
        self._released_by_assigned={name:False for name in ASSIGNED}
        self._distance={name:0. for name in CARS};self._last_s={}
        self._power={name:None for name in CARS}
        action=self.initial_action()
        for side,start in (('left',0),('right',7)):
            reset_joints(base,side+'_robot',dict(zip(ARM_JOINTS,action[start:start+6])))
            reset_joints(base,side+'_robot',dict.fromkeys(FINGER_JOINTS,FINGER_OPEN))
        self._captured={name:False for name in ASSIGNED}
        self._extracted={name:False for name in ASSIGNED}
        self._low_speed={name:0. for name in CARS}
        self._failure_reason=''
        self._warming_up=True
        self._initialize_native(base)

    def pose_state(self,base,name,part=None):
        return pose_state(base.scene[name+'_'+part])

    def set_evidence_directory(self,folder):
        self._contact_file=(folder/'native-car-events.jsonl').open('w')

    def begin_episode(self,base):
        if self._error or self._unsafe:
            raise RuntimeError(f'Invalid slot-car warmup: observer={self._error}, contact={self._unsafe}')
        self._warming_up=False
        self._step=0
        self._distance={name:0. for name in CARS}
        self._last_s.clear()

    def _initialize_native(self,base):
        import omni.usd
        from omni.physx import get_physx_interface
        from pxr import UsdPhysics
        self._base=base;self._step=0
        self._subscription=None
        base.sim.set_setting('/physics/disableContactProcessing',False)
        if base.sim.get_setting('/physics/disableContactProcessing'):
            raise RuntimeError('Contact processing remains disabled')
        stage=omni.usd.get_context().get_stage()
        self._guide_debug=base.scene.sensors['guide_road_contacts'].contact_view
        if self._guide_debug is None:raise RuntimeError('Guide diagnostic contact view unavailable')
        raw=list(self._guide_debug.filter_paths)
        self._guide_filters=list(raw[0]) if len(raw)==1 and not isinstance(raw[0],str) else raw
        expected=[p.replace('{ENV_REGEX_NS}','/World/envs/env_0') for p in self._road_filters]
        if set(self._guide_filters)!=set(expected) or len(self._guide_filters)!=len(expected):
            raise RuntimeError('Native guide/road diagnostic filters did not resolve exactly')
        if self._contact_file:
            self._contact_file.write(json.dumps(dict(event='guide_diagnostic_configuration',
                sensors=list(self._guide_debug.sensor_paths),filters=list(self._guide_debug.filter_paths)))+'\n')
        for name in CARS:
            for part in PARTS:
                self._views[name,part]=base.scene[name+'_'+part].root_physx_view
                view=base.scene.sensors[f'contacts_{name}_{part}'].contact_view
                if view is None:raise RuntimeError('Native car contact view unavailable')
                raw=list(view.filter_paths)
                filters=list(raw[0]) if len(raw)==1 and not isinstance(raw[0],str) else raw
                expected=self._contact_filters[name]
                if (list(view.sensor_paths)!=[f'/World/envs/env_0/{name}/{part}']
                        or set(filters)!=set(expected) or len(filters)!=len(expected)
                        or view.filter_count!=len(expected)):
                    raise RuntimeError(f'Native contact filters do not cover all cars/robot links: {name}/{part}')
                self._contact_views[name,part]=(view,filters)
                if self._contact_file:
                    self._contact_file.write(json.dumps(dict(event='contact_configuration',
                        sensor=list(view.sensor_paths),filters=filters,capacity=512))+'\n')
            for i in (2,3):
                prim=stage.GetPrimAtPath('/World/envs/env_0/'+name+'/axle'+str(i))
                if not prim.IsValid():raise RuntimeError('Missing native wheel drive')
                self._drives[name,i]=UsdPhysics.DriveAPI.Get(prim,'angular')
                self._drives[name,i].GetMaxForceAttr().Set(0.)
                self._drives[name,i].GetTargetVelocityAttr().Set(0.)
        self._subscription=get_physx_interface().subscribe_physics_on_step_events(self._after_step,pre_step=False,order=0)

    def _after_step(self,dt):
        try:self._observe_native(dt)
        except Exception as error:
            self._error=repr(error)
            # Surface callback exceptions on the next control tick; never turn
            # unavailable safety measurements into a successful episode.

    def _observe_native(self,dt):
        from yam_arena.urdf import quat_xyzw_to_matrix
        self._step+=1; events=[]; states={};contacted={name:False for name in CARS}
        for name in CARS:
            net_impulses={};robot_arms=set();gripper_arms=set()
            for part in PARTS:
                view,filters=self._contact_views[name,part]
                if name=='red':
                    net_impulses[part]=np.array(view.get_net_contact_forces(dt=1.).numpy()[0],copy=True).tolist()
                data=view.get_contact_data(dt=1.)
                counts=np.array(data[4].numpy(),copy=True)
                if counts.shape!=(1,len(filters)):raise RuntimeError('Unexpected native contact count shape')
                if np.sum(counts)>=512:raise RuntimeError('Car contact buffer overflow')
                if not np.any(counts):continue
                separations=np.array(data[3].numpy(),copy=True)
                starts=np.array(data[5].numpy(),copy=True)
                # Filters exclude own wheels/guide and all normal track contact.
                for other,start,count in zip(filters,starts.ravel(),counts.ravel()):
                    start,count=int(start),int(count)
                    if start<0 or start+count>len(separations):raise RuntimeError('Invalid contact buffer range')
                    touching=count and np.any(separations[start:start+count]<=1e-6)
                    if touching:
                        contacted[name]=True
                        for arm in ('LeftArm','RightArm'):
                            if '/'+arm+'/' in other:
                                robot_arms.add(arm)
                                if other.rsplit('/',1)[-1] in ('tip_left','tip_right'):
                                    gripper_arms.add(arm)
                        points=np.array(data[1].numpy(),copy=True)[start:start+count]
                        gaps=separations[start:start+count]
                        if not np.isfinite(points).all() or not np.isfinite(gaps).all():
                            raise RuntimeError('Nonfinite native contact measurement')
                        events.append(dict(event='contact',car=name,part=part,other=other,
                            protected=name=='red',points=points.tolist(),separation_m=gaps.tolist()))
                        allowed=permitted_contact(name,other)
                        if not allowed:
                            self._unsafe=True;self._failure_reason='forbidden_contact'

            if name in ASSIGNED and robot_arms:
                if assigned_release(name,robot_arms,gripper_arms):
                    self._captured[name]=True
                elif robot_arms != {ASSIGNED[name]}:
                    self._unsafe=True;self._failure_reason='wrong_arm'

            p=np.array(self._views[name,'guide'].get_transforms().numpy()[0],copy=True)
            v=np.array(self._views[name,'chassis'].get_velocities().numpy()[0],copy=True)
            if not np.isfinite(np.r_[p,v]).all():raise RuntimeError('Nonfinite car state')
            offset,s=track_coordinates(p)
            # 3 mm pickup braids overlap the 2 mm conductors across 2.5 mm
            # lateral offset. Blade/slot clearance is a separate contact problem.
            powered=bool(abs(offset)<.0025 and abs(p[2]-CAR_Z)<.0015)
            # This switch models rail/pickup electrical engagement only. Native
            # contacts determine motion and physical slot capture independently.
            if powered!=self._power[name]:
                target=(.25/WHEEL_RADIUS)*(180/math.pi) if powered else 0.
                for i in (2,3):
                    self._drives[name,i].GetMaxForceAttr().Set(.002 if powered else 0.)
                    self._drives[name,i].GetTargetVelocityAttr().Set(target)
                self._power[name]=powered
            length=4*HALF_STRAIGHT+2*math.pi*RADIUS
            delta=0. if name not in self._last_s else (s-self._last_s[name]+length/2)%length-length/2
            self._last_s[name]=s
            forward=quat_xyzw_to_matrix(p[3:])[:,0]
            moving=powered and float(np.dot(v[:3],forward))>.04
            rotation=quat_xyzw_to_matrix(p[3:])
            guide_bottom=float(p[2]+(rotation@np.array([0.,0.,-.009]))[2]
                               -np.linalg.norm(rotation[2]*np.array([.006,.0008,.0025])))
            forward_speed=float(np.dot(v[:3],forward))
            if not self._warming_up:
                must_move=name=='red' or not self._extracted[name]
                if must_move:
                    self._low_speed[name]=self._low_speed[name]+dt if forward_speed<.04 else 0.
                    if self._low_speed[name]>(.20 if name=='red' else .10):
                        self._unsafe=True;self._failure_reason='car_stopped_on_lane:'+name
                if name in ASSIGNED and not self._extracted[name] and guide_bottom>ROAD_Z+.002:
                    if (not self._captured[name] or forward_speed<.08
                            or not assigned_release(name,robot_arms,gripper_arms)):
                        self._unsafe=True;self._failure_reason='invalid_extraction'
                    else:
                        self._extracted[name]=True
                        events.append(dict(event='moving_extraction',car=name,speed_m_s=forward_speed))
                if name=='red' and not powered:
                    self._unsafe=True;self._failure_reason='red_lost_lane'
            if moving:self._distance[name]+=max(0.,delta)
            states[name]=dict(guide_pose=p.tolist(),velocity=v.tolist(),power=powered,
                running=self._running[name],distance_m=self._distance[name],contact=contacted[name],
                native_net_contact_impulse_Ns=net_impulses)
        self._latest_states=states
        if self._contact_file:
            normal=[np.array(v.numpy(),copy=True) for v in self._guide_debug.get_contact_data(dt=1.)]
            guide_contacts=[]
            for collider,start,count in zip(self._guide_filters,normal[5].ravel(),normal[4].ravel()):
                start,count=int(start),int(count)
                if start<0 or start+count>len(normal[0]):raise RuntimeError('Invalid native guide contact range')
                for i in range(start,start+count):
                    guide_contacts.append(dict(collider=collider,point=normal[1][i].tolist(),normal=normal[2][i].tolist(),
                        impulse_Ns=float(normal[0][i,0]),separation_m=float(normal[3][i,0])))
            self._contact_file.write(json.dumps(dict(phase='warmup' if self._warming_up else 'episode',
                step=self._step,time_s=self._step*dt,cars=states,events=events,unsafe=self._unsafe,
                guide_contacts=guide_contacts))+'\n')

    def evaluate(self,base):
        if self._error:raise RuntimeError(self._error)
        from yam_arena.runtime import tcp_world
        parked={}; result={}
        for name in CARS:
            p,q,v=self.pose_state(base,name,'chassis')
            p,q,v=[x.cpu().numpy() for x in (p,q,v)]
            result[name+'_speed_m_s']=float(np.linalg.norm(v[:3]))
            result[name+'_powered']=bool(self._power[name])
            if name in ASSIGNED:
                pad=np.asarray(PADS[name]); near=bool(np.all(np.abs(p[:2]-pad)<[.030,.030]))
                # Actual released contact state comes from the native substep observer.
                contact=bool(self._latest_states[name]['contact']) if hasattr(self,'_latest_states') else True
                parked[name]=bool(self._extracted[name] and near and .751<p[2]<.79
                                  and np.linalg.norm(v[:3])<.02 and np.linalg.norm(v[3:])<.3 and not contact)
                result[name+'_extracted']=self._extracted[name]
                result[name+'_assigned_capture']=self._captured[name]
                result[name+'_parked']=parked[name]
        clear=all(abs(track_coordinates(tcp_world(base.scene[side+'_robot']))[0])>.09
                  or tcp_world(base.scene[side+'_robot'])[2]>.90 for side in ('left','right'))
        value=score(parked,self._unsafe)
        success=bool(value==1 and clear)
        if value==1 and not success:value=.5
        return dict(result,score=value,success=success,unsafe_event=self._unsafe,
                    touched_running=self._unsafe,hands_clear=bool(clear))

    def validate_reset(self,base):
        if self._unsafe or self._error:raise RuntimeError('Invalid warmup')
        for name in CARS:
            _,_,v=self.pose_state(base,name,'chassis')
            if not self._power[name] or np.linalg.norm(v.cpu().numpy()[:3])<.08:
                raise RuntimeError(f'{name} is not already running at scored reset')

    def scripted_actions(self,base,policy,max_steps):
        from .intercept import actions
        yield from actions(self,base,policy,max_steps)

    def close(self):
        self._subscription=None
        if self._contact_file:self._contact_file.close();self._contact_file=None


def make_task(condition):
    if condition not in CONDITIONS:raise ValueError(condition)
    return SlotCarInterception()


def create_assets(directory,condition='recoverable'):
    if condition not in CONDITIONS:raise ValueError(condition)
    assets,_=geometry()
    return {name:create_asset(data,Path(directory)/(name+'.usda'),FOLDER) for name,data in assets.items()}
