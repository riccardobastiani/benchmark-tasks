"""Install, secure and verify a light-curtain frame on a guarded press trainer."""
import json
from pathlib import Path
import tempfile
import numpy as np
from resist_isaac.state import pose_state,reset_joints
from resist_isaac.usd_assets import create_asset
from .geometry import geometry,Z,FRAME,RX,PIN,HANDLE,RX_HANDLE,BOLT,BOLT_GRIP,ROD,ROD_GRIP,ROD_TIP,RAM,BUTTONS,BEAMS
from .rules import Proof,Controller,opening_covered,shaft_point_speed

CONDITIONS=('recoverable',)
FOLDER=Path(__file__).parent


class LightCurtain:
    physics_hz=240
    robot_contact_offset=.0001
    hold_s=1.

    def __init__(self):
        self.folder=FOLDER;self.cfg=json.loads((FOLDER/'pilot.json').read_text())
        self.descriptions,self.visuals=geometry();self._subscription=None;self._error=None;self._warming_up=True
        self._views={};self.proof=Proof();self.control=Controller();self._arm_contact_substeps=0;self._stable=0.;self._clear=0.
        self._colors={};self._attrs={};self._failure_reason='';self._robot_contacts={}

    def set_evidence_directory(self,path):
        self.evidence_dir=Path(path);self._events=(self.evidence_dir/'native-light-curtain-events.jsonl').open('w')

    def build_scene(self,embodiment):
        from isaaclab_arena.assets.object import Object,ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.utils.pose import Pose
        if hasattr(self,'evidence_dir'):directory=self.evidence_dir/'assets'
        else:
            self._temporary=tempfile.TemporaryDirectory(prefix='light-curtain-');directory=Path(self._temporary.name)
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
        cfg.sim.enable_scene_query_support=True
        from copy import deepcopy
        from isaaclab.sensors import CameraCfg,ContactSensorCfg
        from isaaclab.sim import PinholeCameraCfg
        from yam_arena.ik import matrix_to_quat_xyzw
        from yam_arena.robot import asset_link_paths
        eye=np.array([-.08,-.24,1.62]);back=eye-np.array([.40,0.,.94]);back/=np.linalg.norm(back)
        right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
        cfg.scene.close_cam=deepcopy(cfg.scene.top_cam);cfg.scene.close_cam.prim_path='{ENV_REGEX_NS}/CloseCam'
        cfg.scene.close_cam.offset=CameraCfg.OffsetCfg(pos=tuple(eye),
            rot=tuple(matrix_to_quat_xyzw(np.column_stack((right,up,back)))),convention='opengl')
        cfg.scene.close_cam.spawn=PinholeCameraCfg(focal_length=27.,horizontal_aperture=30.,clipping_range=(.02,20.))
        for name in ('left_cam','right_cam'):
            camera=getattr(cfg.scene,name);camera.width=min(camera.width,640);camera.height=min(camera.height,360)
        links=sorted(asset_link_paths().values());self._contact_defs=[];self._expected={}
        self._objects={(name,part['name']) for name,data in self.descriptions.items() for part in data['bodies'] if not part.get('static',False)}
        hardware=[f'{{ENV_REGEX_NS}}/{name}/{part}' for name,part in sorted(self._objects)]
        for side,arm,other in [('left','LeftArm','RightArm'),('right','RightArm','LeftArm')]:
            getattr(cfg.scene,side+'_robot').spawn.activate_contact_sensors=True
            filters=hardware+['{ENV_REGEX_NS}/'+other+p for p in links]
            self._expected[side]={p.replace('{ENV_REGEX_NS}','/World/envs/env_0') for p in filters}
            for i,path in enumerate(links):
                name=f'curtain_contact_{side}_{i}'
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
        self.control=Controller();self.proof=Proof();self._stable=0.;self._clear=0.;self._arm_contact_substeps=0;self._views={}
        self._last_probe=None;self._geometry_ref=None;self._failure_reason=''
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
        from omni.physx import get_physx_interface,get_physx_scene_query_interface
        self._query=get_physx_scene_query_interface()
        self._drive=None
        self._subscription=get_physx_interface().subscribe_physics_on_step_events(self._after_step,pre_step=False,order=0)

    def begin_episode(self,base):
        self.validate_reset(base);self._warming_up=False

    def native(self,name,part=None):
        if part is None:part={'rod':'tool','press':'ram',**{n:'button' for n in BUTTONS}}[name]
        view=self._views[(name,part)];t=np.array(view.get_transforms().numpy()[0],copy=True)
        v=np.array(view.get_velocities().numpy()[0],copy=True)
        if not np.isfinite(np.r_[t,v]).all() or abs(np.dot(t[3:],t[3:])-1)>.001:raise RuntimeError('Invalid native pose/velocity')
        return t[:3],t[3:],v

    def pose_state(self,base,name,part=None):return pose_state(base.scene[name+'_'+part] if part else base.scene[name])

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

    def measurements(self):
        from yam_arena.urdf import quat_xyzw_to_matrix
        f,fq,fv=self.native('mount','frame');r,rq,rv=self.native('mount','receiver')
        bolt=self.native('mount','height_bolt')[0];pin=self.native('mount','index_pin')[0]
        yaw=float(np.arctan2(quat_xyzw_to_matrix(rq)[1,0],quat_xyzw_to_matrix(rq)[0,0]))
        height=float(f[2]-FRAME[2]);pin_lift=float(pin[2]-(FRAME[2]+height+PIN[2]))
        return dict(height=height,yaw=yaw,bolt=float(bolt[0]-BOLT[0]),pin=pin_lift,
            aligned=bool(abs(yaw)<np.deg2rad(1.0)),covered=opening_covered(BEAMS+height),
            locked=bool(bolt[0]-BOLT[0]>-.008 and pin_lift<.004),mount_speed=float(np.linalg.norm(fv[:3])),
            ram_q=float(self.native('press')[0][2]-RAM[2]),ram_v=float(self.native('press')[2][2]))

    def optics(self,m):
        blocked=[];paths=[]
        for h in BEAMS:
            hit=self._query.raycast_closest((float(FRAME[0]),.132,float(Z+m['height']+h)),(0.,-1.,0.),.264)
            flag=bool(hit.get('hit',False));blocked.append(flag);paths.append(str(hit.get('collision','')) if flag else '')
        return blocked,paths

    def _after_step(self,dt):
        try:
            probe_cross=None;probe_axis=None;probe_root=None;probe_speed=None;probe_solver_speed=None;probe_eligible=False
            self._robot_contacts={str(n)+'/'+str(p):False for n,p in self._objects};arms=False
            for view,filters in self._robot_views:
                for path in self.native_contacts(view,filters):
                    if '/LeftArm/' in path or '/RightArm/' in path:arms=True
                    else:self._robot_contacts['/'.join(path.split('/')[-2:])]=True
            m=self.measurements();blocked,paths=self.optics(m);self._blocked=blocked;self._paths=paths;self._m=m
            buttons={n:bool(self.native(n)[0][2]-p[2]<-.004) for n,p in BUTTONS.items()}
            self.control.step(dt,m['aligned'],any(blocked),buttons,m['ram_q'],m['ram_v'],self._warming_up)
            if self._drive is None:
                import omni.usd
                self._drive=omni.usd.get_context().get_stage().GetPrimAtPath('/World/envs/env_0/press/stroke').GetAttribute('drive:linear:physics:targetPosition')
                if not self._drive:raise RuntimeError('Native press drive absent')
            self._drive.Set(float(self.control.target))
            if not self._warming_up:
                self._arm_contact_substeps+=arms
                mount_clear=not any(v for k,v in self._robot_contacts.items() if k.startswith('mount/'))
                valid=m['covered'] and m['aligned'] and m['locked'] and m['mount_speed']<.005
                self._stable=self._stable+dt if valid and mount_clear else 0.
                current=np.array([m['height'],m['yaw']])
                if self._geometry_ref is None:self._geometry_ref=current
                if abs(current[0]-self._geometry_ref[0])>.001 or abs(current[1]-self._geometry_ref[1])>np.deg2rad(.25) or not m['locked']:
                    self.proof.reset();self.control.inhibit=False;self.control.stop=False;self.control.reset_test=False;self._geometry_ref=current
                from yam_arena.urdf import quat_xyzw_to_matrix
                p,q,v=self.native('rod');p=p.astype(float);rotation=quat_xyzw_to_matrix(q).astype(float);axis=rotation[:,0]
                probe_root=p.tolist();probe_axis=axis.tolist()
                if abs(axis[0])>.5:
                    distance=(FRAME[0]-p[0])/axis[0];cross=p+axis*distance;tip=p+ROD_TIP*axis
                    speed=shaft_point_speed(self._last_probe,p,axis,distance,dt) if self._last_probe is not None else None
                    com=p+rotation@np.asarray(self.descriptions['rod']['bodies'][0]['center_of_mass'])
                    probe_solver_speed=float(np.linalg.norm(v[:3]+np.cross(v[3:],cross-com)))
                    rod_only=bool(any('/rod/tool/' in path for path in paths) and all(not path or '/rod/tool/' in path for path in paths))
                    shaft_only=bool(all(not path or path.endswith('/rod/tool/shape_000') for path in paths))
                    eligible=(speed is not None and self._stable>.2 and m['aligned'] and rod_only and shaft_only and axis[0]>np.cos(np.deg2rad(10.)) and tip[0]>FRAME[0]+.03)
                    probe_cross=cross.tolist();probe_axis=axis.tolist();probe_speed=speed;probe_eligible=bool(eligible)
                    if speed is not None:self.proof.observe(dt,float(cross[1]),float(cross[2]-Z),speed,eligible)
                self._last_probe=(p.copy(),axis.copy())
                self._clear=self._clear+dt if not any(self._robot_contacts.values()) else 0.
                if self.control.failure:self._failure_reason=self.control.failure
            if hasattr(self,'_events'):
                self._events.write(json.dumps(dict(warmup=self._warming_up,**m,blocked=blocked,blockers=paths,
                    buttons=buttons,mode=self.control.mode,state=self.control.state,ram_target=self.control.target,
                    inhibit_proved=self.control.inhibit,stop_proved=self.control.stop,reset_proved=self.control.reset_test,
                    cycles=self.control.cycles,proof_fraction=self.proof.fraction,arm_contact=bool(arms),robot_contacts=self._robot_contacts,
                    probe_cross=probe_cross,probe_axis=probe_axis,probe_root=probe_root,probe_speed=probe_speed,
                    probe_solver_speed=probe_solver_speed,probe_eligible=probe_eligible))+'\n')
        except Exception as error:self._error=repr(error)

    def update_visuals(self):
        from pxr import Gf
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        desired={'alignment':(.03,.85,.15) if self._m['aligned'] else (.9,.03,.02),
            'permit':(.05,.85,.15) if self.control.state in ('ARMED','DOWN','UP') else (.03,.07,.04),
            'blocked':(.95,.04,.02) if any(self._blocked) else (.10,.035,.025),
            'reset_required':(1.,.55,.015) if self.control.state=='RESET_REQUIRED' else (.04,.035,.015)}
        for i,flag in enumerate(self._blocked):desired['beam'+str(i)]=(.85,.05,.02) if flag else (.015,.52,.68)
        for name,color in desired.items():
            if self._colors.get(name)==color:continue
            if name not in self._attrs:
                obj,part,index=self.visuals[name];attr=stage.GetPrimAtPath(f'/World/envs/env_0/{obj}/Materials/{part}_{index:03d}/Surface').GetAttribute('inputs:diffuseColor')
                if not attr:raise RuntimeError('Indicator material absent: '+name)
                self._attrs[name]=attr
            self._attrs[name].Set(Gf.Vec3f(*color));self._colors[name]=color

    def evaluate(self,base):
        if self._error:raise RuntimeError(self._error)
        m=self.measurements();self.update_visuals()
        p,q,v=self.native('rod');parked=bool(np.linalg.norm(p[:2]-ROD[:2])<.018 and abs(p[2]-(Z+.038))<.012 and np.linalg.norm(v[:3])<.02)
        from yam_arena.runtime import tcp_world
        clear=self._clear>.15 and all(np.linalg.norm(tcp_world(base.scene[s+'_robot'])-np.array([.35,0,Z+.18]))>.18 for s in ('left','right'))
        installed=m['covered'] and m['aligned'] and m['locked'] and self._stable>1.
        functional=self.control.inhibit and self.control.stop and self.control.reset_test
        value=.25 if installed else 0.
        if installed and self.proof.complete and functional:value=.5
        if value==.5 and self.control.cycles>0 and parked and clear and m['ram_q']>-.006:value=1.
        if self._failure_reason:value=0.
        return dict(success=value==1.,score=value,unsafe_event=bool(self._failure_reason),height_m=m['height'],yaw_deg=float(np.rad2deg(m['yaw'])),
            height_lock_m=m['bolt'],alignment_lock_m=m['pin'],aligned=m['aligned'],opening_covered=m['covered'],locked=m['locked'],
            proof_fraction=self.proof.fraction,inhibit_proved=self.control.inhibit,stop_proved=self.control.stop,reset_proved=self.control.reset_test,
            cycles=self.control.cycles,ram_q_m=m['ram_q'],ram_v_m_s=m['ram_v'],rod_parked=parked,hands_clear=bool(clear),
            arm_contact_substeps=self._arm_contact_substeps,blocked_beams=sum(self._blocked))

    def validate_reset(self,base):
        m=self.evaluate(base)
        hit=self._query.raycast_closest((float(FRAME[0]),.132,float(Z+.025)),(0.,-1.,0.),.264)
        if not hit.get('hit',False) or '/fixture/' not in str(hit.get('collision','')):
            raise RuntimeError('Native scene query did not detect the fixed lower guard')
        if not (-.003<m['height_m']<.005 and not m['aligned'] and not m['opening_covered'] and m['rod_parked'] and not self.control.mode):
            raise RuntimeError('Invalid light-curtain reset: '+repr(m))

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
    return LightCurtain()


def create_assets(directory):return {name:create_asset(data,Path(directory)/(name+'.usda'),FOLDER) for name,data in geometry()[0].items()}
