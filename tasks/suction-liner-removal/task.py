"""Native suction-liner trainer with pose/contact-based process scoring.

The loaded liner is rigid. Geometric closure retention does not establish a
clinical seal, and the simulator does not model fluid, pressure or infection.
"""
from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np

from resist_isaac.state import pose_state, reset_joints
from resist_isaac.usd_assets import create_asset
from .geometry import BIN_OPEN_ANGLE, BIN_HEIGHT, CANISTER, CANISTER_HEIGHT, CAP_SEAT, CATCH, LID, LINER_DEPTH, PORT, RECEIVER, geometry

CONDITIONS = ('design_preview','recoverable')
FOLDER = Path(__file__).parent


def process_failure(cap_seated, vacuum_off, extracted, previously_isolated, previously_extracted):
    """Classify measured process states; later repairs cannot erase these events."""
    if vacuum_off and not cap_seated:
        return 'vacuum_off_before_closure_seated'
    if not cap_seated and (previously_isolated or previously_extracted):
        return 'closure_lost_after_isolation_or_extraction'
    if extracted and not cap_seated:
        return 'liner_lifted_without_seated_closure'
    if extracted and not vacuum_off:
        return 'liner_lifted_before_vacuum_off'
    return ''


class SuctionLinerPreview:
    robot_contact_offset = .0001

    def __init__(self, scoring_enabled=False):
        self.descriptions = geometry()
        self.scoring_enabled = scoring_enabled
        self.active = False
        self.unsafe_reason = ''
        self.error = None
        self.arm_contact_steps = 0
        self.hardware_clear_s = 0.
        self.seated_s = 0.
        self.was_extracted = False
        self.was_isolated = False
        self.views = {}
        self.contact_views = []
        self.subscription = None

    def set_evidence_directory(self, directory):
        self.events = (Path(directory)/'native-events.jsonl').open('w')

    def build_scene(self, embodiment):
        from isaaclab_arena.assets.object import Object, ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.utils.pose import Pose
        self._temporary = tempfile.TemporaryDirectory(prefix='suction-liner-design-')
        result = []
        for name, data in self.descriptions.items():
            path = create_asset(data, Path(self._temporary.name)/(name+'.usda'), FOLDER)
            if name in ('canister','liner'):
                clear_plastic(path, data)
            obj = Object(name=name, object_type=ObjectType.BASE, usd_path=str(path),
                         initial_pose=Pose(position_xyz=(0,0,0)))
            result.append(obj)
            for part in data['bodies']:
                if not part.get('static', False):
                    result.append(ObjectReference(parent_asset=obj, name=name+'_'+part['name'],
                        prim_path=obj.get_prim_path()+'/'+part['name'], object_type=ObjectType.RIGID))
        return result

    def configure_env(self, cfg):
        from isaaclab.sensors import CameraCfg
        from isaaclab.sim import PinholeCameraCfg
        from yam_arena.ik import matrix_to_quat_xyzw
        if self.scoring_enabled:
            from isaaclab.sensors import ContactSensorCfg
            from yam_arena.robot import asset_link_paths
            links = sorted(asset_link_paths().values())
            hardware = [f'{{ENV_REGEX_NS}}/{name}/{part["name"]}'
                        for name,data in self.descriptions.items() for part in data['bodies']
                        if not part.get('static',False)]
            hardware += [f'{{ENV_REGEX_NS}}/waste_bin/{part["name"]}/shape_{i:03d}'
                         for part in self.descriptions['waste_bin']['bodies'] if part.get('static',False)
                         for i,shape in enumerate(part['shapes']) if shape.get('collision',True)]
            self.contact_definitions = []
            for side,arm,other in [('left','LeftArm','RightArm'),('right','RightArm','LeftArm')]:
                getattr(cfg.scene,side+'_robot').spawn.activate_contact_sensors = True
                filters = hardware+['{ENV_REGEX_NS}/'+other+link for link in links]
                for i,link in enumerate(links):
                    name = f'liner_contacts_{side}_{i}'
                    setattr(cfg.scene,name,ContactSensorCfg(prim_path='{ENV_REGEX_NS}/'+arm+link,
                        update_period=0.,history_length=0,track_contact_points=True,
                        max_contact_data_count_per_prim=1024,filter_prim_paths_expr=filters))
                    self.contact_definitions.append((name,side,filters))
        # Preserve the calibrated top camera; add a wider review view for the raised bin lid.
        cfg.scene.overview_cam = deepcopy(cfg.scene.top_cam)
        cfg.scene.overview_cam.prim_path = '{ENV_REGEX_NS}/OverviewCam'
        eye, target = np.array([-.1665,-.009,2.10]),np.array([.40,0,.90])
        back = eye-target
        back /= np.linalg.norm(back)
        right = np.cross([0,0,1],back)
        right /= np.linalg.norm(right)
        up = np.cross(back,right)
        cfg.scene.overview_cam.offset = CameraCfg.OffsetCfg(pos=tuple(eye),
            rot=tuple(matrix_to_quat_xyzw(np.column_stack((right,up,back)))),convention='opengl')
        eye, target = np.array([.02, .02, 1.58]), np.array([.45, -.03, .96])
        back = eye-target
        back /= np.linalg.norm(back)
        right = np.cross([0,0,1], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        cfg.scene.close_cam = deepcopy(cfg.scene.top_cam)
        cfg.scene.close_cam.prim_path = '{ENV_REGEX_NS}/CloseCam'
        cfg.scene.close_cam.offset = CameraCfg.OffsetCfg(pos=tuple(eye),
            rot=tuple(matrix_to_quat_xyzw(np.column_stack((right,up,back)))), convention='opengl')
        cfg.scene.close_cam.spawn = PinholeCameraCfg(focal_length=28., horizontal_aperture=30.,
                                                    clipping_range=(.02,20.))
        for name in ('left_cam','right_cam'):
            camera = getattr(cfg.scene, name)
            camera.width, camera.height = 320, 180

    def initial_action(self):
        from yam_arena import ik, station
        values = []
        for side, p, rot in [('left',[.22,.26,.96],[0,1,0,0]),
                             ('right',[.18,-.40,.96],[1,0,0,0])]:
            q, pe, re = ik.solve(p, rot, base_pos=station.BASES[side], iters=500)
            if pe > .002 or re > .02:
                raise RuntimeError('Preview ready pose unreachable: '+side)
            values.extend([*q,1.])
        return np.asarray(values)

    def reset(self, base):
        import carb.settings
        from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN
        # Apply after the simulator's rendering preset, which overrides startup settings.
        settings = carb.settings.get_settings()
        desired = {'/rtx/translucency/enabled':True,'/rtx/translucency/maxRefractionBounces':12,
                   '/rtx/raytracing/fractionalCutoutOpacity':True,
                   '/rtx/pathtracing/fractionalCutoutOpacity':True}
        for key,value in desired.items():
            settings.set(key,value)
        self.render_settings = {key:settings.get(key) for key in desired}
        action = self.initial_action()
        for side, start in [('left',0),('right',7)]:
            reset_joints(base, side+'_robot', dict(zip(ARM_JOINTS, action[start:start+6])))
            reset_joints(base, side+'_robot', dict.fromkeys(FINGER_JOINTS, FINGER_OPEN))
        self.base = base
        self.active = False
        self.error = None
        self.unsafe_reason = ''
        self.arm_contact_steps = 0
        self.hardware_clear_s = 0.
        self.seated_s = 0.
        self.was_extracted = False
        self.was_isolated = False
        self.time_s = 0.
        self.views = {(name,part['name']):base.scene[name+'_'+part['name']].root_physx_view
                      for name,data in self.descriptions.items() for part in data['bodies']
                      if not part.get('static',False)}
        if self.scoring_enabled:
            self.contact_views = []
            for name,side,expected in self.contact_definitions:
                view = base.scene.sensors[name].contact_view
                paths = list(view.filter_paths)
                if len(paths)==1 and not isinstance(paths[0],str):
                    paths = list(paths[0])
                if set(paths)!={p.replace('{ENV_REGEX_NS}','/World/envs/env_0') for p in expected}:
                    raise RuntimeError('Incomplete native contact coverage')
                self.contact_views.append((view,side,paths))
            from omni.physx import get_physx_interface
            self.subscription = get_physx_interface().subscribe_physics_on_step_events(
                self._after_step,pre_step=False,order=0)

    def native(self, name, part):
        view = self.views[(name,part)]
        transform = np.array(view.get_transforms().numpy()[0],copy=True)
        velocity = np.array(view.get_velocities().numpy()[0],copy=True)
        if not np.isfinite(np.r_[transform,velocity]).all():
            raise RuntimeError('Non-finite native rigid state')
        return transform[:3],transform[3:],velocity

    def measurements(self):
        from yam_arena.urdf import quat_xyzw_to_matrix
        liner,q,v = self.native('liner','lid_and_load')
        rotation = quat_xyzw_to_matrix(q)
        cap,cq,_ = self.native('liner','closure')
        cap_local = rotation.T@(cap-liner)
        error = cap_local-CAP_SEAT
        cap_up = float(np.dot(quat_xyzw_to_matrix(cq)[:,2],rotation[:,2]))
        connector,_,_ = self.native('connector','elbow')
        shell,sq,_ = self.native('canister','shell')
        _,lq,_ = self.native('waste_bin','lid')
        _,vq,_ = self.native('vacuum','lever')
        cap_seated = np.linalg.norm(error[:2])<.0012 and -.0008<error[2]<.0010 and cap_up>.99
        return dict(cap_seated=bool(cap_seated),cap_error_m=float(np.linalg.norm(error)),
            cap_seat_gap_m=float(error[2]),cap_radial_error_m=float(np.linalg.norm(error[:2])),
            cap_up=cap_up,liner_lift_m=float(liner[2]-shell[2]-CANISTER_HEIGHT),
            switch_angle_rad=float(2*np.arctan2(vq[2],vq[3])),
            bin_lid_angle_rad=float(2*np.arctan2(lq[1],lq[3])),
            connector_removed=bool(np.linalg.norm(rotation.T@(connector-liner)-PORT-[0,0,.008])>.035),
            canister_drift_m=float(np.linalg.norm(shell-CANISTER)),
            liner_speed_m_s=float(np.linalg.norm(v[:3])),
            liner_spin_rad_s=float(np.linalg.norm(v[3:])))

    def begin_episode(self):
        m = self.measurements()
        if m['canister_drift_m']>.01 or abs(m['liner_lift_m'])>.005 or m['connector_removed']:
            raise RuntimeError('Invalid native start: '+repr(m))
        self.active = True

    def _after_step(self, dt):
        import json
        try:
            m = self.measurements()
            arm_contact = hardware_contact = False
            participation = {'left':False,'right':False}
            for view,side,paths in self.contact_views:
                data = view.get_contact_data(dt=1.)
                counts = np.asarray(data[4].numpy()).ravel()
                starts = np.asarray(data[5].numpy()).ravel()
                gaps = np.asarray(data[3].numpy()).ravel()
                if len(counts)!=len(paths) or counts.sum()>=1024:
                    raise RuntimeError('Native contact detector overflow')
                for path,start,count in zip(paths,starts,counts):
                    start,count = int(start),int(count)
                    if start<0 or start+count>len(gaps):
                        raise RuntimeError('Invalid native contact range')
                    contact = count and np.any(gaps[start:start+count]<=1e-6)
                    if contact:
                        if '/LeftArm/' in path or '/RightArm/' in path:
                            arm_contact = True
                        else:
                            hardware_contact = True
                            if '/liner/' in path:
                                participation[side] = True
            if self.active:
                self.time_s += float(dt)
                self.arm_contact_steps += bool(arm_contact)
                self.hardware_clear_s = 0. if hardware_contact else self.hardware_clear_s+float(dt)
                self.seated_s = self.seated_s+float(dt) if m['cap_seated'] else 0.
                extracted = m['liner_lift_m']>.012
                reason = process_failure(m['cap_seated'],m['switch_angle_rad']>.90,extracted,
                                         self.was_isolated,self.was_extracted)
                self.was_extracted |= extracted
                self.was_isolated |= m['switch_angle_rad']>.90
                p,_,_ = self.native('liner','lid_and_load')
                if not (-.15<p[0]<1.05 and abs(p[1])<.65 and p[2]>.90):
                    reason = 'liner_lost_outside_containment'
                if reason and not self.unsafe_reason:
                    self.unsafe_reason = reason
                if hasattr(self,'events'):
                    self.events.write(json.dumps(dict(time_s=self.time_s,**m,
                        arm_contact=bool(arm_contact),hardware_contact=bool(hardware_contact),
                        participation=participation,unsafe_reason=self.unsafe_reason))+'\n')
        except Exception as error:
            self.error = repr(error)

    def evaluate(self, base):
        if self.error:
            raise RuntimeError(self.error)
        if self.scoring_enabled:
            m = self.measurements()
            p,q,v = self.native('liner','lid_and_load')
            from yam_arena.urdf import quat_xyzw_to_matrix
            rotation = quat_xyzw_to_matrix(q)
            shell,sq,_ = self.native('canister','shell')
            shell_rotation = quat_xyzw_to_matrix(sq)
            supported = (np.linalg.norm((shell_rotation.T@(p-shell))[:2])<.006
                         and abs(m['liner_lift_m'])<.006
                         and np.dot(rotation[:,2],shell_rotation[:,2])>.97)
            canister_retained = (m['canister_drift_m']<.010 and shell_rotation[2,2]>.97)
            corners = np.array([[x,y,z] for x in (-.08,.08) for y in (-.108,.108)
                                for z in (-LINER_DEPTH,.036)])@rotation.T+p
            local = corners-RECEIVER
            inside = bool(np.all(np.abs(local[:,0])<.107) and np.all(np.abs(local[:,1])<.117)
                          and local[:,2].min()>-.002 and local[:,2].max()<BIN_HEIGHT)
            # Include every passive tether capsule, rather than only the filled body.
            for part in self.descriptions['liner']['bodies']:
                if not part['name'].startswith('tether'):
                    continue
                tp,tq,_ = self.native('liner',part['name'])
                shape = part['shapes'][0]
                endpoints = np.asarray(shape['fromto']).reshape(2,3)
                points = endpoints@quat_xyzw_to_matrix(tq).T+tp-RECEIVER
                radius = shape['size'][0]
                inside &= bool(np.all(np.abs(points[:,0])+radius<.107)
                               and np.all(np.abs(points[:,1])+radius<.117)
                               and points[:,2].min()-radius>-.002
                               and points[:,2].max()+radius<BIN_HEIGHT)
            connector,_,_ = self.native('connector','elbow')
            parked = bool(abs(connector[0]-CATCH[0])<.070 and abs(connector[1]-CATCH[1])<.038
                          and abs(connector[2]-(CATCH[2]+.020))<.040)
            clear = self.hardware_clear_s>=1.
            still = np.linalg.norm(v[:3])<.005 and np.linalg.norm(v[3:])<.05
            score = 0.
            if not self.unsafe_reason and canister_retained and m['cap_seated'] and m['switch_angle_rad']>.90:
                if supported and m['connector_removed']:
                    score = .5
                if inside and clear and still and parked:
                    score = 1. if abs(m['bin_lid_angle_rad'])<.0524 else .75
            return dict(success=score==1.,score=score,scoring_enabled=True,**m,
                        liner_inside=inside,connector_parked=parked,hands_clear=clear,
                        liner_supported=bool(supported),canister_retained=bool(canister_retained),
                        unsafe_event=bool(self.unsafe_reason),arm_contact_substeps=self.arm_contact_steps)
        shell, _, shell_v = pose_state(base.scene['canister_shell'])
        liner, _, liner_v = pose_state(base.scene['liner_lid_and_load'])
        connector, _, connector_v = pose_state(base.scene['connector_elbow'])
        _, lid_q, lid_v = pose_state(base.scene['waste_bin_lid'])
        lid_angle = 2*np.arctan2(float(lid_q[1]),float(lid_q[3]))
        shell, liner, connector = (p.cpu().numpy() for p in (shell, liner, connector))
        return dict(success=False, score=0., scoring_enabled=False,
                    canister_drift_m=float(np.linalg.norm(shell-CANISTER)),
                    liner_drift_m=float(np.linalg.norm(liner-LID)),
                    connector_drift_m=float(np.linalg.norm(connector-(LID+PORT+[0,0,.012]))),
                    bin_lid_angle_rad=float(lid_angle),
                    bin_lid_angle_error_rad=float(abs(lid_angle-BIN_OPEN_ANGLE)),
                    max_linear_speed_m_s=max(float(np.linalg.norm(v[:3].cpu().numpy()))
                                             for v in (shell_v, liner_v, connector_v, lid_v)),
                    max_angular_speed_rad_s=max(float(np.linalg.norm(v[3:].cpu().numpy()))
                                              for v in (shell_v, liner_v, connector_v, lid_v)))

    def close(self):
        self.active = False
        self.subscription = None
        if hasattr(self,'events'):
            self.events.flush()
            self.events.close()


def make_task(condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    return SuctionLinerPreview(scoring_enabled=condition=='recoverable')


def clear_plastic(path, data):
    """Author double-sided residue and compliant rubber contact materials."""
    from pxr import PhysxSchema, Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    root = '/'+data['name']
    for part in data['bodies']:
        for i,shape in enumerate(part['shapes']):
            if shape.get('compliant_fit'):
                prim = stage.GetPrimAtPath(f"{root}/Materials/{part['name']}_{i:03d}")
                material = PhysxSchema.PhysxMaterialAPI.Apply(prim)
                # ponytail: lumped contact spring; calibrate from bench forces when available.
                material.CreateCompliantContactStiffnessAttr(250.0)
                material.CreateCompliantContactDampingAttr(.3)
                material.CreateCompliantContactAccelerationSpringAttr(False)
            if shape['type'] == 'mesh' and not shape.get('collision',True):
                prim = stage.GetPrimAtPath(root+'/'+part['name']+f'/shape_{i:03d}')
                UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
    stage.GetRootLayer().Save()
