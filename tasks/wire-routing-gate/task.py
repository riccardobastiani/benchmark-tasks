"""Route a connected wire span between two metal posts and seat its final clip."""
import json
from pathlib import Path
import tempfile

import numpy as np
from resist_isaac.state import pose_state, reset_joints
from resist_isaac.usd_assets import create_asset
from .geometry import (geometry, wire_nodes, UPSTREAM, TARGET, WIRE_RADIUS, PLATE_Z,
                       GATE_X, GATE_LOW, GATE_HIGH, GATE_WIRE_HEIGHT)
from .rules import crossings, clip_state, gate_route, score

CONDITIONS = ('recoverable',)
FOLDER = Path(__file__).parent


class WireRoutingGate:
    physics_hz = 240
    robot_contact_offset = .0001
    hold_s = 2.

    def __init__(self):
        self.folder = FOLDER
        self.cfg = json.loads((FOLDER/'pilot.json').read_text())
        self.descriptions = geometry()
        self._warming_up = True
        self._unsafe = False
        self._error = None
        self._failure_reason = ''
        self._coordinated = False
        self._both_contact_time = 0.
        self._subscription = None
        self._native_views = {}
        self.contacts = {'left': set(), 'right': set()}
        self._motion_substeps = {'arms': 0, 'posts': 0}
        self._motion_contacts = []

    def build_scene(self, embodiment):
        from isaaclab_arena.assets.object import Object, ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.utils.pose import Pose
        if hasattr(self, 'evidence_dir'):
            directory = self.evidence_dir/'assets'
        else:
            self._temporary = tempfile.TemporaryDirectory(prefix='wire-routing-gate-')
            directory = Path(self._temporary.name)
        objects = []
        for name, data in self.descriptions.items():
            path = create_asset(data, directory/(name+'.usda'), FOLDER)
            obj = Object(name=name, object_type=ObjectType.BASE, usd_path=str(path),
                         initial_pose=Pose(position_xyz=(0., 0., 0.)))
            objects.append(obj)
            for part in data['bodies']:
                if not part.get('static', False):
                    objects.append(ObjectReference(parent_asset=obj, name=name+'_'+part['name'],
                        prim_path=obj.get_prim_path()+'/'+part['name'], object_type=ObjectType.RIGID))
        return objects

    def set_evidence_directory(self, folder):
        self.evidence_dir = Path(folder)
        self._events = (self.evidence_dir/'native-wire-events.jsonl').open('w')

    def configure_native(self, cfg):
        from copy import deepcopy
        from isaaclab.sensors import CameraCfg, ContactSensorCfg
        from isaaclab.sim import PinholeCameraCfg
        from yam_arena.ik import matrix_to_quat_xyzw
        eye = np.array([.69, -.48, 1.15]); back = eye-np.array([.285, .080, .78])
        back /= np.linalg.norm(back)
        right = np.cross([0., 0., 1.], back); right /= np.linalg.norm(right)
        up = np.cross(back, right)
        cfg.scene.close_cam = deepcopy(cfg.scene.top_cam)
        cfg.scene.close_cam.prim_path = '{ENV_REGEX_NS}/CloseCam'
        cfg.scene.close_cam.offset = CameraCfg.OffsetCfg(pos=tuple(eye),
            rot=tuple(matrix_to_quat_xyzw(np.column_stack((right, up, back)))), convention='opengl')
        cfg.scene.close_cam.spawn = PinholeCameraCfg(focal_length=32., horizontal_aperture=30.,
                                                    clipping_range=(.02, 20.))
        # Keep unrecorded wrist views at the station's normal render resolution.
        # The top and close cameras retain the requested native video size.
        for name in ('left_cam','right_cam'):
            camera=getattr(cfg.scene,name)
            camera.width=min(camera.width,640); camera.height=min(camera.height,360)
        from yam_arena.robot import asset_link_paths
        self._contact_definitions = []
        wires = ['{ENV_REGEX_NS}/wire/'+p['name'] for p in self.descriptions['wire']['bodies']]
        links = sorted(asset_link_paths().values())
        posts = [f'{{ENV_REGEX_NS}}/fixture/plate/shape_{i:03d}'
                 for i, shape in enumerate(self.descriptions['fixture']['bodies'][0]['shapes'])
                 if shape['type']=='cylinder']
        self._expected_filters = {}
        for side, arm in [('left', 'LeftArm'), ('right', 'RightArm')]:
            getattr(cfg.scene, side+'_robot').spawn.activate_contact_sensors = True
            other = 'RightArm' if side=='left' else 'LeftArm'
            filters = wires+posts+['{ENV_REGEX_NS}/'+other+p for p in links]
            self._expected_filters[side] = {p.replace('{ENV_REGEX_NS}', '/World/envs/env_0') for p in filters}
            for index, path in enumerate(links):
                name = f'wire_contacts_{side}_{index}'
                setattr(cfg.scene, name, ContactSensorCfg(prim_path='{ENV_REGEX_NS}/'+arm+path,
                    update_period=0., history_length=0, track_contact_points=True,
                    max_contact_data_count_per_prim=2048, filter_prim_paths_expr=filters))
                self._contact_definitions.append((name, side, path.rsplit('/', 1)[-1]))

    def initial_action(self):
        from yam_arena import ik, station
        action = []
        for side, y, rotation in [('left', .26, [0, 1, 0, 0]), ('right', -.26, [1, 0, 0, 0])]:
            q, pe, re = ik.solve([.22, y, .96], rotation, base_pos=station.BASES[side], iters=500)
            if pe>.002 or re>.02: raise ValueError(f'Initial {side} IK error: {pe}, {re}')
            action.extend([*q, 1.])
        return np.asarray(action)

    def reset(self, base):
        from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN
        action = self.initial_action()
        for side, start in [('left', 0), ('right', 7)]:
            reset_joints(base, side+'_robot', dict(zip(ARM_JOINTS, action[start:start+6])))
            reset_joints(base, side+'_robot', dict.fromkeys(FINGER_JOINTS, FINGER_OPEN))
        self._warming_up = True; self._unsafe = False; self._error = None
        self._failure_reason = ''; self._both_contact_time = 0.; self._coordinated = False
        self._motion_substeps = {'arms': 0, 'posts': 0}
        self._motion_contacts = []
        self._native_views = {part['name']:base.scene['wire_'+part['name']].root_physx_view
                              for part in self.descriptions['wire']['bodies']}
        self._contact_views = []
        for name, side, link in self._contact_definitions:
            view = base.scene.sensors[name].contact_view
            if view is None: raise RuntimeError('Missing native robot/wire contact view')
            raw = list(view.filter_paths)
            filters = list(raw[0]) if len(raw)==1 and not isinstance(raw[0], str) else raw
            if set(filters)!=self._expected_filters[side]:
                raise RuntimeError('Native filters must cover every wire link, post and opposite-arm link')
            self._contact_views.append((view, side, link, filters))
        from omni.physx import get_physx_interface
        self._subscription = get_physx_interface().subscribe_physics_on_step_events(
            self._after_step, pre_step=False, order=0)

    def begin_episode(self, base):
        if self._error: raise RuntimeError(self._error)
        self.validate_reset(base)
        self._warming_up = False

    def _after_step(self, dt):
        try:
            state = self._wire_state(None)
            contacts = {'left': set(), 'right': set()}; any_contact = False
            motion_contacts = []
            for view, side, link, filters in self._contact_views:
                data = view.get_contact_data(dt=1.)
                counts = np.array(data[4].numpy(), copy=True).ravel()
                starts = np.array(data[5].numpy(), copy=True).ravel()
                gaps = np.array(data[3].numpy(), copy=True).ravel()
                if len(counts)!=len(filters) or counts.sum()>=2048:
                    raise RuntimeError('Invalid or overflowing native wire contacts')
                for path, start, count in zip(filters, starts, counts):
                    start, count = int(start), int(count)
                    if start<0 or start+count>len(gaps): raise RuntimeError('Invalid wire contact range')
                    if count and np.any(gaps[start:start+count]<=1e-6):
                        if '/wire/' in path:
                            any_contact = True
                            if link in ('tip_left', 'tip_right'):
                                contacts[side].add(int(path.rsplit('link', 1)[-1]))
                        else:
                            kind = 'posts' if '/fixture/' in path else 'arms'
                            motion_contacts.append((kind, side, link, path, float(gaps[start:start+count].min())))
            self.contacts = contacts; self._any_robot_contact = any_contact
            self._motion_contacts = motion_contacts
            if not self._warming_up:
                for kind in self._motion_substeps:
                    self._motion_substeps[kind] += any(c[0]==kind for c in motion_contacts)
                if not clip_state(state, UPSTREAM)[1]:
                    self._unsafe = True; self._failure_reason = 'upstream_wire_released'
                if max(np.linalg.norm(a[3]-b[2]) for a,b in zip(state,state[1:]))>.003:
                    self._unsafe = True; self._failure_reason = 'wire_joint_separated'
                over_clip = any(abs(p[0]-TARGET[0])<.015 and TARGET[2]<p[2]<TARGET[2]+.08
                                for _,p in crossings(state,1,TARGET[1]))
                both = bool(contacts['left'] and contacts['right'] and over_clip)
                self._both_contact_time = self._both_contact_time+dt if both else 0.
                if self._both_contact_time>=.2: self._coordinated = True
            if hasattr(self, '_events'):
                self._events.write(json.dumps(dict(warmup=self._warming_up, unsafe=self._unsafe,
                    failure_reason=self._failure_reason,
                    motion_contacts=motion_contacts,
                    upstream_crossings=[(i,p.tolist()) for i,p in crossings(state,1,UPSTREAM[1])],
                    contacts={s:sorted(v) for s,v in contacts.items()},
                    wire=[p.tolist() for p,_,_,_ in state]))+'\n')
        except Exception as error:
            self._error = repr(error)

    def pose_state(self, base, name, part=None):
        return pose_state(base.scene[name+'_'+part] if part else base.scene[name])

    def _wire_state(self, base):
        from yam_arena.urdf import quat_xyzw_to_matrix
        state = []
        for definition in self.descriptions['wire']['bodies']:
            if self._native_views:
                view = self._native_views[definition['name']]
                transform = np.array(view.get_transforms().numpy()[0], copy=True)
                p, q = transform[:3], transform[3:]
                v = np.array(view.get_velocities().numpy()[0], copy=True)
            else:
                p, q, v = [value.cpu().numpy() for value in self.pose_state(base, 'wire', definition['name'])]
            if not np.isfinite(np.r_[p, q, v]).all() or abs(np.dot(q, q)-1)>.001:
                raise RuntimeError('Invalid native wire measurement')
            half = definition['shapes'][0]['size'][1]
            tangent = quat_xyzw_to_matrix(q)[:, 2]
            state.append((p, v, p-half*tangent, p+half*tangent))
        return state

    def evaluate(self, base):
        if self._error: raise RuntimeError(self._error)
        state = self._wire_state(base)
        gaps = [np.linalg.norm(a[3]-b[2]) for a, b in zip(state, state[1:])]
        upstream_seated, upstream_retained = clip_state(state, UPSTREAM)
        target_seated, _ = clip_state(state, TARGET)
        aligned = any(abs(p[0]-TARGET[0])<.008 and TARGET[2]+.004<p[2]<TARGET[2]+.05
                      for _,p in crossings(state,1,TARGET[1]))
        routed, gate_hits = gate_route(state, UPSTREAM, TARGET, GATE_X, GATE_LOW,
                                      GATE_HIGH, PLATE_Z, WIRE_RADIUS, GATE_WIRE_HEIGHT)
        from yam_arena.runtime import tcp_world
        clear = all(min(np.linalg.norm(tcp_world(base.scene[s+'_robot'])-p) for p,_,_,_ in state)>.07
                    for s in ('left','right'))
        speed = max(np.linalg.norm(v[:3]) for _,v,_,_ in state)
        released = clear and not getattr(self,'_any_robot_contact',True) and speed<.03
        value = score(self._unsafe,routed and aligned and self._coordinated,
                      upstream_seated and target_seated,released,self._coordinated)
        return dict(success=value==1., score=value, upstream_retained=upstream_retained,
            arm_contact_substeps=self._motion_substeps['arms'],
            post_contact_substeps=self._motion_substeps['posts'],
            upstream_seated=upstream_seated, target_occupied=target_seated, routed=routed,
            coordinated_grasp=self._coordinated, hands_clear=clear, gate_crossing_count=len(gate_hits),
            gate_crossing_max_height_m=float(max((p[2]-PLATE_Z for p in gate_hits),default=0.)),
            gate_crossing_min_y_m=float(min((p[1] for p in gate_hits),default=0.)),
            gate_crossing_max_y_m=float(max((p[1] for p in gate_hits),default=0.)),
            left_contact_count=len(self.contacts['left']),right_contact_count=len(self.contacts['right']),
            wire_gap_max_m=float(max(gaps)), wire_speed_max_m_s=float(speed),
            wire_min_centre_z_m=float(min(p[2] for p, _, _, _ in state)),
            unsafe_event=self._unsafe)

    def validate_reset(self, base):
        result = self.evaluate(base)
        if not result['upstream_retained'] or result['target_occupied']:
            raise RuntimeError(f'Invalid clip occupancy at reset: {result}')
        if result['wire_gap_max_m']>.002 or result['wire_min_centre_z_m']<PLATE_Z+WIRE_RADIUS-.001:
            raise RuntimeError(f'Invalid wire mechanics at reset: {result}')

    def scripted_actions(self, base, policy, max_steps):
        from .interaction import actions
        for action, phase, errors in actions(self,base,policy,max_steps):
            if policy=='feed_and_clip' and any(self._motion_substeps.values()):
                raise RuntimeError(f'Careful demo stopped after robot contact: {self._motion_contacts}')
            yield action, phase, errors


def make_task(condition):
    if condition not in CONDITIONS: raise ValueError(condition)
    return WireRoutingGate()


def create_assets(directory):
    directory = Path(directory)
    return {name: create_asset(data, directory/(name+'.usda'), FOLDER)
            for name, data in geometry().items()}
