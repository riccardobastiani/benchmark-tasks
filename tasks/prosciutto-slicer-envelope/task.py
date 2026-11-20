"""Scripted food-slicing and paper-packing scene; cut separation is an event proxy."""

from copy import deepcopy
import math
from pathlib import Path
import tempfile
import numpy as np

from resist_isaac.recovery import create_asset
from resist_isaac.state import reset_joints
from .geometry import (BLADE_CENTER, BLADE_GUARD_RADIUS, BLADE_HALF_THICKNESS, BLADE_RADIUS,
                       GUARD_ROD_RADIUS, GUARD_NORMAL_OFFSET, HAM_COLLIDER_CENTER,
                       CARRIAGE_START, CARRIAGE_TRAVEL_LIMIT, HAM_HALF_SIZE, PORTION_HALF_SIZE,
                       POWER_OFF, POWER_ON, SWITCH_ACTUATION, geometry)
from .scoring import ArmBounds, blade_zone_intersects, slice_inside_envelope

CONDITIONS = ('recoverable',)
FOLDER = Path(__file__).parent


class ProsciuttoTask:
    renderer = 'PathTracing'
    preview = False
    render_samples = 4
    reset_settle_s = .6
    hold_s = 1 / 240  # The task already requires a one-second stable endpoint.
    scene_id = 'prosciutto-slicer-packaging-prototype'
    instruction = 'Slice one portion of prosciutto with the guarded slicer, place it in the food envelope, and fold the envelope closed.'

    def __init__(self):
        self.descriptions = geometry()
        self.unsafe_reason = ''
        self.active = False
        self._monitor = None
        self.portion_cut = False
        self.slicer_stopped = False
        self.portion_placed = False
        self.flap_closed = False
        self.time_s = 0.

    def build_scene(self, embodiment):
        from isaaclab_arena.assets.object import Object, ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.utils.pose import Pose

        self._temporary = tempfile.TemporaryDirectory(prefix='prosciutto-slicer-')
        result = []
        for name, data in self.descriptions.items():
            path = create_asset(data, Path(self._temporary.name)/(name+'.usda'), FOLDER)
            obj = Object(name=name, object_type=ObjectType.BASE, usd_path=str(path),
                         initial_pose=Pose(position_xyz=(0., 0., 0.)))
            result.append(obj)
            for part in data['bodies']:
                if not part.get('static', False):
                    result.append(ObjectReference(parent_asset=obj, name=name+'_'+part['name'],
                        prim_path=obj.get_prim_path()+'/'+part['name'], object_type=ObjectType.RIGID))
        return result

    def configure_env(self, cfg):
        from isaaclab.sensors import CameraCfg
        from isaaclab.sim import PinholeCameraCfg
        from isaaclab_physx.physics import PhysxCfg
        from yam_arena.ik import matrix_to_quat_xyzw
        cfg.sim.use_fabric = False
        cfg.sim.physics = PhysxCfg(enable_ccd=True)
        cfg.sim.render_interval = round(240/getattr(self, 'render_hz', 10))
        cfg.scene.filter_collisions = False
        if getattr(self, 'physics_only', False):
            return
        if getattr(self, 'preview', False):
            cfg.scene.left_cam = None
            cfg.scene.right_cam = None
            cfg.observations.camera_obs = None
        eye = np.asarray([.68, .55, 1.27])
        target = np.asarray([.355, .015, .825])
        back = eye-target
        back /= np.linalg.norm(back)
        right = np.cross([0., 0., 1.], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        close = deepcopy(cfg.scene.top_cam)
        close.prim_path = '{ENV_REGEX_NS}/close_cam'
        close.offset = CameraCfg.OffsetCfg(pos=tuple(eye),
            rot=tuple(matrix_to_quat_xyzw(np.column_stack((right, up, back)))), convention='opengl')
        close.spawn = PinholeCameraCfg(focal_length=28., horizontal_aperture=30., clipping_range=(.02, 20.))
        close.update_period = 1/getattr(self, 'render_hz', 10)
        cfg.scene.close_cam = close
        cfg.scene.top_cam.update_period = 1/getattr(self, 'render_hz', 10)

    def initial_action(self):
        from yam_arena.constants import READY
        return np.asarray([*READY, 1., *READY, 1.])

    def reset(self, base):
        import carb.settings
        import omni.usd
        from pxr import UsdPhysics
        self.base = base
        self._monitor = None
        self.stage = omni.usd.get_context().get_stage()
        self.cut_joint = UsdPhysics.FixedJoint.Get(self.stage, '/World/envs/env_0/workpiece/cut_link')
        if not self.cut_joint:
            raise RuntimeError('Cut proxy joint is missing from the native stage')
        self.cut_joint.CreateJointEnabledAttr(True)
        settings = carb.settings.get_settings()
        settings.set('/physics/suppressReadback', False)
        settings.set('/rtx/rendermode', self.renderer)
        settings.set('/rtx/pathtracing/spp', getattr(self, 'render_samples', 4))
        settings.set('/rtx/pathtracing/totalSpp', getattr(self, 'render_samples', 4))
        settings.set('/rtx/pathtracing/optixDenoiser/enabled', True)
        from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN
        action = self.initial_action()
        for side, start in [('left', 0), ('right', 7)]:
            reset_joints(base, side+'_robot', dict(zip(ARM_JOINTS, action[start:start+6])))
            reset_joints(base, side+'_robot', dict.fromkeys(FINGER_JOINTS, FINGER_OPEN))
        self.views = {}
        for asset, body_name in [('workpiece', 'carriage'), ('workpiece', 'ham'),
                                 ('workpiece', 'portion'), ('workpiece', 'power_on'),
                                 ('workpiece', 'power_off'), ('envelope', 'flap')]:
            scene_name = asset+'_'+body_name
            self.views[(asset, body_name)] = base.scene[scene_name].root_physx_view
        self.portion_cut = False
        self.active = False
        self.start_pressed = False
        self.stroke_started_clear = False
        self.edge_contact_seen = False
        self.slicer_running = False
        self.blade_angle = 0.
        self.set_cut_visual(False)
        self.blade_spin = None
        if not getattr(self, 'physics_only', False):
            from pxr import UsdGeom
            machine = self.descriptions['workpiece']['bodies'][0]
            spin_index = next(i for i, shape in enumerate(machine['shapes']) if shape.get('blade_spin'))
            path = f'/World/envs/env_0/workpiece/machine/shape_{spin_index:03d}'
            prim = self.stage.GetPrimAtPath(path)
            rotation = prim.GetAttribute('xformOp:rotateY')
            self.blade_spin = (UsdGeom.XformOp(rotation) if rotation
                               else UsdGeom.Xformable(prim).AddRotateYOp())
            self.blade_spin.Set(0.)
        self.slicer_stopped = False
        self.portion_placed = False
        self.flap_closed = False
        self.unsafe_reason = ''
        self._measurement_error = None
        self._arm_blade_incursion = False
        self._packed_still_s = 0.
        self.time_s = 0.

    def native(self, name, part):
        view = self.views[(name, part)]
        state = np.array(view.get_transforms().numpy()[0], copy=True)
        velocity = np.array(view.get_velocities().numpy()[0], copy=True)
        if not np.isfinite(np.r_[state, velocity]).all():
            raise RuntimeError('Invalid native rigid-body state')
        return state[:3], state[3:], velocity

    def set_cut_visual(self, sliced):
        from pxr import UsdGeom
        for part in self.descriptions['workpiece']['bodies']:
            for index, shape in enumerate(part['shapes']):
                if tag := shape.get('cut_visual'):
                    path = f"/World/envs/env_0/workpiece/{part['name']}/shape_{index:03d}"
                    visual = UsdGeom.Imageable(self.stage.GetPrimAtPath(path))
                    if (tag == 'sliced') == sliced:
                        visual.MakeVisible()
                    else:
                        visual.MakeInvisible()

    def _release_cut_portion(self):
        self.cut_joint.GetJointEnabledAttr().Set(False)
        self.set_cut_visual(True)
        # Model the deflector's outward motion at the cut before the native drop.
        from torch import tensor
        self.base.scene['workpiece_portion'].write_root_velocity_to_sim(
            tensor([[0., .31, .035, 0., 0., 0.]], device=self.base.device))
        self.portion_cut = True

    def _update_slicer(self, m):
        start = m['start_button_travel_m'] >= SWITCH_ACTUATION
        stop = m['stop_button_travel_m'] >= SWITCH_ACTUATION
        if stop:
            self.slicer_stopped |= self.slicer_running
            self.slicer_running = False
        elif start and not self.start_pressed:
            self.slicer_running = True
            self.slicer_stopped = False
        self.start_pressed = start

        if not self.slicer_running or self.portion_cut or self.unsafe_reason:
            self.stroke_started_clear = False
            self.edge_contact_seen = False
        elif not m['blade_edge_reached']:
            self.stroke_started_clear = True
            self.edge_contact_seen = False
        elif self.stroke_started_clear:
            carriage_x = m['carriage_position_m'][0]
            if carriage_x > self.previous_carriage_x:
                self.edge_contact_seen = True
            at_end = carriage_x >= CARRIAGE_START[0] + CARRIAGE_TRAVEL_LIMIT - .001
            if (at_end and self.edge_contact_seen and m['blade_sweep_complete']
                    and not m['portion_overlaps_blade_guard']):
                self._release_cut_portion()
        self.previous_carriage_x = m['carriage_position_m'][0]

    def measurements(self):
        from yam_arena.urdf import quat_xyzw_to_matrix
        carriage = self.native('workpiece', 'carriage')
        ham, ham_quat, _ = self.native('workpiece', 'ham')
        ham_rot = quat_xyzw_to_matrix(ham_quat)
        ham_front = ham[1] + ham_rot[1] @ HAM_COLLIDER_CENTER + np.abs(ham_rot[1]) @ HAM_HALF_SIZE
        ham_blade_clearance = BLADE_CENTER[1] - BLADE_HALF_THICKNESS - ham_front
        portion = self.native('workpiece', 'portion')
        flap = self.native('envelope', 'flap')
        position, rotation, velocity = portion
        extent = np.abs(quat_xyzw_to_matrix(rotation)) @ PORTION_HALF_SIZE
        relative = np.abs(position-BLADE_CENTER)
        nearest = np.maximum(relative[[0, 2]]-extent[[0, 2]], 0.)
        touches_edge = np.linalg.norm(nearest) <= BLADE_RADIUS
        swept_face = np.linalg.norm(relative[[0, 2]]+extent[[0, 2]]) <= BLADE_RADIUS
        guard_overlap = (position[1]+extent[1] >= BLADE_CENTER[1]+GUARD_NORMAL_OFFSET-GUARD_ROD_RADIUS
                         and position[1]-extent[1] <= BLADE_CENTER[1]+.0016
                         and np.linalg.norm(nearest) <= BLADE_GUARD_RADIUS)
        lower, upper = position-extent, position+extent
        in_envelope = slice_inside_envelope(lower, upper)
        released = in_envelope and self._arm_bounds.clear_of_slice(lower, upper)
        speed = float(np.linalg.norm(velocity[:3]))
        still = speed <= .010 and float(np.linalg.norm(velocity[3:])) <= .20
        flap_normal = quat_xyzw_to_matrix(flap[1]) @ np.array([0., 1., 0.])
        angle = math.acos(float(np.clip(flap_normal[2], -1., 1.)))
        self.portion_placed = bool(in_envelope)
        self.flap_closed = bool(angle <= math.radians(10) and np.linalg.norm(flap[2][3:]) <= .20)
        return dict(time_s=self.time_s, portion_cut=self.portion_cut,
                    slicer_running=self.slicer_running,
                    start_button_travel_m=float(POWER_ON[2]-self.native('workpiece', 'power_on')[0][2]),
                    stop_button_travel_m=float(POWER_OFF[2]-self.native('workpiece', 'power_off')[0][2]),
                    slicer_stopped=self.slicer_stopped, portion_in_envelope=self.portion_placed,
                    portion_released=released, portion_still=still,
                    envelope_flap_angle_rad=angle, envelope_closed=self.flap_closed,
                    carriage_position_m=carriage[0].tolist(), portion_position_m=position.tolist(),
                    portion_speed_m_s=speed,
                    ham_blade_clearance_m=float(ham_blade_clearance),
                    blade_edge_reached=bool(touches_edge), blade_sweep_complete=bool(swept_face),
                    portion_overlaps_blade_guard=bool(guard_overlap), packed_still_s=self._packed_still_s,
                    arm_blade_incursion=self._arm_blade_incursion, unsafe_event=bool(self.unsafe_reason))

    def _check_arm_blade_clearance(self, dt):
        if self._arm_blade_incursion or self._measurement_error:
            return
        try:
            self._arm_blade_incursion = blade_zone_intersects(*self._arm_bounds.world())
        except Exception as error:
            self._measurement_error = error

    def begin_episode(self):
        from omni.physx import get_physx_interface

        self.time_s = 0.
        self._arm_bounds = ArmBounds(self.base)
        self.start_metrics = self.measurements()
        self.previous_carriage_x = self.start_metrics['carriage_position_m'][0]
        self._check_arm_blade_clearance(0.)
        self._monitor = get_physx_interface().subscribe_physics_on_step_events(
            self._check_arm_blade_clearance, pre_step=False, order=0)
        self.active = True

    def evaluate(self, base):
        if self._measurement_error is not None:
            raise RuntimeError('Blade clearance measurement failed') from self._measurement_error
        if self.active:
            self.time_s += float(base.step_dt)
        m = self.measurements()
        if self._arm_blade_incursion:
            self.unsafe_reason = 'arm or fingertip entered the blade zone'
        elif m['ham_blade_clearance_m'] < 0.:
            self.unsafe_reason = 'main ham crossed the infeed face of the blade'
        elif m['portion_cut'] and m['portion_overlaps_blade_guard']:
            self.unsafe_reason = 'released slice entered the blade or guard zone'
        if self.active:
            self._update_slicer(m)
            if self.slicer_running and self.blade_spin is not None:
                self.blade_angle = (self.blade_angle + 950.*base.step_dt) % 360.
                self.blade_spin.Set(self.blade_angle)
        m.update(portion_cut=self.portion_cut, slicer_running=self.slicer_running,
                 slicer_stopped=self.slicer_stopped, unsafe_event=bool(self.unsafe_reason))
        m['carriage_x_m'] = m.pop('carriage_position_m')[0]
        del m['portion_position_m']  # Native vectors remain in trace_objects().
        packed = (not self.unsafe_reason and self.portion_cut and self.slicer_stopped
                  and m['portion_in_envelope'] and m['portion_released']
                  and m['portion_still'] and m['envelope_closed'])
        self._packed_still_s = self._packed_still_s + float(base.step_dt) if packed else 0.
        m['packed_still_s'] = self._packed_still_s
        success = self._packed_still_s >= 1. - 1e-9
        return dict(success=success, score=float(success), **m)

    def trace_objects(self):
        return {f'{name}/{part}': [value.tolist() for value in self.native(name, part)]
                for name, part in [('workpiece', 'carriage'), ('workpiece', 'portion'), ('envelope', 'flap')]}

    def close(self):
        self._monitor = None
        if hasattr(self, '_temporary'):
            self._temporary.cleanup()


def make_task(condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    return ProsciuttoTask()
