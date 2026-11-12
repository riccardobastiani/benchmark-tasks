"""Open the sliding glass door while retaining its loose track obstruction."""

import math
from pathlib import Path
import runpy
import tempfile

from resist_isaac.state import pose_state, reset_joints, unit_quaternion


CONDITIONS = ("recoverable",)
INSTRUCTION = "Slide the front glass door fully open. Keep the loose key within x=0.05-0.75 m, y=-0.50-0.50 m and z=0.745-1.30 m in the station frame (x forward, y left; z from the floor)."
FRAME_POSITION = (0.53, 0.0, 0.7475)
KEY_POSITION = (0.3776, 0.10, 0.7875)


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


class ObstructedSlidingTrack:
    """A fixed display, passive sliding glass pane and independent loose key."""

    hold_s = 1.0
    robot_contact_offset = 0.001

    @staticmethod
    def configure_env(env_cfg):
        import numpy as np
        from isaaclab.sensors import CameraCfg
        from isaaclab.sim import PinholeCameraCfg
        from yam_arena.ik import matrix_to_quat_xyzw

        # An overview includes the tall display, both arms and the loose key.
        eye = np.array([-0.65, 0.0, 1.95])
        back = eye - np.array([0.32, 0.0, 0.94])
        back /= np.linalg.norm(back)
        right = np.cross([0., 0., 1.], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        rotation = matrix_to_quat_xyzw(np.column_stack((right, up, back)))
        env_cfg.scene.top_cam.offset = CameraCfg.OffsetCfg(
            pos=tuple(eye), rot=tuple(rotation), convention="opengl")
        env_cfg.scene.top_cam.spawn = PinholeCameraCfg(
            focal_length=22.0, horizontal_aperture=36.0, clipping_range=(0.05, 20.0))
        # Keep the station's physical camera stand at the side of the bench,
        # clear of the independent overview camera's line of sight.
        for name in ("stand_foot", "stand_mast", "stand_arm", "camera_housing"):
            config = getattr(env_cfg.scene, name)
            x, y, z = config.init_state.pos
            config.init_state.pos = (x + 0.12, y + 0.58, z)

    def reset(self, base):
        self._key_escaped = False
        self._invalid = False
        reset_joints(base, "display", {"door_slide": 0.0})

    def build_scene(self, embodiment):
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab_arena.assets.object import Object, ObjectType
        from isaaclab_arena.utils.pose import Pose

        generator = runpy.run_path(str(Path(__file__).with_name("build_scene.py")))["create_assets"]
        self._asset_dir = tempfile.TemporaryDirectory(prefix="obstructed-sliding-track-")
        assets = generator(self._asset_dir.name)
        display = Object(name="display", object_type=ObjectType.ARTICULATION,
                         usd_path=str(assets["display"]), initial_pose=Pose(position_xyz=FRAME_POSITION))
        display.object_cfg.actuators = {"passive": ImplicitActuatorCfg(
            joint_names_expr=["door_slide"], stiffness=None, damping=None)}
        return [
            display,
            Object(name="track_key", object_type=ObjectType.RIGID, usd_path=str(assets["track_key"]), initial_pose=Pose(position_xyz=KEY_POSITION)),
        ]

    def evaluate(self, base):
        display = base.scene["display"]
        key = base.scene["track_key"]
        joint = display.joint_names.index("door_slide")
        slide = float(_tensor(display.data.joint_pos)[0, joint].item())
        speed = float(_tensor(display.data.joint_vel)[0, joint].item())
        body_positions = _tensor(display.data.body_link_pose_w)[0, :, :3]
        frame_position = body_positions[display.body_names.index("Frame")]
        door_position = body_positions[display.body_names.index("Door")]
        frame_error = float((frame_position - frame_position.new_tensor(FRAME_POSITION)).norm().item())
        expected = frame_position + frame_position.new_tensor([-0.1524, -0.15 + slide, 0.232])
        slide_error = float((door_position - expected).norm().item())
        key_position, key_quaternion, key_velocity = pose_state(key)
        key_speed = float(key_velocity[:3].norm().item())
        values = [slide, speed, frame_error, slide_error, key_speed] + [float(v) for v in key_position.tolist()]
        if not all(math.isfinite(v) for v in values):
            return {"success": False, "score": 0.0, "invalid_state": 1.0}
        orientations = _tensor(display.data.body_link_pose_w)[0, :, 3:]
        door_quaternion = orientations[display.body_names.index("Door")]
        frame_quaternion = orientations[display.body_names.index("Frame")]
        if not all(unit_quaternion(q) for q in (door_quaternion, frame_quaternion, key_quaternion)):
            self._key_escaped = True
            return {"success": False, "score": 0.0, "invalid_state": 1.0}
        orientation_error = 2 * math.acos(min(1.0, abs(float(door_quaternion[3].item()))))
        frame_orientation_error = 2 * math.acos(min(1.0, abs(float(frame_quaternion[3].item()))))
        key_x, key_y, key_z = map(float, key_position.tolist())
        key_retained = 0.05 <= key_x <= 0.75 and -0.50 <= key_y <= 0.50 and 0.745 <= key_z <= 1.30
        self._key_escaped = getattr(self, "_key_escaped", False) or not key_retained
        intact = frame_error <= 0.002 and frame_orientation_error <= 0.01 and slide_error <= 0.002 and orientation_error <= 0.01 and -0.001 <= slide <= 0.291 and not self._key_escaped
        self._invalid = self._invalid or not intact
        intact = not self._invalid
        success = intact and slide >= 0.28 and abs(speed) < 0.02 and key_speed < 0.02
        score = 1.0 if success else (0.5 if intact and slide >= 0.14 else 0.25 if intact and slide >= 0.05 else 0.0)
        return {"success": success, "score": score, "slide_m": slide, "slide_speed_m_s": speed, "frame_anchor_error_m": frame_error, "slide_anchor_error_m": slide_error, "door_orientation_error_rad": orientation_error, "key_retained": float(key_retained), "key_speed_m_s": key_speed,
                "key_x_m": key_x, "key_y_m": key_y, "key_z_m": key_z, "invalid_state": self._invalid}


def make_task(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported track condition: {condition}")
    return ObstructedSlidingTrack()


def demo():
    """Remove the loose key with the left hand, then slide with the right."""
    from resist_isaac.scripted import Waypoint
    from yam_arena.constants import FINGER_OPEN

    # Native reset leaves the short leg resting across the front track lip.
    pick = (0.3476, 0.0993, 0.800)
    place = (0.20, 0.10, 0.7645)
    lean = (0.0, math.sin(math.pi / 3), 0.0, math.cos(math.pi / 3))
    down = (0.0, 1.0, 0.0, 0.0)
    opening = 0.4 * FINGER_OPEN

    def key_step(t, pos, grip, label, rotation=lean):
        return Waypoint(t, pos, grip, label, rotation_xyzw=rotation)

    left = [
        key_step(3.0, (0.30, pick[1], 0.87), opening, "approach loose key"),
        key_step(2.0, pick, opening, "reach short leg"),
        key_step(0.8, pick, 0.0, "grasp key"),
        key_step(1.2, (pick[0], pick[1], 0.86), 0.0, "lift clear of track"),
        key_step(1.3, (place[0], place[1], 0.86), 0.0, "bring key onto bench"),
        key_step(1.2, place, 0.0, "lower key", down),
        key_step(0.7, place, FINGER_OPEN, "leave key on bench", down),
        key_step(0.7, (place[0], place[1], 0.86), FINGER_OPEN, "clear key", down),
        key_step(1.0, (0.18, 0.28, 0.90), FINGER_OPEN, "park clear of sliding arm", down),
    ]
    front = (-0.5, 0.5, -0.5, 0.5)
    handle = (FRAME_POSITION[0] - 0.1524 - 0.024 - 0.008, -0.225, 0.9795)

    def side(t, pos, grip, label):
        return Waypoint(t, pos, grip, label, rotation_xyzw=front)

    opening = 0.4 * FINGER_OPEN
    right = [
        side(3.0, (handle[0] - 0.035, handle[1], handle[2]), opening, "approach door pull"),
        side(1.5, handle, opening, "reach door pull"),
        side(0.8, handle, 0.0, "grasp door pull"),
        side(7.0, handle, 0.0, "wait until key is on bench"),
        side(4.0, (handle[0], handle[1] + 0.286, handle[2]), 0.0, "slide pane open"),
        side(2.0, (handle[0], handle[1] + 0.286, handle[2]), 0.0, "hold fully open"),
    ]
    return {"left": left, "right": right}
