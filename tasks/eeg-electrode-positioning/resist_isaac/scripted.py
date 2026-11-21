"""Play a task folder's bimanual Waypoint demo with the existing YAM IK policy."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from yam_arena import ik, station
from yam_arena.demo import ArmPlan, Waypoint as YamWaypoint, grip_rotation
from yam_arena.constants import FINGER_OPEN
from yam_arena.scripted import YamScriptedPolicy
from yam_arena.urdf import quat_xyzw_to_matrix

from .runtime import _task_module


@dataclass
class Waypoint(YamWaypoint):
    """YAM waypoint with an optional full orientation for side grasps."""

    rotation_xyzw: tuple[float, float, float, float] | None = None
    blend: bool = False


def _smootherstep(value: float) -> float:
    """Quintic easing with zero velocity and acceleration at both ends."""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


class CartesianArmPlan(ArmPlan):
    def __init__(self, waypoints, base_pos):
        if any(w.duration <= 0 or not FINGER_OPEN <= w.finger <= 0 for w in waypoints):
            raise ValueError("Waypoints need positive duration and finger targets between FINGER_OPEN and zero")
        super().__init__(waypoints, base_pos)
        self._approach_q = None
        if waypoints and getattr(waypoints[0], "rotation_xyzw", None) is not None:
            first = waypoints[0]
            goal, position_error, rotation_error = ik.solve(
                first.pos, first.rotation_xyzw, base_pos=base_pos)
            if position_error > 0.002 or rotation_error > 0.02:
                raise ValueError(f"Unreachable approach: {position_error=}, {rotation_error=}")
            self._approach_q = goal
            folded = self.q.copy()
            folded[1:4] = [0.6, 2.0, -1.4]
            turned = folded.copy()
            turned[0] = goal[0]
            self._approach_stages = [self.q.copy(), folded, turned, goal]
            self._approaching = True

    def action7(self, t):
        # Side grasps first move through joint space to a checked staging pose.
        # This avoids a wrist-limit IK branch during the initial 180-degree turn.
        if self._approach_q is not None and self._approaching:
            first = self.waypoints[0]
            if t <= first.duration:
                progress = min(3 * t / first.duration, 3.0)
                index = min(int(progress), 2)
                s = _smootherstep(progress - index)
                before, after = self._approach_stages[index:index + 2]
                self.q = before + s * (after - before)
                finger = self.finger_start + _smootherstep(t / first.duration) * (first.finger - self.finger_start)
                return np.r_[self.q, finger / FINGER_OPEN], first.label
            self.q = self._approach_q.copy()
            self._approaching = False
        return super().action7(t)

    @staticmethod
    def _tangent(index, positions, times, waypoints):
        """Return a bounded pass-through tangent at a marked transit waypoint."""
        if index <= 0 or index >= len(positions) - 1:
            return np.zeros(3)
        waypoint = waypoints[index - 1]
        if not getattr(waypoint, "blend", False):
            return np.zeros(3)
        before, point, after = positions[index - 1:index + 2]
        before_dt = times[index] - times[index - 1]
        after_dt = times[index + 1] - times[index]
        before_distance = float(np.linalg.norm(point - before))
        after_distance = float(np.linalg.norm(after - point))
        if min(before_distance, after_distance) < 1e-6:
            return np.zeros(3)
        tangent = (after - before) / (times[index + 1] - times[index - 1])
        limit = 1.5 * min(before_distance / before_dt, after_distance / after_dt)
        speed = float(np.linalg.norm(tangent))
        if speed > limit:
            tangent *= limit / speed
        return tangent

    def target(self, t):
        positions = [np.asarray(self.start_pos, dtype=float)]
        rotations = [self.start_rot]
        fingers = [self.finger_start]
        times = [0.0]
        for waypoint in self.waypoints:
            quaternion = getattr(waypoint, "rotation_xyzw", None)
            rotation = (quat_xyzw_to_matrix(np.asarray(quaternion)) if quaternion is not None
                        else grip_rotation(waypoint.yaw))
            positions.append(np.asarray(waypoint.pos, dtype=float))
            rotations.append(rotation)
            fingers.append(waypoint.finger)
            times.append(times[-1] + waypoint.duration)

        if t >= times[-1]:
            return positions[-1], rotations[-1], fingers[-1], "done"
        segment = next(i for i in range(len(self.waypoints)) if t < times[i + 1])
        duration = times[segment + 1] - times[segment]
        u = float(np.clip((t - times[segment]) / duration, 0.0, 1.0))
        p0, p1 = positions[segment], positions[segment + 1]
        m0 = self._tangent(segment, positions, times, self.waypoints)
        m1 = self._tangent(segment + 1, positions, times, self.waypoints)
        if np.any(m0) or np.any(m1):
            u2, u3 = u * u, u * u * u
            pos = ((2 * u3 - 3 * u2 + 1) * p0
                   + (u3 - 2 * u2 + u) * duration * m0
                   + (-2 * u3 + 3 * u2) * p1
                   + (u3 - u2) * duration * m1)
            pos = np.clip(pos, np.minimum(p0, p1), np.maximum(p0, p1))
        else:
            pos = p0 + _smootherstep(u) * (p1 - p0)
        s = _smootherstep(u)
        rotation = ik.slerp_matrix(rotations[segment], rotations[segment + 1], s)
        finger = fingers[segment] + s * (fingers[segment + 1] - fingers[segment])
        return pos, rotation, finger, self.waypoints[segment].label


class ResistScriptedPolicy(YamScriptedPolicy):
    """Load ``demo()`` from the same task folder passed to ``resist_isaac``."""

    def __init__(self, task: str, hold_s: float = 3.0):
        super().__init__(hold_s=hold_s)
        self.task_path = Path(task).expanduser().resolve()
        self._module = None
        self.info = replace(self.info, name="resist_scripted")

    def bind(self, embodiment_info):
        super().bind(embodiment_info)
        self._bases = dict(station.BASES)
        self._module, _ = _task_module(self.task_path)

    def plan_for(self, instruction, target=None):
        if self._module is None or not callable(getattr(self._module, "demo", None)):
            raise ValueError(f"{self.task_path} must expose demo() for resist_scripted")
        plan = dict(self._module.demo())
        unknown = sorted(set(plan) - {"left", "right"})
        if unknown:
            raise ValueError(f"Demo keys must be arm names 'left' or 'right', got {unknown}")
        return plan

    def reset(self, scene):
        plan = self.plan_for(scene.instruction, scene.target)
        self._plans = {arm: CartesianArmPlan(list(points), self._bases[arm])
                       for arm, points in plan.items()}
        self._end = max((plan.duration for plan in self._plans.values()), default=0.0)


def resist_scripted_policy(task: str, hold_s: float = 3.0):
    """Create a scripted Cartesian waypoint policy for an authored RESIST task."""
    return ResistScriptedPolicy(task=task, hold_s=hold_s)
