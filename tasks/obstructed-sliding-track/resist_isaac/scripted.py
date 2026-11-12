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
                s = ik.smoothstep(progress - index)
                before, after = self._approach_stages[index:index + 2]
                self.q = before + s * (after - before)
                finger = self.finger_start + ik.smoothstep(t / first.duration) * (first.finger - self.finger_start)
                return np.r_[self.q, finger / FINGER_OPEN], first.label
            self.q = self._approach_q.copy()
            self._approaching = False
        return super().action7(t)

    def target(self, t):
        prev_pos, prev_rot, prev_finger = self.start_pos, self.start_rot, self.finger_start
        for waypoint in self.waypoints:
            goal = np.asarray(waypoint.pos, dtype=float)
            quaternion = getattr(waypoint, "rotation_xyzw", None)
            rotation = (quat_xyzw_to_matrix(np.asarray(quaternion)) if quaternion is not None
                        else grip_rotation(waypoint.yaw))
            if t < waypoint.duration:
                s = ik.smoothstep(t / waypoint.duration)
                return (prev_pos + s * (goal - prev_pos),
                        ik.slerp_matrix(prev_rot, rotation, s),
                        prev_finger + s * (waypoint.finger - prev_finger), waypoint.label)
            t -= waypoint.duration
            prev_pos, prev_rot, prev_finger = goal, rotation, waypoint.finger
        return prev_pos, prev_rot, prev_finger, "done"


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
