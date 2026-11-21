"""Apply six labeled EEG training electrodes to their prescribed mannequin sites."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import tempfile

import numpy as np

from resist_isaac.usd_assets import create_asset
from .geometry import (
    FOLDER as GEOMETRY_FOLDER, GRIP_CENTER_HEIGHT, HEAD_CENTER, HEAD_RADII, ORDER, PAD_TIP_PAST_TCP,
    STARTS, TABLE_TOP, PUCK_HALF_HEIGHT, GEL_THICKNESS,
    geometry, site_position, target_position,
)


CONDITIONS = ("normal",)
FOLDER = Path(__file__).parent
INSTRUCTION = "Place the six EEG electrodes on the mannequin in the order shown on the card."


def demo():
    """Seat the electrodes in card order, alternating arms while keeping the leads clear."""
    from resist_isaac.scripted import Waypoint
    from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
    from yam_arena.demo import DOWN_GRIP

    down = tuple(float(value) for value in DOWN_GRIP)
    arm_for = {"Fp1": "left", "Fp2": "right", "C3": "left",
               "C4": "right", "O1": "left", "O2": "right"}

    def pick_and_place(site):
        sx, sy, sz = STARTS[site]
        tx, ty, tz = target_position(site)
        command_x, command_y = tx, ty
        pick_z = sz + GRIP_CENTER_HEIGHT + PAD_TIP_PAST_TCP
        place_z = tz + GRIP_CENTER_HEIGHT + PAD_TIP_PAST_TCP
        # The raised lead connector keeps the large gripper clear of the scalp.
        # Carry just above the head, route around the same-side edge, then enter
        # the target from outside the mannequin's horizontal footprint. A direct
        # diagonal crosses the high crown and pushes the electrode off its seat.
        # The opened gripper must rise above the connector before moving
        # laterally; the shorter lift grazes the released electrode on exit.
        travel_z = max(TABLE_TOP + 0.27, place_z + 0.060)
        clear_z = TABLE_TOP + 0.11
        route_x = 0.40
        dx = tx - float(HEAD_CENTER[0])
        side_edge = float(HEAD_RADII[1]) * np.sqrt(max(0.0, 1.0 - (dx / HEAD_RADII[0]) ** 2))
        route_dx = route_x - float(HEAD_CENTER[0])
        route_edge = float(HEAD_RADII[1]) * np.sqrt(
            max(0.0, 1.0 - (route_dx / HEAD_RADII[0]) ** 2))
        side_clearance = max(abs(ty) + 0.030, side_edge + 0.015, route_edge + 0.015)
        side_y = side_clearance if ty > 0 else -side_clearance
        withdraw_y = 0.28 if ty > 0 else -0.28
        return [
            Waypoint(4.5, (sx, sy, clear_z), FINGER_OPEN, f"approach {site}", rotation_xyzw=down),
            Waypoint(1.8, (sx, sy, pick_z), FINGER_OPEN, f"reach {site} electrode", rotation_xyzw=down),
            Waypoint(0.8, (sx, sy, pick_z), FINGER_CLOSED, f"grasp {site}", rotation_xyzw=down),
            Waypoint(0.5, (sx, sy, pick_z), FINGER_CLOSED, f"secure {site}", rotation_xyzw=down),
            Waypoint(1.2, (sx, sy, clear_z), FINGER_CLOSED, f"lift {site}",
                     rotation_xyzw=down, blend=True),
            Waypoint(1.2, (route_x, sy, clear_z), FINGER_CLOSED, f"move outside head {site}",
                     rotation_xyzw=down, blend=True),
            Waypoint(1.0, (route_x, sy, travel_z), FINGER_CLOSED, f"raise beside head {site}",
                     rotation_xyzw=down, blend=True),
            Waypoint(1.2, (route_x, side_y, travel_z), FINGER_CLOSED, f"route around head {site}",
                     rotation_xyzw=down, blend=True),
            Waypoint(2.0, (command_x, side_y, travel_z), FINGER_CLOSED, f"carry alongside head {site}",
                     rotation_xyzw=down, blend=True),
            Waypoint(1.4, (command_x, command_y, travel_z), FINGER_CLOSED, f"align above {site}", rotation_xyzw=down),
            Waypoint(1.8, (command_x, command_y, place_z), FINGER_CLOSED, f"seat {site}", rotation_xyzw=down),
            Waypoint(0.8, (command_x, command_y, place_z), -0.012, f"release {site}", rotation_xyzw=down),
            Waypoint(0.8, (command_x, command_y, travel_z), FINGER_OPEN,
                     f"lift clear {site}", rotation_xyzw=down),
            # Leave along the same-side route before withdrawing. Retreating
            # across the head at the target's y coordinate sweeps the open
            # gripper over previously seated frontal/central electrodes.
            Waypoint(1.0, (command_x, side_y, travel_z), FINGER_OPEN,
                     f"retreat outside head {site}", rotation_xyzw=down, blend=True),
            Waypoint(2.4, (0.40, withdraw_y, travel_z), FINGER_OPEN, f"withdraw clear {site}", rotation_xyzw=down),
        ]

    steps = {side: [] for side in ("left", "right")}
    durations = {side: 0.0 for side in ("left", "right")}
    elapsed = 0.0
    # Park each arm on its own outer side throughout the other arm's turn.
    # Returning to the calibrated ready pose crosses the mannequin and can
    # disturb a newly placed electrode.
    def clear_pose(side):
        y = 0.28 if side == "left" else -0.28
        return (0.40, y, TABLE_TOP + 0.22)

    for site in ORDER:
        side = arm_for[site]
        if durations[side] < elapsed - 1e-6:
            dwell = elapsed - durations[side]
            steps[side].append(Waypoint(dwell, clear_pose(side), FINGER_OPEN,
                                        "wait clear for next electrode", rotation_xyzw=down))
            durations[side] += dwell
        action = pick_and_place(site)
        steps[side].extend(action)
        duration = sum(waypoint.duration for waypoint in action)
        durations[side] += duration
        elapsed += duration

    longer = max(durations.values())
    for side in ("left", "right"):
        remaining = longer - durations[side]
        if remaining > 1e-6:
            steps[side].append(Waypoint(remaining, clear_pose(side), FINGER_OPEN,
                                        "wait clear while sequence finishes", rotation_xyzw=down))
        steps[side].append(Waypoint(1.5, clear_pose(side), FINGER_OPEN,
                                    "observe completed EEG setup", rotation_xyzw=down))
    return steps


class EEGPlacementTask:
    """Native PhysX scalp placement; no electrical signal is simulated."""

    hold_s = 1.0
    robot_contact_offset = 0.001
    render_hz = 10

    def __init__(self):
        self.assets, self.poses = geometry()
        self._temporary = None
        self._views = {}
        self._stable_s = {site: 0.0 for site in ORDER}
        self._stability_history = {site: deque(maxlen=6) for site in ORDER}
        self._events = {}
        self._elapsed = 0.0

    def build_scene(self, embodiment):
        from isaaclab_arena.assets.object import Object, ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.utils.pose import Pose

        self._temporary = tempfile.TemporaryDirectory(prefix="eeg-placement-")
        result = []
        for name, description in self.assets.items():
            path = create_asset(description, Path(self._temporary.name) / (name + ".usda"), GEOMETRY_FOLDER)
            obj = Object(name=name, object_type=ObjectType.BASE, usd_path=str(path),
                         initial_pose=Pose(position_xyz=tuple(self.poses[name])))
            result.append(obj)
            if name.startswith("electrode_"):
                result.append(ObjectReference(parent_asset=obj, name=name + "_body",
                    prim_path=obj.get_prim_path() + "/body", object_type=ObjectType.RIGID))
        return result

    @staticmethod
    def initial_action():
        from yam_arena.constants import READY
        return np.asarray([*READY, 1.0, *READY, 1.0], dtype=float)

    def configure_env(self, cfg):
        # Adhesive joints are authored at contact during the run, so use the
        # USD update path, as the repository's other dynamic-joint tasks do.
        cfg.sim.use_fabric = False
        # The Arena observer expects both wrist sensors to exist, but their
        # images are not part of this top-view review clip.
        for name in ("left_cam", "right_cam"):
            camera = getattr(cfg.scene, name)
            camera.width = min(camera.width, 320)
            camera.height = min(camera.height, 180)
            camera.update_period = 0.2
        cfg.scene.top_cam.update_period = 1.0 / self.render_hz

    def reset(self, base):
        import carb.settings
        import omni.usd
        settings = carb.settings.get_settings()
        settings.set("/physics/suppressReadback", False)
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", 4)
        settings.set("/rtx/pathtracing/totalSpp", 4)
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        self._base = base
        stage = omni.usd.get_context().get_stage()
        self._stage = stage
        for site in ORDER:
            stage.RemovePrim(f"/World/EEGPasteBonds/{site}")
        self._gel_bonds = {}
        self._bonded = set()
        self._previous_positions = {}
        self._views = {site: base.scene["electrode_" + site.lower() + "_body"].root_physx_view
                       for site in ORDER}
        if any(view is None for view in self._views.values()):
            raise RuntimeError("Missing native electrode rigid-body views")
        self._stable_s = {site: 0.0 for site in ORDER}
        self._stability_history = {site: deque(maxlen=6) for site in ORDER}
        self._events = {}
        self._elapsed = 0.0

    def _native(self, site):
        view = self._views[site]
        transform = np.array(view.get_transforms().numpy()[0], copy=True)
        velocity = np.array(view.get_velocities().numpy()[0], copy=True)
        if not np.isfinite(np.r_[transform, velocity]).all():
            raise RuntimeError("Non-finite native electrode state: " + site)
        return transform[:3], transform[3:], velocity

    def evaluate(self, base):
        from yam_arena.runtime import tcp_world
        from pxr import Gf, UsdPhysics

        dt = float(base.step_dt)
        self._elapsed += dt
        positions, speeds, errors = {}, {}, {}
        for site in ORDER:
            pos, quat, velocity = self._native(site)
            # Contact-conditioned paste proxy. It works anywhere on the scalp,
            # including wrong sites; it never attracts an electrode to a target.
            # Retain the measured contact pose with an ideal native paste joint
            # after slow contact during pressing. No body pose is rewritten.
            if site not in self._bonded:
                rotation = Gf.Rotation(Gf.Quatd(float(quat[3]), Gf.Vec3d(*map(float, quat[:3]))))
                samples = [(0., 0., -PUCK_HALF_HEIGHT - GEL_THICKNESS)]
                samples += [(0.006 * np.cos(a), 0.006 * np.sin(a), samples[0][2])
                            for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
                points = np.array([pos + np.array(rotation.TransformDir(Gf.Vec3d(*p))) for p in samples])
                radial = np.linalg.norm((points - HEAD_CENTER) / HEAD_RADII, axis=1)
                gap = (float(radial.min()) - 1.0) * float(HEAD_RADII.min())
                previous = self._previous_positions.get(site, pos)
                contact_speed = float(np.linalg.norm(pos - previous) / dt)
                if (pos[2] > HEAD_CENTER[2] and -0.004 <= gap <= 0.0015
                        and contact_speed < 0.05):
                    joint = UsdPhysics.FixedJoint.Define(self._stage, f"/World/EEGPasteBonds/{site}")
                    joint.CreateBody1Rel().SetTargets([f"/World/envs/env_0/electrode_{site.lower()}/body"])
                    joint.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
                    joint.CreateLocalRot1Attr(Gf.Quatf(1.))
                    joint.CreateExcludeFromArticulationAttr(True)
                    self._gel_bonds[site] = joint
                    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos)))
                    joint.CreateLocalRot0Attr(Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3]))))
                    joint.CreateJointEnabledAttr(True)
                    self._bonded.add(site)
                    print(f"paste contact: {site} at {pos.tolist()} body_valid={bool(self._stage.GetPrimAtPath(str(joint.GetBody1Rel().GetTargets()[0])))}", flush=True)
            self._previous_positions[site] = pos.copy()
            error = float(np.linalg.norm(pos - target_position(site)))
            speed = float(np.linalg.norm(velocity[:3]))
            positions[site], speeds[site], errors[site] = pos, speed, error
            history = self._stability_history[site]
            history.append(pos.copy())
            stable_window = len(history) == history.maxlen and max(
                float(np.linalg.norm(sample - history[0])) for sample in history
            ) <= 0.010
            if site in self._bonded and error <= 0.014 and stable_window:
                self._stable_s[site] += dt
                if self._stable_s[site] >= 0.25 and site not in self._events:
                    self._events[site] = self._elapsed
            else:
                self._stable_s[site] = 0.0

        final_sites = [site for site in ORDER if site in self._bonded and errors[site] <= 0.014
                       and self._stable_s[site] >= 0.25]
        ordered_prefix = 0
        for site, _ in sorted(self._events.items(), key=lambda event: event[1]):
            if ordered_prefix < len(ORDER) and site == ORDER[ordered_prefix]:
                ordered_prefix += 1
            else:
                break

        head_xy = np.asarray([0.43, 0.0])
        hands = {side: tcp_world(base.scene[side + "_robot"]) for side in ("left", "right")}
        clearances = {side: float(np.linalg.norm(tcp[:2] - head_xy)) for side, tcp in hands.items()}
        hands_clear = all(clearance >= 0.20 and hands[side][2] >= TABLE_TOP + 0.12
                          for side, clearance in clearances.items())
        score = 0.6 * len(final_sites) / len(ORDER) + 0.4 * ordered_prefix / len(ORDER)
        success = len(final_sites) == len(ORDER) and ordered_prefix == len(ORDER) and hands_clear
        if success:
            score = 1.0
        else:
            score = min(score, 0.999)
        return dict(success=bool(success), score=float(score),
                    electrodes_seated=len(final_sites), ordered_prefix=ordered_prefix,
                    hands_clear=bool(hands_clear),
                    gel_bonds_created=len(self._bonded),
                    first_site_error_m=float(errors[ORDER[0]]),
                    first_site_stable_s=float(self._stable_s[ORDER[0]]),
                    first_site_speed_m_s=float(speeds[ORDER[0]]),
                    max_position_error_m=max(errors.values()),
                    min_hand_clearance_m=min(clearances.values()))

    def validate_reset(self, base):
        if self._events:
            raise RuntimeError("EEG reset unexpectedly contains placement events")
        for site in ORDER:
            expected = np.asarray(STARTS[site])
            actual, _, _ = self._native(site)
            if np.linalg.norm(actual - expected) > 0.006:
                raise RuntimeError(f"Invalid initial electrode pose for {site}: {actual.tolist()}")

    def close(self):
        if self._temporary is not None:
            self._temporary.cleanup()


def make_task(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported EEG placement condition: {condition}")
    return EEGPlacementTask()

