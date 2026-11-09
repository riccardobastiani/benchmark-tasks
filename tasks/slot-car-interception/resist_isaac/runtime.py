"""Run native passive apparatus with the installed YAM observation and control interface.

Task modules expose CONDITIONS and make_task(condition). The resulting task provides
build_scene(embodiment), evaluate(base), and optionally reset(base), hold_s, and
configure_native(env_cfg) for task-owned native scene configuration before warmup.
Only robot joint targets act on the scene after reset.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import math
import os
from dataclasses import replace
from numbers import Real
from pathlib import Path
import sys

import numpy as np
from inspect_robots.embodiment import PRIVILEGED_SUCCESS
from inspect_robots.types import StepResult
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

_APP = None


def _task_module(directory: Path):
    source = directory / "task.py"
    if not source.is_file():
        raise FileNotFoundError(f"No implemented Isaac task at {source}")
    sources = sorted(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    fingerprint = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        fingerprint.update(path.name.encode() + b"\0")
        fingerprint.update(hashlib.sha256(path.read_bytes()).digest())
    for path in sources:
        fingerprint.update(path.relative_to(directory).as_posix().encode() + b"\0")
        fingerprint.update(hashlib.sha256(path.read_bytes()).digest())
    digest = fingerprint.hexdigest()
    name = f"_resistance_task_{digest[:16]}"
    spec = importlib.util.spec_from_file_location(name, source, submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Isaac task at {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, digest


def _close_app():
    if _APP is not None:
        sys.stdout.flush()
        sys.stderr.flush()
        _APP.close()


class ResistanceEmbodiment(YamArenaEmbodiment):
    """A native Arena scene with fourteen robot controls and evaluator-only task state."""

    def __init__(self, *, task: str, condition: str = "recoverable", control_hz: float = 20, **kwargs):
        directory = Path(task).expanduser().resolve()
        module, digest = _task_module(directory)
        conditions = getattr(module, "CONDITIONS", ())
        if condition not in conditions:
            raise ValueError(f"Unsupported condition {condition!r}; implemented conditions: {conditions}")
        if kwargs.pop("object_jitter", 0) != 0:
            raise ValueError("These fixed task resets require object_jitter=0")
        if kwargs.pop("scene", "station") != "station":
            raise ValueError("These tasks use the calibrated station scene")
        self._mechanism = module.make_task(condition)
        self._physics_hz = float(getattr(self._mechanism, "physics_hz", 240))
        if not math.isfinite(self._physics_hz) or self._physics_hz < 240:
            raise ValueError("Task physics_hz must be finite and at least 240")
        if (not math.isfinite(control_hz) or control_hz <= 0
                or not (self._physics_hz / control_hz).is_integer()):
            raise ValueError("control_hz must be positive and divide the task physics rate exactly")
        for method in ("build_scene", "evaluate"):
            if not callable(getattr(self._mechanism, method, None)):
                raise TypeError(f"Task must implement {method}()")
        self._hold_s = float(getattr(self._mechanism, "hold_s", 1.0))
        if not math.isfinite(self._hold_s) or self._hold_s <= 0:
            raise ValueError("Task hold_s must be finite and positive")
        super().__init__(scene="station", object_jitter=0, control_hz=control_hz, **kwargs)
        self.task = self._mechanism
        if hasattr(self._mechanism, "overview_camera"):
            self.rig = replace(self.rig, top_cam=self._mechanism.overview_camera)
        room_docs = ROOM_DOCS["station"]
        if hasattr(self._mechanism, "overview_camera"):
            room_docs = room_docs.replace("from a stand behind them", "from the task's bench-side stand")
        self.info = replace(
            self.info,
            name="resist_isaac",
            supported_target_kinds=frozenset(),
            capabilities=self.info.capabilities | {PRIVILEGED_SUCCESS},
            docs="\n\n".join((JOINT_GUIDE, room_docs, (module.__doc__ or "").strip())),
            environment_id=f"resistance_{digest[:16]}",
            environment_revision=digest[:12],
        )
        self._partial_score = 0.0

    def _ensure_sim(self, goal=None):
        """Create the apparatus through Arena's public builder and native USD assets."""
        global _APP
        if self._env is not None:
            return
        if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in {"YES", "Y", "TRUE", "1"}:
            raise RuntimeError(
                "Isaac Sim requires your license acceptance. Review NVIDIA's terms and set "
                "OMNI_KIT_ACCEPT_EULA yourself; this runner does not accept them automatically."
            )
        if _APP is None:
            from isaaclab.app import AppLauncher

            launch = dict(headless=self.headless, device=self.device, enable_cameras=True, multi_gpu=False)
            if not self.headless:
                launch["visualizer"] = ["kit"]
            _APP = AppLauncher(**launch).app
            atexit.register(_close_app)

        import torch
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from yam_arena.embodiment import BimanualYamEmbodiment
        from yam_arena.scene import env_cfg_callback, task_room

        robot = BimanualYamEmbodiment(
            enable_cameras=True, contact_sensors=False, rig=self.rig, wrist_cam_mount=self.wrist_cam_mount,
            cam_height=self.cam_height, cam_width=self.cam_width,
            robot_contact_offset=getattr(self._mechanism, "robot_contact_offset", None),
        )
        apparatus = self._mechanism.build_scene(robot)
        if not apparatus:
            raise ValueError("Task build_scene() returned no apparatus")
        room = task_room("station", self.rig.top_cam)
        if hasattr(self._mechanism, "overview_camera"):
            from yam_arena import station
            from yam_arena.scene import _stand_part
            from yam_arena.urdf import quat_xyzw_to_matrix

            # The stock stand assumes a camera looking along +X. Keep its mast behind other views.
            camera = self.rig.top_cam
            back = quat_xyzw_to_matrix(np.asarray(camera.rot_xyzw))[:2, 2]
            back = back / np.linalg.norm(back)
            foot = np.asarray(getattr(self._mechanism, "overview_stand_foot",
                                      np.asarray(camera.pos[:2]) + .07 * back))
            on_table = np.all(np.abs(foot - np.asarray(station.TABLE_CENTER[:2])) + .08
                              < np.asarray(station.TABLE_SIZE[:2]) / 2)
            support = station.TABLE_TOP if on_table else station.FLOOR_Z
            yaw = math.atan2(-back[1], -back[0])
            mast = station._segment((*foot, support + .016), (*foot, camera.pos[2]))
            end = np.asarray(camera.pos[:2]) + .02 * back
            arm = station._segment((*foot, camera.pos[2]), (*end, camera.pos[2] + .02))
            parts = {
                "stand_foot": ("box", (.11, .11, .016), (*foot, support + .008),
                               (0., 0., math.sin(yaw / 2), math.cos(yaw / 2))),
                "stand_mast": ("capsule", (.012, mast[2]), mast[0], mast[1]),
                "stand_arm": ("capsule", (.009, arm[2]), arm[0], arm[1]),
            }
            room = [obj for obj in room if obj.name not in parts]
            room.extend(_stand_part(name, *part) for name, part in parts.items())
        configure_yam = env_cfg_callback(self.info.control_hz)

        def configure_native(env_cfg):
            env_cfg = configure_yam(env_cfg)
            env_cfg.sim.dt = 1.0 / self._physics_hz
            env_cfg.decimation = int(self._physics_hz / self.info.control_hz)
            env_cfg.sim.render_interval = env_cfg.decimation
            if callable(getattr(self._mechanism, "configure_native", None)):
                self._mechanism.configure_native(env_cfg)
            return env_cfg

        arena = IsaacLabArenaEnvironment(
            name=self.info.environment_id,
            scene=Scene([*room, *apparatus]),
            embodiment=robot,
            env_cfg_callback=configure_native,
        )
        config = ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=self.device)
        self._env = ArenaEnvBuilder(arena, config).make_registered()
        self._base = self._env.unwrapped
        self._torch = torch
        self._ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=self._base.device)
        self._ids = {
            name: (
                self._base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
                self._base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0],
            )
            for name in ARMS
        }

    def reset(self, scene, *, seed=None):
        """Restore the registered native reset and settle without jittering the apparatus."""
        self._ensure_sim()
        self._env.reset(seed=seed)
        if callable(getattr(self._mechanism, "reset", None)):
            self._mechanism.reset(self._base)
        settle_action = self._ready
        if callable(getattr(self._mechanism, "initial_action", None)):
            values = np.asarray(self._mechanism.initial_action(), dtype=float)
            if values.shape != (14,) or not np.isfinite(values).all():
                raise ValueError("Initial action must contain fourteen finite joint/gripper targets")
            if np.any(values < self._low) or np.any(values > self._high):
                raise ValueError("Initial action exceeds robot joint/gripper limits")
            settle_action = self._torch.tensor(values, dtype=self._torch.float32,
                                               device=self._base.device).unsqueeze(0)
        self._instruction = scene.instruction
        self._held = 0
        self._partial_score = 0.0
        for _ in range(max(1, math.ceil(0.5 / self._base.step_dt))):
            self._env.step(settle_action)
        for _ in range(4):
            self._base.sim.render()
        if callable(getattr(self._mechanism, "begin_episode", None)):
            self._mechanism.begin_episode(self._base)
        self._t = 0.0
        return self._observation()

    def step(self, action):
        """Validate the joint vector, then reuse the YAM action/observation implementation."""
        values = np.asarray(action.data)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("Action must contain exactly fourteen finite joint/gripper targets")
        return super().step(action)

    def _task_step(self):
        metrics = self._mechanism.evaluate(self._base)
        if not isinstance(metrics, dict) or not {"success", "score"} <= metrics.keys():
            raise ValueError("Task evaluate() must return success and score")
        candidate, score = metrics["success"], metrics["score"]
        if not isinstance(candidate, (bool, np.bool_)):
            raise ValueError("Task success must be boolean")
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Task score must be a finite number in [0, 1]")
        if bool(candidate) != (score == 1):
            raise ValueError("Task success must agree with its full-score endpoint")
        self._held = self._held + 1 if candidate else 0
        complete = self._held >= math.ceil(self._hold_s * self.info.control_hz)
        if score < 1:
            self._partial_score = float(score)
        info = {"success": complete, "score": 1.0 if complete else self._partial_score}
        for key, value in metrics.items():
            if key in info:
                continue
            if isinstance(value, (bool, np.bool_)):
                info[key] = bool(value)
            elif isinstance(value, Real) and math.isfinite(value):
                info[key] = float(value)
            else:
                raise ValueError(f"Task diagnostic {key!r} must be finite numeric or boolean")
        return StepResult(
            observation=self._observation(),
            terminated=complete,
            termination_reason="success" if complete else None,
            info=info,
        )

    def close(self):
        """Release the environment; keep Kit alive until the process has printed its scores."""
        if self._env is not None:
            try:
                if callable(getattr(self._mechanism, "close", None)):
                    self._mechanism.close()
            finally:
                try:
                    self._env.close()
                finally:
                    self._env = self._base = None
