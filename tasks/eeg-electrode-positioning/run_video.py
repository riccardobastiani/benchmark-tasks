"""Run and record the native EEG task with the scripted or hosted Dream-1 policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

import numpy as np


FOLDER = Path(__file__).resolve().parent
REPO = FOLDER.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, choices=(10, 20), default=10)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--policy", choices=("scripted", "dream"), default="scripted")
    args = parser.parse_args()
    if min(args.width, args.height, args.max_steps) <= 0:
        parser.error("Output dimensions and control budget must be positive")
    if args.policy == "dream" and args.max_steps > 3000:
        parser.error("The EEG task permits at most 3,000 Dream-1 control steps")

    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"], text=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(FOLDER, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    receipt = dict(status="starting", task="eeg-electrode-positioning", host=socket.gethostname(),
                   gpu=gpu.strip().splitlines(), source_revision=revision,
                   resolution=[args.width, args.height], fps=args.fps, control_hz=20, seed=0,
                   policy=args.policy, model="dreamlabs/dream-1-beta" if args.policy == "dream" else None,
                   max_steps=args.max_steps,
                   scope="Native placement trial; not a clinically validated EEG setup")
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    robot = policy = None
    encoder = None
    trace = None
    started = time.monotonic()
    steps = 0
    final_info = {}
    try:
        import cv2
        from inspect_robots.scene import Scene
        from resist_isaac.runtime import ResistanceEmbodiment, _task_module
        from resist_isaac.scripted import resist_scripted_policy
        from resist_isaac.state import tensor
        from yam_arena import station
        from yam_arena.runtime import tcp_world

        module, fingerprint = _task_module(FOLDER)
        receipt.update(task_fingerprint=fingerprint, instruction=module.INSTRUCTION,
                       mechanism_sha256={name: hashlib.sha256((FOLDER / name).read_bytes()).hexdigest()
                                         for name in ("geometry.py", "task.py")})
        # This prototype uses a compact clinical-workstation arm placement so
        # both arms can reach the scalp targets without stretching across the head.
        bases = {
            "left": (0.20, 0.205, station.BASES["left"][2]),
            "right": (0.20, -0.205, station.BASES["right"][2]),
        }
        station.BASES = bases
        observation_size = (320, 180) if args.policy == "dream" else (args.width, args.height)
        robot = ResistanceEmbodiment(task=str(FOLDER), condition="normal", control_hz=20,
            headless=True, device=args.gpu, cam_width=observation_size[0], cam_height=observation_size[1],
            render_width=args.width, render_height=args.height)
        robot.task.render_hz = args.fps
        receipt.update(robot_base_positions=bases, observation_resolution=list(observation_size))
        scene = Scene(id="eeg-electrode-positioning-" + args.policy, instruction=module.INSTRUCTION)
        approver, guard_store = None, {}
        if args.policy == "dream":
            from inspect_robots.cli import _build_guardrails
            from inspect_robots_dream import DreamPolicy

            policy = DreamPolicy(model="dreamlabs/dream-1-beta", record=False, timeout_s=300.)
            approver, active, notices = _build_guardrails(robot.info.action_space, None, robot)
            receipt.update(guardrails=active, guardrail_notices=notices,
                           policy_versions={name: importlib.metadata.version(name) for name in
                               ("inspect-robots", "inspect-robots-dream", "dreamlabs-client")})
        else:
            policy = resist_scripted_policy(task=str(FOLDER), hold_s=2.0)
        policy.bind(robot.info)
        policy.reset(scene)
        observation = robot.reset(scene, seed=0)
        task = robot.task
        task.validate_reset(robot._base)
        for _ in range(12):
            robot._base.sim.render()

        def frame(obs):
            image = obs.images.get("top_cam") if obs is not None and args.policy == "scripted" else None
            if image is None:
                image = tensor(robot._base.scene["top_cam"].data.output["rgb"])[0, :, :, :3].cpu().numpy()
            if image.shape[:2] != (args.height, args.width):
                raise RuntimeError(f"Unexpected top-camera dimensions: {image.shape}")
            return np.ascontiguousarray(image)

        # The observation returned by reset contains a cached first camera
        # image, which can be the render target's gray clear frame. After the
        # warm-up renders, read the live sensor buffer for the setup still and
        # first video frame so the clip begins on the actual reset layout.
        initial = frame(None)
        cv2.imwrite(str(output / "setup-top.png"), cv2.cvtColor(initial, cv2.COLOR_RGB2BGR))
        encoder = subprocess.Popen([
            "ffmpeg", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{args.width}x{args.height}", "-framerate", str(args.fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "top-view.mp4")
        ], stdin=subprocess.PIPE)
        trace = (output / "trace.jsonl").open("w")
        stride = max(1, 20 // args.fps)

        def write_frame(obs):
            if steps % stride == 0:
                encoder.stdin.write(frame(obs).tobytes())

        encoder.stdin.write(initial.tobytes())
        max_steps = args.max_steps if args.policy == "dream" else min(
            args.max_steps, int(np.ceil((policy._end + policy.hold_s + 1.0) * 20)))
        for steps in range(1, max_steps + 1):
            sent = time.monotonic()
            chunk = policy.act(observation)
            action = chunk.actions[0]
            diagnostics = {}
            if approver is not None:
                raw = action
                action = approver.review(raw, guard_store)
                values = np.asarray(action.data)
                if values.shape != (14,) or not np.isfinite(values).all():
                    raise RuntimeError("Invalid Dream-1 action after guardrails")
                diagnostics = dict(raw_action=np.asarray(raw.data).tolist(),
                                   approval_meta=dict(action.meta),
                                   policy_wall_seconds=time.monotonic() - sent)
            result = robot.step(action)
            observation = result.observation
            final_info = result.info
            moment = steps / 20.0
            labels = ({side: "dream" for side in ("left", "right")} if args.policy == "dream" else
                      {side: plan.target(moment)[3] for side, plan in policy._plans.items()})
            positions = {site: task._native(site)[0].tolist() for site in task._views}
            trace.write(json.dumps(dict(step=steps, time_s=moment, labels=labels,
                                       action=np.asarray(action.data).tolist(), diagnostics=diagnostics,
                                       metrics=final_info, electrode_positions=positions,
                                       tcp_positions={side: tcp_world(robot._base.scene[side + "_robot"]).tolist()
                                                      for side in ("left", "right")})) + "\n")
            write_frame(observation)
            if steps % 20 == 0:
                trace.flush()
                print(f"step={steps} phase={labels} metrics={json.dumps(final_info)}", flush=True)
            if result.terminated or final_info.get("invalid_state") or final_info.get("escaped_workspace"):
                receipt["termination"] = result.termination_reason or "invalid_state"
                break
        receipt.setdefault("termination", "control_budget")

        final_image = frame(observation)
        cv2.imwrite(str(output / "final-top.png"), cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        receipt.update(status=("invalid" if final_info.get("invalid_state") or final_info.get("escaped_workspace")
                               else "completed") if args.policy == "dream" else
                               ("completed" if final_info.get("success") else "incomplete"),
                       final_metrics=final_info, steps=steps, duration_s=steps / 20.0,
                       task_fingerprint=fingerprint, robot_base_positions=bases)
    except BaseException:
        receipt.update(status="error", error=traceback.format_exc(), steps=steps,
                       final_metrics=final_info)
        raise
    finally:
        if args.policy == "dream" and policy is not None:
            try:
                (output / "dream-transcript.json").write_text(json.dumps(policy.transcript(), indent=2) + "\n")
            except Exception as error:
                receipt["transcript_error"] = repr(error)
            try:
                policy.close()
            except Exception as error:
                receipt["policy_close_error"] = repr(error)
        if trace is not None:
            trace.close()
        if encoder is not None:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            code = encoder.wait(timeout=60)
            if code:
                receipt["video_error"] = f"ffmpeg exited with {code}"
        if robot is not None:
            robot.close()
        receipt["wall_time_s"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"status": receipt["status"], "steps": steps,
                          "output": str(output), "metrics": final_info}, indent=2), flush=True)


if __name__ == "__main__":
    main()

