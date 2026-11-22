#!/usr/bin/env python3
"""Record a scored native PhysX trial with ordinary robot joint actions."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import traceback

import numpy as np


def source_hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--policy', choices=('scripted', 'dream', 'idle', 'hover'), default='scripted')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model', default='dreamlabs/dream-1-beta')
    args = parser.parse_args()
    folder, out = Path(__file__).resolve().parent, args.output.resolve()
    if out == folder or folder in out.parents:
        parser.error('Record outside the source directory')
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error('Video dimensions must be positive and even')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(folder, out/'source', ignore=shutil.ignore_patterns('__pycache__'))
    config = json.loads((folder/'pilot.json').read_text())
    receipt = dict(status='starting', policy=args.policy, hostname=socket.gethostname(),
        instruction=config['instruction'], seed=config['seed'], max_steps=config['max_steps'],
        control_hz=config['control_hz'], resolution=[args.width, args.height], policy_resolution=[320, 180],
        source_sha256=source_hashes(folder), model=args.model if args.policy == 'dream' else None,
        versions={name: importlib.metadata.version(name) for name in
                  ('isaacsim', 'isaaclab', 'isaaclab-arena', 'yam-arena')},
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), render_flags=[])
    def save_receipt():
        (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    save_receipt()
    started = time.monotonic()
    emb = policy = video = None
    steps, last, samples, nodes = 0, {}, [], []
    try:
        import cv2
        from inspect_robots import Scene
        from inspect_robots.types import Action
        from resist_isaac.runtime import ResistanceEmbodiment
        from resist_isaac.state import tensor, pose_state
        from yam_arena.constants import FINGER_JOINTS, FINGER_OPEN
        from yam_arena.runtime import tcp_world

        emb = ResistanceEmbodiment(task=str(folder), condition='recoverable', device=args.device,
            render_width=args.width, render_height=args.height, cam_width=320, cam_height=180)
        scene = Scene(id=folder.name, instruction=config['instruction'])
        observation = emb.reset(scene, seed=config['seed'])
        task, base = emb.task, emb._base
        receipt.update(status='running', physics_dt_s=base.physics_dt, control_dt_s=base.step_dt,
                       render_kit_args=task.render_kit_args, native_device=str(base.device))
        if abs(base.physics_dt-1/240) > 1e-8 or abs(base.step_dt-1/20) > 1e-8:
            raise RuntimeError('Unexpected physics/control rate')
        arms = {side: base.scene[side+'_robot'] for side in ('left', 'right')}
        fingers = {side: arm.find_joints(FINGER_JOINTS, preserve_order=True)[0] for side, arm in arms.items()}
        render_reference = None

        def frame():
            nonlocal render_reference
            rgb = tensor(base.scene['top_cam'].data.output['rgb'])[0, :, :, :3].cpu().numpy()
            if rgb.shape != (args.height, args.width, 3):
                raise RuntimeError('Native camera size mismatch; no upscaling permitted')
            if np.percentile(rgb[::8, ::8], 95)-np.percentile(rgb[::8, ::8], 5) < 12:
                receipt['render_flags'].append(dict(step=steps, issue='near-uniform frame'))
            if steps % 20 == 0:
                position = pose_state(base.scene['iron'])[0].cpu().numpy()
                landmarks = np.array([position, *(tcp_world(arm) for arm in arms.values())])
                if render_reference is None or np.linalg.norm(landmarks-render_reference[0], axis=1).max() >= .08:
                    sampled = rgb[::4, ::4].astype(np.int16)
                    if render_reference is not None:
                        changed = float(np.mean(np.max(np.abs(sampled-render_reference[1]), axis=2) > 20))
                        if changed < .006:
                            receipt['render_flags'].append(dict(step=steps, issue='possibly stale frame',
                                                                changed_fraction=changed))
                    render_reference = landmarks, sampled
            return np.ascontiguousarray(rgb)

        rgb = frame()
        cv2.imwrite(str(out/'setup.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        video = subprocess.Popen(['ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', f'{args.width}x{args.height}', '-framerate', '20',
            '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast',
            '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/'top-camera.mp4')],
            stdin=subprocess.PIPE)
        video.stdin.write(rgb.tobytes())

        def snapshot(phase):
            p, q, v = pose_state(base.scene['iron'])
            sample = dict(step=steps, phase=phase, iron_position_m=p.cpu().tolist(),
                iron_orientation_native=q.cpu().tolist(), iron_velocity=v.cpu().tolist(),
                tcp_m={side: np.asarray(tcp_world(arm)).tolist() for side, arm in arms.items()},
                opening={side: float(tensor(arm.data.joint_pos)[0, fingers[side]].mean())/FINGER_OPEN
                         for side, arm in arms.items()}, **last)
            samples.append(sample)
            nodes.append(task.nodes(base)[0])

        snapshot('reset')
        if args.policy == 'dream':
            from inspect_robots_dream import DreamPolicy
            from inspect_robots.cli import _build_guardrails
            policy = DreamPolicy(model=args.model, record=False, timeout_s=300.)
            policy.bind(emb.info)
            policy.reset(scene)
            approver, active, notices = _build_guardrails(emb.info.action_space, None, emb)
            receipt.update(guardrails=active, guardrail_notices=notices,
                policy_versions={name: importlib.metadata.version(name) for name in
                    ('inspect-robots', 'inspect-robots-dream', 'dreamlabs-client')})
            guard_store = {}
            def model_actions():
                for _ in range(config['max_steps']):
                    raw = policy.act(observation).actions[0]
                    approved = approver.review(raw, guard_store)
                    yield 'dream', np.asarray(approved.data).copy(), dict(
                        raw_action=np.asarray(raw.data).tolist(), approval_meta=dict(approved.meta))
            actions = model_actions()
        elif args.policy == 'idle':
            actions = (('idle', task.initial_action(), {}) for _ in range(config['max_steps']))
        else:
            from interaction import actions as scripted_actions
            actions = ((phase, command, {}) for phase, command in scripted_actions(
                task.initial_action(), task.iron_rest, hover=args.policy == 'hover'))
        save_receipt()
        previous_phase = None
        with (out/'trace.jsonl').open('w') as trace:
            for phase, command, diagnostics in actions:
                if steps >= config['max_steps']:
                    break
                if phase != previous_phase:
                    print(f'Control {steps}: {phase}', flush=True)
                result = emb.step(Action(data=command))
                observation = result.observation
                last = dict(result.info)
                steps += 1
                trace.write(json.dumps(dict(step=steps, phase=phase, action=command.tolist(),
                    metrics=last, diagnostics=diagnostics))+'\n')
                video.stdin.write(frame().tobytes())
                if steps % 5 == 0:
                    snapshot(phase)
                if steps == 100:
                    import warp as wp
                    materials = {}
                    for side, arm in arms.items():
                        values = wp.to_torch(arm.root_view.get_material_properties()).cpu().numpy().reshape(-1, 3)
                        materials[side] = int(np.count_nonzero(np.isclose(values[:, 1], 1.)))
                        if materials[side] < 8:
                            raise RuntimeError(f'{side} native pad friction inactive')
                    receipt['native_pad_shape_counts'] = materials
                if steps % 20 == 0:
                    trace.flush()
                    receipt.update(steps=steps, phase=phase, final_metrics=last)
                    save_receipt()
                    if policy is not None:
                        (out/'dream-transcript.json').write_text(json.dumps(policy.transcript(), indent=2)+'\n')
                    print(f'Control {steps}: score {last["score"]}', flush=True)
                previous_phase = phase
                if args.policy == 'dream' and result.terminated:
                    receipt['termination'] = 'success'
                    break
        if samples[-1]['step'] != steps:
            snapshot(previous_phase)
        np.savez_compressed(out/'cloth.npz', positions=np.asarray(nodes), rest=task.rest, faces=task.faces)
        receipt.update(status='completed', steps=steps, final_metrics=last, observations=samples)
        if args.policy == 'scripted' and last.get('score') != 1.:
            receipt['validation_error'] = 'Scripted demonstration did not achieve full score'
        cv2.imwrite(str(out/'final.png'), cv2.cvtColor(frame(), cv2.COLOR_RGB2BGR))
    except BaseException as error:
        receipt.update(status='interrupted' if isinstance(error, KeyboardInterrupt)
                       else 'error', steps=steps, final_metrics=last, error=traceback.format_exc())
        raise
    finally:
        if policy is not None:
            (out/'dream-transcript.json').write_text(json.dumps(policy.transcript(), indent=2)+'\n')
            try:
                policy.close()
            except Exception as error:
                receipt['policy_close_error'] = repr(error)
        if video is not None:
            video.stdin.close()
            if video.wait(timeout=60) != 0:
                receipt.update(status='error', video_error='ffmpeg failed')
        if emb is not None:
            emb.close()
        receipt.update(source_unchanged=source_hashes(folder) == receipt['source_sha256'],
                       wall_seconds=time.monotonic()-started)
        if not receipt['source_unchanged']:
            receipt.update(status='error', source_error='Task source changed during run')
        save_receipt()
        print(json.dumps({key: receipt.get(key) for key in
                         ('status', 'policy', 'steps', 'source_unchanged', 'final_metrics')}), flush=True)


if __name__ == '__main__':
    main()
