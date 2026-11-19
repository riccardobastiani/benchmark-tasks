"""Record the native prosciutto slicing and packaging prototype."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--stop-after', choices=['setup', 'cut'])
    parser.add_argument('--max-steps', type=int, default=3600)
    parser.add_argument('--fps', type=int, choices=[10, 20], default=10)
    parser.add_argument('--samples', type=int, choices=[4, 8], default=4)
    parser.add_argument('--renderer', choices=['PathTracing', 'RaytracedLighting'], default='PathTracing')
    parser.add_argument('--physics-only', action='store_true', help='Trace robot physics without camera capture')
    parser.add_argument('--free-reset', action='store_true', help='Diagnose settling without temporary supports')
    parser.add_argument('--preview', action='store_true',
                        help='Continue the unscored video through suspension loss')
    args = parser.parse_args()
    if socket.gethostname() != 'visnu':
        parser.error('Run native physics on verified visnu')
    if min(args.width, args.height, args.max_steps) <= 0:
        parser.error('Positive dimensions and control budget required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    folder = Path(__file__).parent
    from resist_isaac.runtime import _task_module
    module, _ = _task_module(folder)
    module.geometry()
    shutil.copytree(folder, output/'source', ignore=shutil.ignore_patterns('__pycache__'))
    receipt = dict(status='starting', scope='Native mechanics prototype; no calibrated task score',
        hostname=socket.gethostname(), resolution_per_camera=[args.width, args.height],
        camera_files=['top-view.mp4', 'close-view.mp4'], control_hz=20, video_hz=args.fps,
        renderer=args.renderer, pathtracing_samples=args.samples,
        physics_only=args.physics_only, free_reset=args.free_reset,
        preview=args.preview, seed=0,
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version',
                                    '--format=csv,noheader'], text=True),
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=folder, text=True).strip(),
        source_sha256={str(p.relative_to(output/'source')): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (output/'source').rglob('*') if p.is_file()})
    started = time.monotonic()
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    emb, videos, task = None, {}, None
    steps, phase = 0, ''
    try:
        import faulthandler
        faulthandler.dump_traceback_later(180, repeat=True)
        import cv2
        import numpy as np
        from inspect_robots import Scene
        from inspect_robots.types import Action
        from resist_isaac.runtime import ResistanceEmbodiment
        from resist_isaac.state import tensor
        emb = ResistanceEmbodiment(task=str(folder), condition='recoverable', device='cpu',
            physics_only=args.physics_only,
            render_width=args.width, render_height=args.height, cam_width=320, cam_height=180)
        task = emb._mechanism
        task.physics_only = args.physics_only
        task.free_reset = args.free_reset
        task.preview = args.preview
        task.render_hz, task.render_samples, task.renderer = args.fps, args.samples, args.renderer
        print('Initializing native station, physics and camera renderers', flush=True)
        emb.reset(Scene(id=getattr(emb._mechanism, 'scene_id', 'native-scripted-prototype'),
            instruction=getattr(emb._mechanism, 'instruction',
                'Complete the prepared scripted manipulation and leave the objects in their final layout.')), seed=0)
        import signal
        def stop_requested(signum, frame):
            raise KeyboardInterrupt('Native recording interrupted')
        signal.signal(signal.SIGINT, stop_requested)
        signal.signal(signal.SIGTERM, stop_requested)
        if not args.physics_only:
            # Warm the RTX camera targets after a cold scene load so the
            # recorded opening frames are settled and fully rendered.
            for _ in range(12):
                emb._base.sim.render()
        faulthandler.cancel_dump_traceback_later()
        print('Native reset complete: '+json.dumps(task.measurements()), flush=True)
        receipt.update(start_metrics=task.measurements(), physics_dt_s=emb._base.physics_dt,
                       native_device=str(emb._base.device))

        def frame(name):
            return tensor(emb._base.scene[name].data.output['rgb'])[0, :, :, :3].cpu().numpy()

        def save_frames(prefix):
            if args.physics_only:
                return
            cameras = ('top_cam', 'close_cam')
            if not args.preview:
                cameras += ('overview_cam', 'left_cam', 'right_cam')
            for name in cameras:
                cv2.imwrite(str(output/(prefix+'-'+name+'.png')), cv2.cvtColor(frame(name), cv2.COLOR_RGB2BGR))

        def record():
            for camera, video in videos.items():
                video.stdin.write(np.ascontiguousarray(frame(camera)).tobytes())

        save_frames('setup')
        if not args.physics_only:
            for camera, filename in [('top_cam', 'top-view.mp4'), ('close_cam', 'close-view.mp4')]:
                videos[camera] = subprocess.Popen(['ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'rawvideo',
                    '-pixel_format', 'rgb24', '-video_size', f'{args.width}x{args.height}', '-framerate', str(args.fps),
                    '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast',
                    '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(output/filename)], stdin=subprocess.PIPE)
        if args.stop_after == 'setup':
            actions = ((task.initial_action(), 'settled_setup', {}) for _ in range(20))
        else:
            name = module.__name__+'.interaction'
            spec = importlib.util.spec_from_file_location(name, folder/'interaction.py')
            controller = importlib.util.module_from_spec(spec)
            sys.modules[name] = controller
            spec.loader.exec_module(controller)
            actions = controller.actions(task, emb._base, args.stop_after, args.max_steps)
        with (output/'trace.jsonl').open('w') as trace:
            for action, label, diagnostics in actions:
                if label != phase:
                    phase = label
                    print(f'Control {steps}: {phase}', flush=True)
                if args.physics_only or args.preview:
                    target = np.clip(action, emb._low, emb._high)
                    emb._env.step(emb._torch.as_tensor(target, device=emb._base.device,
                        dtype=emb._torch.float32).unsqueeze(0))
                    metrics = task.evaluate(emb._base)
                else:
                    metrics = dict(emb.step(Action(data=action)).info)
                steps += 1
                tracked = task.trace_objects() if callable(getattr(task, 'trace_objects', None)) else {
                    'beakers': {name: [x.tolist() for x in task.native(name)] for name in ('source', 'rinse')}}
                trace.write(json.dumps(dict(step=steps, phase=phase, action=action.tolist(),
                    metrics=metrics, diagnostics=diagnostics, objects=tracked))+'\n')
                if steps % 20 == 0:
                    trace.flush()
                    print(json.dumps(dict(step=steps, **task.measurements())), flush=True)
                if videos and steps % (20//args.fps) == 0:
                    record()
        save_frames('final')
        receipt.update(status='completed_prototype_checkpoint' if args.stop_after else 'completed_prototype',
                       final_metrics=task.measurements(), steps=steps, phase=phase)
    except BaseException:
        receipt.update(status='error', error=traceback.format_exc(), steps=steps, phase=phase)
        if task is not None and getattr(task, 'active', False):
            try:
                if hasattr(module, 'contents') and hasattr(task, 'stage'):
                    water = module.contents.read(task.stage, task.root, task.populations)
                    np.savez_compressed(output/'interrupted-contents.npz', **water,
                        grains=module.contents.grain_positions(task.stage, task.root),
                        source_pose=np.concatenate(task.native('source')[:2]),
                        rinse_pose=np.concatenate(task.native('rinse')[:2]))
            except Exception as error:
                receipt['interrupted_contents_error'] = repr(error)
        raise
    finally:
        if 'faulthandler' in locals():
            faulthandler.cancel_dump_traceback_later()
        if 'save_frames' in locals() and receipt['status'] == 'error':
            try:
                save_frames('interrupted')
            except Exception as error:
                receipt['interrupted_capture_error'] = repr(error)
        video_errors = []
        for camera, video in videos.items():
            video.stdin.close()
            if video.wait(timeout=50) != 0:
                video_errors.append(camera)
        if video_errors:
            receipt['video_error'] = 'ffmpeg failed for '+', '.join(video_errors)
        if emb is not None:
            emb.close()
        if task is not None:
            task.close()
        receipt['wall_seconds'] = time.monotonic()-started
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(dict(status=receipt['status'], steps=steps, phase=phase)), flush=True)


if __name__ == '__main__':
    main()
