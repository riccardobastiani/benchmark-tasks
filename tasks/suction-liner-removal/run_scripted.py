"""Run and record the real robot-contact demonstration in native Isaac Lab."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--width',type=int,default=1920)
    parser.add_argument('--height',type=int,default=1080)
    parser.add_argument('--stop-after',choices=['connector','closure','vacuum','lift','transfer','deposit'])
    parser.add_argument('--max-steps',type=int,default=3600)
    parser.add_argument('--lid-only',action='store_true',help='Isolated bin mechanism test; not a full task run')
    parser.add_argument('--policy',choices=['scripted','dream'],default='scripted')
    parser.add_argument('--model',default='dreamlabs/dream-1-beta')
    args = parser.parse_args()
    if args.policy=='dream' and (args.lid_only or args.stop_after):
        parser.error('Dream evaluates the complete registered task, without scripted checkpoints')
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    folder = Path(__file__).parent.resolve()
    shutil.copytree(folder,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    if not (folder/'resist_isaac').is_dir():
        shutil.copytree(folder.parent.parent/'resist_isaac',output/'source'/'resist_isaac',
                        ignore=shutil.ignore_patterns('__pycache__'))
    config = json.loads((folder/'pilot.json').read_text())
    assert config['control_hz']==20 and args.max_steps==config['max_steps']
    receipt = dict(scope='Native bin-lid mechanism test' if args.lid_only else 'Actual native Isaac robot-contact execution',status='starting',
        hostname=socket.gethostname(),stop_after=args.stop_after,
        lid_only=args.lid_only,
        policy=args.policy,model=args.model if args.policy=='dream' else None,max_steps=args.max_steps,
        resolution=[args.width,args.height],control_hz=20,seed=0,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version',
                                     '--format=csv,noheader'],text=True),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=folder,text=True).strip(),
        source_sha256={str(p.relative_to(output/'source')):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (output/'source').rglob('*') if p.is_file()},
        versions={name:importlib.metadata.version(name) for name in
                  ('isaacsim','isaaclab','isaaclab-arena','yam-arena')})
    started = time.monotonic()
    emb,task,policy = None,None,None
    videos = {}
    steps,last,phase = 0,{},''
    try:
        import cv2
        import numpy as np
        from inspect_robots import Scene
        from inspect_robots.types import Action
        from resist_isaac.runtime import ResistanceEmbodiment,_task_module
        from resist_isaac.state import tensor
        from yam_arena.runtime import tcp_world
        emb = ResistanceEmbodiment(task=str(folder),condition='recoverable',device='cpu',
            render_width=args.width,render_height=args.height,cam_width=320,cam_height=180)
        task = emb._mechanism
        task.set_evidence_directory(output)
        instruction = config['instruction']
        scene = Scene(id='suction-liner-removal',instruction=instruction)
        observation = emb.reset(scene,seed=0)
        task.begin_episode()
        cameras = ('top_cam','close_cam','overview_cam')
        receipt.update(physics_dt_s=emb._base.physics_dt,control_dt_s=emb._base.step_dt,
                       native_device=str(emb._base.device),instruction=instruction)

        def frames():
            return {name:tensor(emb._base.scene[name].data.output['rgb'])[0,:,:,:3].cpu().numpy()
                    for name in cameras}

        def states():
            return {name+'/'+part:[x.tolist() for x in task.native(name,part)]
                    for name,part in task.views}

        for name,frame in frames().items():
            cv2.imwrite(str(output/f'setup-{name}.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            videos[name] = subprocess.Popen(['ffmpeg','-nostdin','-loglevel','error',
                '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{args.width}x{args.height}',
                '-framerate','20','-i','pipe:0','-an','-c:v','libx264','-threads','2',
                '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',
                str(output/f'{name}.mp4')],stdin=subprocess.PIPE)
            videos[name].stdin.write(np.ascontiguousarray(frame).tobytes())
        receipt['initial_state'] = states()
        if args.policy=='dream':
            from inspect_robots_dream import DreamPolicy
            from inspect_robots.cli import _build_guardrails
            policy = DreamPolicy(model=args.model,record=False,timeout_s=300.)
            policy.bind(emb.info)
            policy.reset(scene)
            approver,active,notices = _build_guardrails(emb.info.action_space,None,emb)
            receipt.update(guardrails=active,guardrail_notices=notices,
                controller_scope='Hosted Dream-1 camera/state policy; stock Inspect Robots guards; no expert actions',
                policy_versions={name:importlib.metadata.version(name) for name in
                    ('inspect-robots','inspect-robots-dream','dreamlabs-client')})
            guard_store = {}
            def dream_actions():
                for tick in range(args.max_steps):
                    if tick%20==0:
                        print(f'Dream control {tick}: '+json.dumps(last),flush=True)
                    sent = time.monotonic()
                    raw = policy.act(observation).actions[0]
                    approved = approver.review(raw,guard_store)
                    values = np.asarray(approved.data)
                    if values.shape!=(14,) or not np.isfinite(values).all():
                        raise RuntimeError('Invalid model action after stock guards')
                    yield values.copy(),'dream',dict(raw_action=np.asarray(raw.data).tolist(),
                        approval_meta=dict(approved.meta),policy_wall_seconds=time.monotonic()-sent)
            actions = dream_actions()
        else:
            module,_ = _task_module(folder)
            spec_name = module.__name__+'.interaction'
            spec = importlib.util.spec_from_file_location(spec_name,folder/'interaction.py')
            controller = importlib.util.module_from_spec(spec)
            sys.modules[spec_name] = controller
            spec.loader.exec_module(controller)
            actions = controller.actions(task,emb._base,args.stop_after,args.max_steps,args.lid_only)
        with (output/'trace.jsonl').open('w') as trace:
            for action,label,diagnostics in actions:
                if label!=phase:
                    phase = label
                    print(f'Control {steps}: {phase}',flush=True)
                result = emb.step(Action(data=action))
                observation = result.observation
                steps += 1
                last = dict(result.info)
                trace.write(json.dumps(dict(step=steps,phase=phase,action=action.tolist(),
                    metrics=last,diagnostics=diagnostics,objects=states(),
                    tcp={side:tcp_world(emb._base.scene[side+'_robot']).tolist() for side in ('left','right')}))+'\n')
                if steps%20==0:
                    trace.flush()
                for name,frame in frames().items():
                    videos[name].stdin.write(np.ascontiguousarray(frame).tobytes())
                if policy is not None and (steps%20==0 or result.terminated or last.get('unsafe_event')):
                    (output/'dream-transcript.json').write_text(json.dumps(policy.transcript(),indent=2)+'\n')
                if policy is not None and (result.terminated or last.get('unsafe_event')):
                    receipt['termination'] = 'success' if result.terminated else task.unsafe_reason
                    break
        last = task.evaluate(emb._base)
        if args.policy=='scripted' and not args.stop_after and not args.lid_only and last['score']!=1.:
            raise RuntimeError('Final native endpoint not reached')
        receipt.update(status='completed_component' if args.lid_only else ('completed_checkpoint' if args.stop_after else 'completed'),final_metrics=last,
                       final_state=states(),steps=steps)
    except Exception:
        receipt.update(status='error',error=traceback.format_exc(),steps=steps,phase=phase,final_metrics=last)
        if task is not None:
            receipt['unsafe_reason'] = task.unsafe_reason
        raise
    finally:
        if policy is not None:
            (output/'dream-transcript.json').write_text(json.dumps(policy.transcript(),indent=2)+'\n')
            try:
                policy.close()
            except Exception as error:
                receipt['policy_close_error']=repr(error)
        if emb is not None and getattr(emb,'_base',None) is not None:
            try:
                import cv2
                from resist_isaac.state import tensor
                for name in ('top_cam','close_cam','overview_cam'):
                    frame=tensor(emb._base.scene[name].data.output['rgb'])[0,:,:,:3].cpu().numpy()
                    cv2.imwrite(str(output/f'final-{name}.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            except Exception as error:
                receipt['final_frame_error']=repr(error)
        for video in videos.values():
            video.stdin.close()
            if video.wait(timeout=60)!=0:
                receipt['video_error']='ffmpeg failed'
        if task is not None:
            task.close()
        if emb is not None:
            emb.close()
        receipt['wall_seconds']=time.monotonic()-started
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(status=receipt['status'],steps=steps,phase=phase,metrics=last)),flush=True)


if __name__=='__main__':
    main()
