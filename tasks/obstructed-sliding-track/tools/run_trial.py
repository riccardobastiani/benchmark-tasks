#!/usr/bin/env python3
"""Run a native sliding-track trial with camera, action, scorer and source evidence.

The original scripted policy uses the existing YAM waypoint implementation.
It sends only robot joint/gripper commands; it never moves task objects.
Dream receives only standard camera images, joint state and the instruction.
"""
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

def source_snapshot(task, output):
    """Retain new task files too: git diff does not contain untracked sources."""
    root=Path(__file__).resolve().parents[1]
    files=[root/'pyproject.toml',Path(__file__).resolve(),*sorted((root/'resist_isaac').glob('*.py'))]
    files.extend(sorted(p for p in task.rglob('*') if p.is_file() and '__pycache__' not in p.parts))
    hashes={}
    for path in files:
        relative=path.relative_to(root)
        destination=output/'source'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        hashes[relative.as_posix()]=hashlib.sha256(destination.read_bytes()).hexdigest()
    return hashes


def execute(args):
    from resist_isaac.runtime import ResistanceEmbodiment, _task_module

    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    task=args.task.resolve()
    source_root=Path(__file__).resolve().parents[1]
    cfg=json.loads((task/'pilot.json').read_text())
    module,fingerprint=_task_module(task)
    if cfg['instruction']!=module.INSTRUCTION:raise ValueError('Pilot/task instruction mismatch')
    budget=cfg['max_steps'] if args.max_steps is None else args.max_steps
    if not 1<=budget<=cfg['max_steps']:raise ValueError('Action budget exceeds declared task limit')
    source_hashes=source_snapshot(task,output)
    patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=source_root)
    (output/'source.patch').write_bytes(patch)
    receipt=dict(scope='Actual native Isaac pilot execution',hostname=socket.gethostname(),
        task=task.name,policy=args.policy,seed=args.seed,task_fingerprint=fingerprint,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_patch_sha256=hashlib.sha256(patch).hexdigest(),
        versions={name:importlib.metadata.version(name) for name in ('isaacsim','isaaclab','isaaclab-arena','yam-arena')},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv,noheader'],text=True),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=source_root).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],text=True,cwd=source_root),
        instruction=cfg['instruction'],max_steps=budget,declared_run_limit=cfg['max_steps'],status='starting',
        python_executable=sys.executable,source_files_sha256=source_hashes,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if args.design_review is not None:
        review=args.design_review.resolve()
        review_bytes=review.read_bytes()
        review_data=json.loads(review_bytes)
        if (not isinstance(review_data,dict) or review_data.get('task')!=task.name
                or review_data.get('instruction')!=cfg['instruction']):
            raise ValueError('Design review must match the exact task and registered instruction')
        (output/'design-review.json').write_bytes(review_bytes)
        receipt['design_review_sha256']=hashlib.sha256(review_bytes).hexdigest()
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    emb=None; dream_policy=None; videos={}; started=time.monotonic(); last={}; steps=0
    try:
        from inspect_robots import Scene
        from inspect_robots.types import Action
        import cv2
        emb=ResistanceEmbodiment(task=str(task),condition='recoverable',control_hz=20,
                                cam_width=320,cam_height=176,render_width=args.width,render_height=args.height,device=args.device)
        receipt['device']=emb.device
        scene=Scene(id=task.name,instruction=cfg['instruction'])
        if callable(getattr(emb._mechanism,'set_evidence_directory',None)):
            emb._mechanism.set_evidence_directory(output)
        obs=emb.reset(scene,seed=args.seed)
        receipt['physics_dt_s']=emb._base.physics_dt
        receipt['control_dt_s']=emb._base.step_dt
        def frames(observation):
            from resist_isaac.state import tensor
            result={}
            for name in args.cameras:
                raw=tensor(emb._base.scene[name].data.output['rgb'])
                if tuple(raw.shape[1:3])!=(args.height,args.width):
                    raise RuntimeError(f'Unexpected native camera shape: {name} {raw.shape}')
                result[name]=raw[0,:,:,:3].cpu().numpy()
            return result
        receipt['native_camera_shapes']={name:list(emb._base.scene[name].data.output['rgb'].shape) for name in args.cameras}
        for name,frame in frames(obs).items():
            cv2.imwrite(str(output/f'setup-{name}.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            h,w=frame.shape[:2]
            videos[name]=subprocess.Popen(['ffmpeg','-nostdin','-loglevel','error','-f','rawvideo',
                '-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate','20','-i','pipe:0',
                '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
                str(output/f'{name}.mp4')],stdin=subprocess.PIPE)
        for name,video in videos.items():video.stdin.write(np.ascontiguousarray(frames(obs)[name]).tobytes())
        from yam_arena.runtime import arm_q,tcp_world
        from yam_arena.constants import FINGER_JOINTS
        from resist_isaac.state import tensor
        receipt['native_devices']=dict(requested=str(emb.device),environment=str(emb._base.device),
            simulation_config=str(emb._base.sim.cfg.device),
            physics_manager=str(emb._base.sim.physics_manager.get_device()),
            robot_tensors={side:str(tensor(emb._base.scene[side+'_robot'].data.joint_pos).device)
                          for side in ('left','right')})
        actual_devices=[receipt['native_devices'][key] for key in
                        ('environment','simulation_config','physics_manager')]
        actual_devices.extend(receipt['native_devices']['robot_tensors'].values())
        if any(device!=str(emb.device) for device in actual_devices):
            raise RuntimeError(f'Requested and actual native devices differ: {receipt["native_devices"]}')
        finger_ids={side:emb._base.scene[side+'_robot'].find_joints(FINGER_JOINTS,preserve_order=True)[0]
                    for side in ('left','right')}

        def gripper_states():
            """Native 20 Hz finger state and command; this does not measure contact force."""
            states={}
            for side,ids in finger_ids.items():
                robot=emb._base.scene[side+'_robot']
                values={key:tensor(getattr(robot.data,field))[0,ids].cpu().tolist()
                        for key,field in (('position_m','joint_pos'),('velocity_m_s','joint_vel'),
                                          ('position_target_m','joint_pos_target'))}
                if not all(np.isfinite(value).all() for value in values.values()):
                    raise RuntimeError(f'Non-finite native finger state: {side}')
                states[side]=dict(joint_names=[robot.joint_names[i] for i in ids],**values)
            return states

        def object_states():
            from resist_isaac.state import pose_state, unit_quaternion
            states={}
            for name,parts in [('display',('Frame','Door')),('track_key',(None,))]:
                for part in parts:
                    values=pose_state(emb._base.scene[name],part)
                    if not all(np.isfinite(v.cpu().numpy()).all() for v in values):
                        raise RuntimeError(f'Invalid native object measurement: {name}/{part}')
                    if not unit_quaternion(values[1]):raise RuntimeError('Invalid native quaternion')
                    states[f'{name}/{part}' if part else name]=[v.cpu().tolist() for v in values]
            return states

        receipt['reset_state']={side:dict(q=arm_q(emb._base.scene[side+'_robot']).tolist(),
            tcp=tcp_world(emb._base.scene[side+'_robot']).tolist()) for side in ('left','right')}
        receipt['gripper_reset_state']=gripper_states()
        receipt['object_reset_state']=object_states()
        initial=emb._mechanism.evaluate(emb._base)
        if (initial.get('invalid_state') or initial['success'] or not initial['key_retained']
                or abs(initial['slide_m'])>.001):
            raise RuntimeError(f'Invalid scored reset: {initial}')
        receipt['initial_metrics']=initial
        if args.policy=='dream':
            from inspect_robots_dream import DreamPolicy
            from inspect_robots.cli import _build_guardrails
            dream_policy=DreamPolicy(model=args.model,record=False,timeout_s=300.)
            dream_policy.bind(emb.info)
            dream_policy.reset(scene)
            approver,active,notices=_build_guardrails(emb.info.action_space,None,emb)
            receipt.update(policy_type='dream',model=args.model,cloud_record=False,
                guardrails=active,guardrail_notices=notices,
                model_camera_names=['top_cam','left_cam','right_cam'],model_frame_size=[320,176],
                policy_versions={name:importlib.metadata.version(name) for name in
                    ('inspect-robots','inspect-robots-dream','dreamlabs-client')})
            print('Dream policy initialized; default guards: '+', '.join(active),flush=True)

        if dream_policy is not None:
            guard_store={}
            def dream_actions():
                # Only camera observations and the 14-D joint state enter Dream.
                # Privileged object states below belong exclusively to evidence/scoring.
                for tick in range(budget):
                    if tick==0 or (tick>=dream_policy.preroll_steps and
                            (tick-dream_policy.preroll_steps)%dream_policy.rows_per_request==0):
                        print(f'Dream policy act at control {tick+1}',flush=True)
                    sent=time.monotonic()
                    chunk=dream_policy.act(obs)
                    raw=chunk.actions[0]
                    if np.asarray(raw.data).shape!=(14,) or not np.isfinite(raw.data).all():
                        raise RuntimeError('Dream returned an invalid robot action')
                    approved=approver.review(raw,guard_store)
                    if not np.isfinite(approved.data).all():raise RuntimeError('Nonfinite approved action')
                    transcript=dream_policy.transcript()
                    (output/'dream-transcript.json').write_text(json.dumps(transcript,indent=2)+'\n')
                    yield np.asarray(approved.data).copy(),'dream',dict(
                        raw_action=np.asarray(raw.data).tolist(),approval_meta=dict(approved.meta),
                        policy_wall_time_s=time.monotonic()-sent,
                        requests_so_far=len(transcript['requests']))
            actions=dream_actions()
            receipt['controller_scope']='Actual hosted Dream-1 policy; standard bounds/delta guards; no expert refinement'
        else:
            from resist_isaac.scripted import resist_scripted_policy
            scripted=resist_scripted_policy(task=str(task),hold_s=2.)
            if args.policy=='scripted':
                scripted.bind(emb.info);scripted.reset(scene)
            ready=np.array([*arm_q(emb._base.scene['left_robot']),1.,
                            *arm_q(emb._base.scene['right_robot']),1.])
            def scripted_actions():
                for tick in range(budget):
                    command=scripted.act(obs).actions[0] if args.policy=='scripted' else Action(data=ready)
                    yield np.asarray(command.data).copy(),args.policy,{}
            actions=scripted_actions()
            receipt['controller_scope']='Original YAM/RESIST waypoint policy; robot commands only' if args.policy=='scripted' else 'Native reset/hold smoke test; no hosted inference'
        receipt['timing_scope']='Host monotonic intervals; environment step includes physics, scoring and camera observation; GPU kernel timing is not measured.'
        with (output/'trace.jsonl').open('w') as trace, (output/'control-timing.jsonl').open('w') as timings:
            for action,phase,ik_errors in actions:
                if steps>=budget:
                    raise RuntimeError('Scripted policy exceeded the registered action budget')
                tick_start=time.monotonic()
                result=emb.step(Action(data=action)); obs=result.observation
                step_end=time.monotonic()
                steps+=1; last=dict(result.info)
                trace.write(json.dumps(dict(step=steps,time_s=steps/20,phase=phase,
                    action=action.tolist(),metrics=last,
                    ik_errors={} if dream_policy is not None else ik_errors,
                    policy_metadata=ik_errors if dream_policy is not None else None,
                    q={side:arm_q(emb._base.scene[side+'_robot']).tolist() for side in ('left','right')},
                    tcp={side:tcp_world(emb._base.scene[side+'_robot']).tolist() for side in ('left','right')},
                    grippers=gripper_states(),
                    objects=object_states()))+'\n')
                trace.flush()
                trace_end=time.monotonic()
                for name,frame in frames(obs).items():videos[name].stdin.write(np.ascontiguousarray(frame).tobytes())
                if steps%20==0:print(f'step={steps} phase={phase} metrics={json.dumps(last)}',flush=True)
                video_end=time.monotonic()
                timings.write(json.dumps(dict(step=steps,environment_step_wall_s=step_end-tick_start,
                    state_and_trace_wall_s=trace_end-step_end,video_write_wall_s=video_end-trace_end,
                    control_wall_s=video_end-tick_start))+'\n')
                timings.flush()
                if result.terminated:
                    receipt.setdefault('first_success_step', steps)
                    break
        for name,frame in frames(obs).items():
            cv2.imwrite(str(output/f'final-{name}.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        receipt.update(status='completed',
                       final_metrics=last,steps=steps)
    except KeyboardInterrupt:
        receipt.update(status='interrupted',error=traceback.format_exc(),final_metrics=last,steps=steps)
        raise
    except Exception:
        receipt.update(status='error',error=traceback.format_exc(),final_metrics=last,steps=steps)
        raise
    finally:
        if dream_policy is not None:
            receipt['policy_transcript']=dream_policy.transcript()
            (output/'dream-transcript.json').write_text(json.dumps(receipt['policy_transcript'],indent=2)+'\n')
            try:dream_policy.close()
            except Exception as error:receipt['dream_close_error']=type(error).__name__+': '+str(error)
        for name,video in videos.items():
            try:
                video.stdin.close()
            except OSError as error:
                receipt.setdefault('video_errors',[]).append(dict(camera=name,stdin_error=str(error)))
            code=video.wait()
            if code: receipt.setdefault('video_errors',[]).append(code)
        receipt['elapsed_wall_s']=time.monotonic()-started
        try:
            current_hashes={name:hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                            for name in source_hashes}
            receipt['source_unchanged']=(_task_module(task)[1]==fingerprint and current_hashes==source_hashes)
        except Exception:
            receipt['source_unchanged']=False
            receipt['source_verification_error']=traceback.format_exc()
        if emb is not None:
            try:
                emb.close()
            except Exception:
                receipt['status']='error'
                receipt['close_error']=traceback.format_exc()
        receipt['evidence_sha256']={}
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in ('.jsonl','.mp4','.png'):
                with path.open('rb') as evidence:
                    receipt['evidence_sha256'][path.name]=hashlib.file_digest(evidence,'sha256').hexdigest()
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--policy',choices=('dream','scripted','hold'),default='dream')
    parser.add_argument('--max-steps',type=int,help='Optional lower budget for a smoke test; never exceeds pilot.json')
    parser.add_argument('--model',default='dreamlabs/dream-1-beta',help='Hosted Dream checkpoint')
    parser.add_argument('--width',type=int,default=1920)
    parser.add_argument('--height',type=int,default=1080)
    parser.add_argument('--cameras',nargs='+',default=['top_cam'])
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cuda:0',help='Native GPU device; recorded in the receipt')
    parser.add_argument('--design-review',type=Path,help='Retain the completed skeptical design review with the native receipt')
    args=parser.parse_args()
    execute(args)
