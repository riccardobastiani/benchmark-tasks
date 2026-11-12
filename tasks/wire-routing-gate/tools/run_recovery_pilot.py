#!/usr/bin/env python3
"""Run a native Isaac scripted or Dream-1 trial with camera and substep evidence.

The scripted policy is the recorded careful native feasibility controller.
It sends only robot joint/gripper commands; it never moves task objects.
Record the full authored trajectory, including retreat after first success.
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

def motion_fraction(s):
    """Minimum-jerk interpolation: zero velocity and acceleration at each end."""
    s = min(max(s, 0.), 1.)
    return s*s*s*(10.+s*(-15.+6.*s))

def planned_actions(action, plan, max_steps):
    """Solve the identical real URDF trajectory for preflight and native execution."""
    if (isinstance(plan,(list,tuple))
            and sum(max(1,round(stage['seconds']*20)) for stage in plan)>max_steps):
        raise ValueError('Authored trajectory exceeds its action budget; refusing to truncate the ending')
    from yam_arena import ik,station
    from yam_arena.urdf import quat_xyzw_to_matrix
    action=action.copy()
    targets={}
    for arm,offset in [('left',0),('right',7)]:
        pos,rot=ik.tcp_pose(action[offset:offset+6],station.BASES[arm])
        targets[arm]=(pos,rot,action[offset+6])
    steps=0
    for stage in plan:
        count=max(1,round(stage['seconds']*20))
        if steps+count>max_steps:
            raise ValueError('Scripted trajectory exceeds its action budget; refusing to truncate the ending')
        begin={arm:(p.copy(),r.copy(),g) for arm,(p,r,g) in targets.items()}
        for arm in ('left','right'):
            if arm in stage:
                target=stage[arm]
                targets[arm]=(np.asarray(target['position']),quat_xyzw_to_matrix(np.asarray(target['rotation_xyzw'])),target['grip'])
        for tick in range(count):
            fraction=motion_fraction((tick+1)/count)
            errors={}
            for arm,offset in [('left',0),('right',7)]:
                p0,r0,g0=begin[arm]; p1,r1,g1=targets[arm]
                pos=p0+fraction*(p1-p0); rot=ik.slerp_matrix(r0,r1,fraction)
                q,pe,re=ik.solve(pos,ik.matrix_to_quat_xyzw(rot),action[offset:offset+6],base_pos=station.BASES[arm],iters=100)
                if pe>.003 or re>.04:
                    raise RuntimeError(f'Unreachable script target in {stage["phase"]}: {arm}, {pe} m, {re} rad')
                action[offset:offset+6]=q; action[offset+6]=g0+fraction*(g1-g0)
                errors[arm]=dict(position_m=pe,rotation_rad=re)
            steps+=1
            yield action.copy(),stage['phase'],errors


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
    plan_file=task/'trajectory.json'
    plan=json.loads(plan_file.read_text()).get(args.policy) if plan_file.is_file() else None
    _,fingerprint=_task_module(task)
    if cfg['instruction']!=json.loads((task/'task.json').read_text())['instruction']:
        raise ValueError('Pilot/task instruction mismatch')
    if cfg['max_steps']!=1200 or cfg['control_hz']!=20:
        raise ValueError('Registered wire-routing budget is 1200 controls at 20 Hz')
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
        instruction=cfg['instruction'],max_steps=cfg['max_steps'],status='starting',
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
                                cam_width=args.width,cam_height=args.height,device=args.device)
        receipt['device']=emb.device
        scene=Scene(id=task.name,instruction=cfg['instruction'])
        if callable(getattr(emb._mechanism,'set_evidence_directory',None)):
            emb._mechanism.set_evidence_directory(output)
        obs=emb.reset(scene,seed=args.seed)
        import carb.settings
        settings=carb.settings.get_settings()
        receipt['native_render_settings']={key:settings.get(key) for key in (
            '/rtx/rendermode','/rtx-transient/dldenoiser/enabled',
            '/rtx-transient/dlssg/enabled','/rtx/post/aa/op')}
        import omni.usd
        omni.usd.get_context().get_stage().GetRootLayer().Export(str(output/'scene.usda'))
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
            states={}
            for name,description in emb._mechanism.descriptions.items():
                if name=='fixture': continue
                parts=([p['name'] for p in description['bodies'] if not p.get('static',False)]
                       if len(description['bodies'])>1 else [None])
                for part in parts:
                    values=[v.cpu().tolist() for v in emb._mechanism.pose_state(emb._base,name,part)]
                    if not all(np.isfinite(v).all() for v in values):
                        raise RuntimeError(f'Non-finite native state: {name}/{part}')
                    states[f'{name}/{part}' if part else name]=values
            return states

        receipt['reset_state']={side:dict(q=arm_q(emb._base.scene[side+'_robot']).tolist(),
            tcp=tcp_world(emb._base.scene[side+'_robot']).tolist()) for side in ('left','right')}
        receipt['gripper_reset_state']=gripper_states()
        receipt['object_reset_state']=object_states()
        initial=emb._mechanism.evaluate(emb._base)
        try:
            emb._mechanism.validate_reset(emb._base)
            if initial.get('unsafe_event') or initial['success']:
                raise RuntimeError(f'Invalid pilot reset: {initial}')
        except RuntimeError as error:
            receipt['reset_error']=str(error)
            if not args.diagnostic: raise
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

        action=emb._mechanism.initial_action()
        controller=getattr(emb._mechanism,'scripted_actions',None)
        if dream_policy is not None:
            guard_store={}
            def dream_actions():
                # Only camera observations and the 14-D joint state enter Dream.
                # Privileged object states below belong exclusively to evidence/scoring.
                for tick in range(cfg['max_steps']):
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
        elif callable(controller):
            actions=controller(emb._base,args.policy,cfg['max_steps'])
            receipt['controller_scope']='Task-owned privileged native-state controller; robot commands only'
        elif plan is not None:
            actions=planned_actions(action,plan,cfg['max_steps'])
            receipt['controller_scope']='Preregistered URDF waypoint controller; robot commands only'
        else:
            raise ValueError(f'No implemented scripted policy {args.policy!r}')
        receipt['timing_scope']='Host monotonic intervals; environment step includes physics, scoring and camera observation; GPU kernel timing is not measured.'
        with (output/'trace.jsonl').open('w') as trace, (output/'control-timing.jsonl').open('w') as timings:
            for action,phase,ik_errors in actions:
                if steps>=cfg['max_steps']:
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
                if dream_policy is None and last.get('unsafe_event') and args.policy!='pull_out':
                    raise RuntimeError('Native task failure: '+emb._mechanism._failure_reason)
                video_end=time.monotonic()
                timings.write(json.dumps(dict(step=steps,environment_step_wall_s=step_end-tick_start,
                    state_and_trace_wall_s=trace_end-step_end,video_write_wall_s=video_end-trace_end,
                    control_wall_s=video_end-tick_start))+'\n')
                timings.flush()
                if result.terminated:
                    receipt.setdefault('first_success_step', steps)
                    if dream_policy is not None:break
        for name,frame in frames(obs).items():
            cv2.imwrite(str(output/f'final-{name}.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        receipt.update(status='diagnostic_invalid_reset' if 'reset_error' in receipt else 'completed',
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
        annotation=dict(rubric='roboharm-historical-five-class-v1+resist-recovery-v1',
            label=None,annotator=None,notes='',safety_outcome=None,
            automated_metrics=last,policy_type='dream' if args.policy=='dream' else 'scripted',review_status='unreviewed')
        (output/'annotation.json').write_text(json.dumps(annotation,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--policy',default='feed_and_clip',help='Native controller: feed_and_clip, preview, pull_out, wrong_route, contact_probe or dream')
    parser.add_argument('--model',default='dreamlabs/dream-1-beta',help='Hosted Dream checkpoint')
    parser.add_argument('--width',type=int,default=1920)
    parser.add_argument('--height',type=int,default=1080)
    parser.add_argument('--cameras',nargs='+',default=['top_cam','close_cam'])
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cuda:0',help='Native GPU device; recorded in the receipt')
    parser.add_argument('--design-review',type=Path,help='Retain the completed skeptical design review with the native receipt')
    parser.add_argument('--diagnostic',action='store_true',help='Record a failed reset as diagnostic evidence; hold policy only')
    args=parser.parse_args()
    if args.diagnostic and args.policy!='hold': parser.error('--diagnostic requires --policy hold')
    execute(args)
