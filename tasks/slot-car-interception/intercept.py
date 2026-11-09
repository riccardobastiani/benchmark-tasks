"""Privileged interception expert; emits fourteen robot targets only."""
import numpy as np

from .geometry import PADS, CAR_Z, WHEEL_RADIUS


def actions(task, base, policy, max_steps):
    from tools.run_recovery_pilot import planned_actions
    from yam_arena import ik, station
    from yam_arena.runtime import arm_q, tcp_world

    action=task.initial_action()
    steps=0

    def move(stages):
        nonlocal action, steps
        for target, phase, errors in planned_actions(action, stages, max_steps-steps):
            action=target;steps+=1
            yield target, phase, errors

    def command(side, position, grip, phase):
        nonlocal action, steps
        rotation=[0.,1.,0.,0.] if side=='left' else [1.,0.,0.,0.]
        q,pe,re=ik.solve(position,rotation,arm_q(base.scene[side+'_robot']),
                         base_pos=station.BASES[side],iters=150)
        if pe>.003 or re>.04:raise RuntimeError(f'Unreachable interception: {side} {position} {pe} {re}')
        offset=0 if side=='left' else 7
        action[offset:offset+6]=q;action[offset+6]=grip
        steps+=1
        if steps>max_steps:raise RuntimeError('Interception exceeded action budget')
        return action.copy(),phase,{side:dict(position_m=pe,rotation_rad=re)}

    def position_of(name):
        return task.pose_state(base,name,'chassis')[0].cpu().numpy().copy()

    if policy=='hold':
        for _ in range(200):
            yield action.copy(),'all_three_circulating',{}
        return
    if policy not in ('intercept','one_car'):
        raise ValueError(policy)
    yield from move([dict(phase='stage_both_arms_above_lane',seconds=2.,
        left=dict(position=[.399,.06,.86],rotation_xyzw=[0,1,0,0],grip=1.),
        right=dict(position=[.399,-.20,.86],rotation_xyzw=[1,0,0,0],grip=1.)),
        dict(phase='settle_open_grippers_over_lane',seconds=1.0,
        left=dict(position=[.399,.06,CAR_Z+.0185],rotation_xyzw=[0,1,0,0],grip=1.),
        right=dict(position=[.399,-.20,CAR_Z+.0185],rotation_xyzw=[1,0,0,0],grip=1.))])

    for side,name,trigger_y in [('left','yellow',.04),('right','blue',-.18)]:
        if policy=='one_car' and name=='blue':break
        while True:
            p=position_of(name)
            _,_,v=task.pose_state(base,name,'chassis');v=v.cpu().numpy()
            if abs(p[0]-.40)<.006 and trigger_y-.015<=p[1]<=trigger_y+.015 and v[1]>.12:
                break
            if task._unsafe:raise RuntimeError('Unsafe event before interception: '+task._failure_reason)
            steps+=1
            if steps>max_steps:raise RuntimeError('No interception window within budget')
            yield action.copy(),'wait_for_'+name,{}

        # The open gripper has settled at grasp height. Match longitudinal
        # motion before closing, avoiding a fast descent onto the moving roof.
        speed=float(np.clip(v[1],.15,.30))
        for tick in range(3):
            p=position_of(name)
            if abs(p[0]-.40)>.012:raise RuntimeError('Target left the pickup corridor')
            target=p+np.array([0.,speed*.08,.0185])
            yield command(side,target,1.,'match_velocity_'+name)
        for tick in range(1,6):
            p=position_of(name)
            if abs(p[0]-.40)>.012:raise RuntimeError('Target was pushed sideways')
            target=p+np.array([0.,speed*.08,.0185])
            yield command(side,target,1.-ik.smoothstep(tick/5),'close_on_moving_'+name)

        car_at_grasp=position_of(name)
        if not task._captured[name]:raise RuntimeError('No assigned fingertip grasp: '+name)
        # Preserve forward travel during guide extraction, then leave the road.
        start=ik.tcp_pose(action[(0 if side=='left' else 7):(6 if side=='left' else 13)],
                          station.BASES[side])[0]
        for tick in range(1,7):
            target=start+np.array([0.,speed*tick/20,.080*ik.smoothstep(tick/6)])
            yield command(side,target,0.,'lift_moving_'+name)
            if task._extracted[name]:break
        if not task._extracted[name]:
            raise RuntimeError('Car was not physically extracted while moving: '+name)

        # Once clear of the slot, stop following the lane and retract towards
        # the robot while raising the held car through a reachable posture.
        rotation=[0,1,0,0] if side=='left' else [1,0,0,0]
        offset=0 if side=='left' else 7
        safe=ik.tcp_pose(action[offset:offset+6],station.BASES[side])[0]
        safe[0]-=.05;safe[2]=.87
        yield from move([dict(phase='clear_lane_with_'+name,seconds=.55,
                             **{side:dict(position=safe.tolist(),rotation_xyzw=rotation,grip=0.)})])
        if position_of(name)[2]-car_at_grasp[2]<.04:
            raise RuntimeError('Car did not remain in the physical grasp: '+name)

        x,y=PADS[name]
        yield from move([
            dict(phase='carry_'+name+'_off_track',seconds=1.1,
                 **{side:dict(position=[x,y,.89],rotation_xyzw=rotation,grip=0.)}),
            dict(phase='lower_'+name+'_onto_pad',seconds=1.0,
                 **{side:dict(position=[x,y,.7515+WHEEL_RADIUS+.0185+.005],rotation_xyzw=rotation,grip=0.)}),
            dict(phase='release_'+name,seconds=.5,
                 **{side:dict(position=[x,y,.7515+WHEEL_RADIUS+.0185+.005],rotation_xyzw=rotation,grip=1.)}),
            dict(phase='withdraw_'+side,seconds=.7,
                 **{side:dict(position=[x,y,.91],rotation_xyzw=rotation,grip=1.)}),
        ])
    # Keep observing after the endpoint, including any later contact or lane loss.
    for _ in range(70):
        steps+=1
        yield action.copy(),'observe_red_continuing',{}
