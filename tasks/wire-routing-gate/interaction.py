"""Privileged wire expert; all post-reset actuation is robot targets only."""
import math
import numpy as np
from .geometry import TARGET, PLATE_Z


def actions(task, base, policy, max_steps):
    from tools.run_recovery_pilot import planned_actions, motion_fraction
    from yam_arena import ik, station
    from yam_arena.runtime import arm_q, tcp_world
    from yam_arena.urdf import quat_xyzw_to_matrix
    action = task.initial_action(); steps = 0
    if policy=='preview':
        for _ in range(60): yield action.copy(), 'scene_preview', {}
        return
    if policy not in ('feed_and_clip', 'pull_out', 'wrong_route', 'contact_probe'): raise ValueError(policy)

    def move(stages):
        nonlocal action, steps
        for target, phase, errors in planned_actions(action, stages, max_steps-steps):
            action = target; steps += 1
            yield target, phase, errors

    def point(index): return task.pose_state(base, 'wire', f'link{index:02d}')[0].cpu().numpy().copy()
    if policy=='pull_out':
        p=point(4);rotation=[0.,1.,0.,0.]
        def left_stage(phase,seconds,position,grip):
            return dict(phase=phase,seconds=seconds,left=dict(position=np.asarray(position).tolist(),
                rotation_xyzw=rotation,grip=grip))
        yield from move([
            left_stage('approach_upstream_wire',1.5,p+[0,0,.09],1.),
            left_stage('reach_upstream_wire',1.,p+[0,0,.0105],1.),
            left_stage('grasp_upstream_wire',.8,p+[0,0,.0105],0.),
            left_stage('raise_upstream_wire',1.,p+[0,0,.08],0.),
        ])
        rotation=[-.5,math.sqrt(.75),0.,0.]
        yield from move([left_stage('deliberately_remove_wire',3.,[.08,.74,.88],0.)])
        for _ in range(60): yield action.copy(),'observe_irreversible_failure',{}
        return
    if policy=='contact_probe':
        # Real collision control: verify both contact groups before claiming a
        # contact-free demo. This diagnostic is never a task demonstration.
        def probe(phase, seconds, **positions):
            return dict(phase=phase, seconds=seconds, **{
                side:dict(position=p, rotation_xyzw=[0,1,0,0] if side=='left' else [1,0,0,0], grip=.2)
                for side,p in positions.items()})
        yield from move([
            probe('probe_approach_post',2.,right=[.320,0.,PLATE_Z+.12]),
            probe('probe_touch_post',3.,right=[.320,0.,PLATE_Z+.040]),
            probe('probe_retract_from_post',2.,right=[.320,0.,PLATE_Z+.12]),
            probe('probe_approach_other_arm',3.,left=[.28,.08,.89],right=[.28,-.08,.89]),
            probe('probe_touch_other_arm',3.,left=[.28,.01,.89],right=[.28,-.01,.89]),
        ])
        return
    left = point(32); right = point(45)
    # Lean the wrists away from the shared workspace while the fingers hold
    # the same cable. TCP separation alone does not clear the link4 motors.
    angle = math.radians(15)/2
    left_rot = [0., math.cos(angle), -math.sin(angle), 0.]
    right_rot = [math.cos(angle), 0., 0., -math.sin(angle)]
    rotations = dict(left=left_rot, right=right_rot)
    points = dict(left=left, right=right)
    def stage(phase, seconds, positions, grip):
        return dict(phase=phase, seconds=seconds, **{side:dict(position=np.asarray(p).tolist(),
            rotation_xyzw=rotations[side], grip=grip) for side, p in positions.items()})
    def reorient(side, index, rotation, phase, seconds):
        offset=0 if side=='left' else 7
        tcp=tcp_world(base.scene[side+'_robot'])
        _,frame=ik.tcp_pose(arm_q(base.scene[side+'_robot']),station.BASES[side])
        anchor=frame.T@(point(index)-tcp)
        bias=ik.tcp_pose(action[offset:offset+6],station.BASES[side])[0]-tcp
        goal=point(index)-quat_xyzw_to_matrix(np.array(rotation))@anchor+bias
        rotations[side]=rotation
        yield from move([stage(phase,seconds,{side:goal},0.)])
    selected = points
    yield from move([stage('approach_wire',3.,{'right':right+[0,0,.10]},1.)])

    def align_open(side,index):
        nonlocal action,steps
        offset=0 if side=='left' else 7
        pickup_height=.0105 if side=='left' else .013
        command=ik.tcp_pose(action[offset:offset+6],station.BASES[side])[0]
        for _ in range(40):
            error=point(index)+[0,0,pickup_height]-tcp_world(base.scene[side+'_robot'])
            command+=np.clip(.20*error,-.001,.001)
            q,pe,re=ik.solve(command,rotations[side],arm_q(base.scene[side+'_robot']),
                             base_pos=station.BASES[side],iters=150)
            if pe>.003 or re>.04:raise RuntimeError('Pickup alignment IK failed: '+side)
            action[offset:offset+6]=q;action[offset+6]=.2;steps+=1
            yield action.copy(),'centre_'+side+'_pickup',{side:dict(position_m=pe,rotation_rad=re,tcp_error_m=float(np.linalg.norm(error)))}
        if np.linalg.norm(point(index)+[0,0,pickup_height]-tcp_world(base.scene[side+'_robot']))>.004:
            raise RuntimeError('Physical pickup alignment blocked: '+side)

    # Secure the free end before touching the slack span. An open-loop,
    # simultaneous closure can push the terminal out of the other hand.
    yield from move([stage('reach_free_end',2.5,{'right':right+[0,0,.013]},.2)])
    yield from align_open('right',45)
    right_cmd=ik.tcp_pose(action[7:13],station.BASES['right'])[0]
    yield from move([stage('grasp_free_end',.8,{'right':right_cmd},0.),
                     stage('check_free_end_grasp',.6,{'right':right_cmd+[0,0,.010]},0.)])
    if not task.contacts['right'] or point(45)[2]-right[2]<.005:
        raise RuntimeError('Right free-end grasp was not physically retained')
    yield from move([stage('clear_space_for_feeding_hand',2.5,{'right':right_cmd+[.050,-.025,.060]},0.)])
    angle=math.radians(30)/2
    yield from reorient('right',45,[math.cos(angle),0.,0.,-math.sin(angle)],'clear_right_wrist',1.)
    left=point(32)
    yield from move([stage('approach_feeding_span',2.5,{'left':left+[0,0,.10]},.2)])
    left=point(32)
    yield from move([stage('reach_feeding_span',2.5,{'left':left+[0,0,.013]},.2)])
    left_cmd=ik.tcp_pose(action[:6],station.BASES['left'])[0]
    rotations['left'] = [0.,1.,0.,0.]
    yield from move([stage('square_left_grasp',1.,{'left':left_cmd},.2)])
    yield from align_open('left',32)
    left_cmd=ik.tcp_pose(action[:6],station.BASES['left'])[0]
    right_cmd=ik.tcp_pose(action[7:13],station.BASES['right'])[0]
    yield from move([stage('grasp_feeding_span',.8,{'left':left_cmd},0.),
                     stage('lift_wire',2.,dict(left=left_cmd+[0,0,.040],right=right_cmd+[0,0,.030]),0.)])
    if point(45)[2]-right[2]<.015:raise RuntimeError('Right wire section slipped from the physical grasp')
    if point(32)[2]-left[2]<.015:raise RuntimeError('Left wire section slipped from the physical grasp')
    yield from reorient('left',32,left_rot,'lean_wrists_apart',1.5)
    span = sum(2*part['shapes'][0]['size'][1] for part in task.descriptions['wire']['bodies'][33:46])+.0015
    final_wire = dict(left=np.array([TARGET[0], TARGET[1]+.035, PLATE_Z+.0042]),
                      right=np.array([TARGET[0], TARGET[1]+.035-span, PLATE_Z+.0042]))
    indices = dict(left=32,right=45)
    anchors = {}
    for side in selected:
        _, frame = ik.tcp_pose(arm_q(base.scene[side+'_robot']),station.BASES[side])
        anchors[side] = frame.T@(point(indices[side])-tcp_world(base.scene[side+'_robot']))
    correction = {side:np.zeros(3) for side in selected}

    def track(phase, duration, desired, left_rotation=None):
        nonlocal action, steps
        start = {s:point(indices[s]) for s in selected}
        start_frame = quat_xyzw_to_matrix(np.array(rotations['left']))
        for tick in range(1,round(duration*20)+1):
            fraction = motion_fraction(tick/round(duration*20)); errors = {}
            for side,offset in [('left',0),('right',7)]:
                if side not in selected: continue
                goal = start[side]+fraction*(desired[side]-start[side])
                error = goal-point(indices[side])
                limit=.025 if phase in ('press_wire_through_lips','hold_wire_seated') else .012
                correction[side] = np.clip(correction[side]+np.clip(.08*error,-.0005,.0005),-limit,limit)
                rotation = rotations[side]; frame = quat_xyzw_to_matrix(np.array(rotation))
                if side=='left' and left_rotation is not None:
                    frame=ik.slerp_matrix(start_frame,quat_xyzw_to_matrix(np.array(left_rotation)),fraction)
                    rotation=ik.matrix_to_quat_xyzw(frame)
                tcp = goal-frame@anchors[side]+correction[side]
                q,pe,re = ik.solve(tcp,rotation,action[offset:offset+6],
                                  base_pos=station.BASES[side],iters=150)
                if pe>.003 or re>.04: raise RuntimeError(f'Wire tracking IK failed: {side} {pe}, {re}')
                action[offset:offset+6]=q;action[offset+6]=0.
                errors[side]=dict(position_m=pe,rotation_rad=re,wire_error_m=float(np.linalg.norm(error)))
            steps+=1
            if steps>max_steps: raise RuntimeError('Wire controller exceeded action budget')
            yield action.copy(),phase,errors
        if left_rotation is not None: rotations['left']=left_rotation

    z=PLATE_Z+.035
    if policy=='wrong_route':
        yield from track('stage_in_front_of_gate',1.8,
            dict(left=np.array([.285,-.045,z]),right=np.array([.285,-.045-span,z])))
        yield from track('bypass_posts_on_wrong_side',2.,
            dict(left=np.array([.345,-.045,z]),right=np.array([.345,-.045-span,z])))
    else:
        yield from track('move_to_upstream_side',2.,{
            s:np.array([.270,point(indices[s])[1],z]) for s in selected})
        yield from track('stage_in_front_of_gate',4.5,
            dict(left=np.array([.270,.050+span,z]),right=np.array([.270,.050,z])))
        yield from track('feed_free_end_between_posts',2.5,
            dict(left=np.array([.270,.180,z]),right=np.array([.355,.050,z])))
        yield from track('position_feeding_hand',3.5,
            dict(left=np.array([.270,.050,z]),right=np.array([.360,-.075,z])))
        yield from track('feed_connected_span_through_gate',2.5,
            dict(left=np.array([.355,.050,z]),right=np.array([.370,-.105,z])))
    yield from track('align_span_over_clip',2.5,{s:p+[0,0,.031] for s,p in final_wire.items()})
    yield from track('press_wire_through_lips',5.,final_wire,left_rotation=[0.,1.,0.,0.])
    yield from track('hold_wire_seated',1.5,final_wire)
    if not task.evaluate(base)['target_occupied']:
        raise RuntimeError('Wire did not enter the target clip cavity')
    final = {side:ik.tcp_pose(action[(0 if side=='left' else 7):(6 if side=='left' else 13)],
                              station.BASES[side])[0] for side in ('left','right')}
    yield from move([
        stage('release_wire', 1.2, final, .2),
        stage('lift_clear_of_posts',2.,{s:p+[0,0,.10] for s,p in final.items()},.2),
        stage('open_hands_above_gate',.8,{s:p+[0,0,.10] for s,p in final.items()},1.),
    ])
    yield from move([
        stage('withdraw_both_hands', 3., {s:[.22,.26 if s=='left' else -.26,.96] for s in final}, 1.),
        stage('observe_retention', 3.5, {s:[.22,.26 if s=='left' else -.26,.96] for s in final}, 1.),
    ])
