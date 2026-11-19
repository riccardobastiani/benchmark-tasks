"""Privileged native demonstration: actuate only the fourteen YAM robot controls."""
import math
import numpy as np

from .geometry import BIN_HANDLE, BIN_HEIGHT, BIN_HINGE, CANISTER, CANISTER_HEIGHT, CAP_SEAT, CATCH, CONTROL, HANDLES, LINER_DEPTH, RECEIVER


def actions(task, base, stop_after=None, max_steps=3600, lid_only=False):
    from yam_arena import ik, station
    from yam_arena.runtime import arm_q, tcp_world
    from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix
    action = task.initial_action()
    steps = 0

    def frame(tilt):
        a = math.radians(tilt)
        z,x = np.array([math.cos(a),0,-math.sin(a)]),np.array([0.,-1.,0.])
        return np.column_stack((x,np.cross(z,x),z))

    def command(side):
        index = 0 if side=='left' else 7
        return ik.tcp_pose(action[index:index+6],station.BASES[side])

    def actual_pose(side):
        return ik.tcp_pose(arm_q(base.scene[side+'_robot']),station.BASES[side])

    position_bias = {side:command(side)[0]-actual_pose(side)[0] for side in ('left','right')}
    rotation_bias = {side:command(side)[1]@actual_pose(side)[1].T for side in ('left','right')}

    def set_target(side,position,rotation,grip,phase):
        index = 0 if side=='left' else 7
        p,r = actual_pose(side)
        position_bias[side] = np.clip(position_bias[side]+np.clip(.15*(np.asarray(position)-p),-.0008,.0008),-.035,.035)
        error = ik._rot_error(rotation,r)
        angle = np.linalg.norm(error)
        if angle>1e-9:
            rotation_bias[side] = axis_angle_matrix(error/angle,min(.004,.12*angle))@rotation_bias[side]
        bias_angle = np.linalg.norm(ik._rot_error(rotation_bias[side],np.eye(3)))
        if bias_angle>.20:
            rotation_bias[side] = ik.slerp_matrix(np.eye(3),rotation_bias[side],.20/bias_angle)
        # Back off integral correction at the real joint limits; the measured grasp check stays strict.
        for position_scale,rotation_scale in ((1.,1.),(1.,.5),(1.,0.),(.5,0.),(0.,0.)):
            rb = ik.slerp_matrix(np.eye(3),rotation_bias[side],rotation_scale)
            pb = position_scale*position_bias[side]
            q,pe,re = ik.solve(np.asarray(position)+pb,
                ik.matrix_to_quat_xyzw(rb@rotation),action[index:index+6],
                base_pos=station.BASES[side],iters=100)
            if pe<=.003 and re<=.04:
                position_bias[side],rotation_bias[side] = pb,rb
                break
        else:
            raise RuntimeError(f'Unreachable {phase}: {side}, {pe:.4f} m, {re:.3f} rad')
        action[index:index+6] = q
        action[index+6] = grip

    def emit(phase, side=None, position=None, rotation=None, grip=None, diagnostics=None):
        nonlocal action,steps
        if steps>=max_steps:
            raise RuntimeError('Script exceeds its control budget')
        if side is not None:
            index = 0 if side=='left' else 7
            if position is not None:
                set_target(side,position,rotation,grip,phase)
            if grip is not None:
                action[index+6] = grip
        steps += 1
        if task.arm_contact_steps:
            raise RuntimeError('Stopped after native inter-arm contact')
        if task.unsafe_reason:
            raise RuntimeError('Unsafe native sequence: '+task.unsafe_reason)
        yield action.copy(),phase,diagnostics or {}

    def hold(phase, seconds):
        for _ in range(round(seconds*20)):
            yield from emit(phase)

    def move(phase, side, position, rotation, grip, seconds=2.):
        begin,r0 = actual_pose(side)
        index = 0 if side=='left' else 7
        g0 = action[index+6]
        count = max(1,round(seconds*20))
        for tick in range(count):
            s = ik.smoothstep((tick+1)/count)
            yield from emit(phase,side,begin+s*(np.asarray(position)-begin),
                            ik.slerp_matrix(r0,rotation,s),g0+s*(grip-g0))

    def align(phase, side, position, rotation, grip):
        for _ in range(35):
            error = np.asarray(position)-tcp_world(base.scene[side+'_robot'])
            yield from emit(phase,side,position,rotation,grip,dict(tcp_error_m=float(np.linalg.norm(error))))
        error = np.linalg.norm(np.asarray(position)-tcp_world(base.scene[side+'_robot']))
        if error>.006:
            raise RuntimeError(f'Native grasp alignment blocked in {phase}: {error:.4f} m')

    def grasp(side, name, part, offset, tilt=45, label=None):
        label = label or name+'_'+part
        if side=='right':
            yield from move('raise_right_above_bin',side,[.28,-.40,1.12],frame(0),.6,2.)
        p,q,_ = task.native(name,part)
        point = p+quat_xyzw_to_matrix(q)@np.asarray(offset)
        rotation = frame(tilt)
        goal = point-.006*rotation[:,2]
        if side=='right':
            yield from move('traverse_right_above_workspace',side,
                            goal-.070*rotation[:,2]+[0,0,.130],rotation,.50,2.)
        yield from move('approach_'+label,side,goal-.070*rotation[:,2],rotation,.50,2.)
        yield from align('settle_approach_'+label,side,goal-.070*rotation[:,2],rotation,.50)
        yield from move('reach_'+label,side,goal,rotation,.42,1.8)
        yield from align('align_'+label,side,goal,rotation,.42)
        yield from move('grasp_'+label,side,actual_pose(side)[0],rotation,0.,.9)
        yield from hold('settle_grasp_'+label,.4)
        from resist_isaac.state import tensor
        from yam_arena.constants import FINGER_JOINTS
        robot = base.scene[side+'_robot']
        ids = robot.find_joints(FINGER_JOINTS,preserve_order=True)[0]
        gap = -float(tensor(robot.data.joint_pos)[0,ids].sum().cpu())
        if not np.isfinite(gap) or gap<.002:
            raise RuntimeError(f'Empty native grasp: {label}, gap {gap:.4f} m')

    def grasp_transform(side,name,part):
        p,q,_ = task.native(name,part)
        tcp,r = ik.tcp_pose(arm_q(base.scene[side+'_robot']),station.BASES[side])
        return r.T@(p-tcp),r.T@quat_xyzw_to_matrix(q)

    def carry(phase,side,name,part,goal,seconds,relative=None,rotation=None):
        offset,object_in_tool = relative or grasp_transform(side,name,part)
        p0,q0,_ = task.native(name,part)
        r0 = quat_xyzw_to_matrix(q0)
        final = r0 if rotation is None else rotation
        count = max(1,round(seconds*20))
        for tick in range(count):
            fraction = ik.smoothstep((tick+1)/count)
            endpoint = np.asarray(goal() if callable(goal) else goal)
            desired = p0+fraction*(endpoint-p0)
            object_rotation = ik.slerp_matrix(r0,final,fraction)
            tool_rotation = object_rotation@object_in_tool.T
            actual = task.native(name,part)[0]
            error = desired-actual
            target = desired-tool_rotation@offset+np.clip(.20*error,-.004,.004)
            yield from emit(phase,side,target,tool_rotation,0.,dict(payload_error_m=float(np.linalg.norm(error))))
        if np.linalg.norm(task.native(name,part)[0]-endpoint)>.020:
            raise RuntimeError('Native payload did not follow grasp: '+phase)

    def release(side,label):
        p,r = actual_pose(side)
        yield from move('release_'+label,side,p,r,1.,.7)
        yield from move('clear_'+label,side,p-.070*r[:,2]+[0,0,.020],r,1.,1.5)

    def park(side):
        p,rot = ([.22,.26,.96],[0,1,0,0]) if side=='left' else ([.18,-.40,.96],[1,0,0,0])
        if side=='right':
            yield from move('clear_bin_before_park',side,[.28,-.40,1.12],frame(0),1.,2.)
        yield from move('park_'+side,side,p,quat_xyzw_to_matrix(np.asarray(rot)),1.,2.)

    def carry_both(phase,goal,seconds):
        relatives = {side:grasp_transform(side,'liner','lid_and_load') for side in ('left','right')}
        p0,q0,_ = task.native('liner','lid_and_load')
        initial_rotation = quat_xyzw_to_matrix(q0)
        count = round(seconds*20)
        for tick in range(count):
            fraction = ik.smoothstep((tick+1)/count)
            desired = p0+fraction*(np.asarray(goal)-p0)
            rotation = ik.slerp_matrix(initial_rotation,np.eye(3),fraction)
            error = desired-task.native('liner','lid_and_load')[0]
            for side,index in [('left',0),('right',7)]:
                offset,relative_rotation = relatives[side]
                tool_rotation = rotation@relative_rotation.T
                target = desired-tool_rotation@offset+np.clip(.20*error,-.005,.005)
                set_target(side,target,tool_rotation,0.,phase)
            yield from emit(phase,diagnostics=dict(payload_error_m=float(np.linalg.norm(error))))
        if np.linalg.norm(task.native('liner','lid_and_load')[0]-np.asarray(goal))>.020:
            raise RuntimeError('Bimanual payload did not follow native grasps: '+phase)

    def release_both():
        starts = {side:(*actual_pose(side),float(action[index+6]))
                  for side,index in [('left',0),('right',7)]}
        for tick in range(20):
            s = ik.smoothstep((tick+1)/20)
            for side,index in [('left',0),('right',7)]:
                action[index+6] = starts[side][2]+s*(1-starts[side][2])
            yield from emit('release_liner_inside_receiver')
        for side in ('left','right'):
            p,r,_ = starts[side]
            delta = -.040*r[:,2]+[0,.120 if side=='left' else -.120,.030]
            yield from move('clear_released_liner_'+side,side,p+delta,r,1.,1.5)

    def close_bin():
        # Pinch the side bar while the passive physical hinge turns.
        yield from grasp('right','waste_bin','lid',BIN_HANDLE,0,'bin_lid_side_bar')
        initial_angle = task.measurements()['bin_lid_angle_rad']
        for tick in range(180):
            angle = initial_angle*(1-ik.smoothstep((tick+1)/180))
            c,s = math.cos(angle),math.sin(angle)
            turn = np.array([[c,0,s],[0,1,0],[-s,0,c]])
            point = RECEIVER+BIN_HINGE+turn@BIN_HANDLE
            goal = point-.006*frame(0)[:,2]
            yield from emit('close_receiver_lid','right',goal,frame(0),0.,dict(lid_target_angle_rad=angle))
        yield from hold('settle_closed_lid',.7)
        yield from release('right','bin_lid')
        yield from park('right')
        yield from hold('finish_hands_clear',2.)

    if lid_only:
        yield from close_bin()
        if abs(task.measurements()['bin_lid_angle_rad'])>.0524:
            raise RuntimeError('Native lid mechanism test did not close the lid')
        return

    # Disconnect and account for the patient-side connector before closing the port.
    yield from grasp('left','connector','elbow',[.018,0,.009],45)
    relative = grasp_transform('left','connector','elbow')
    point = task.native('connector','elbow')[0]
    yield from carry('withdraw_connector','left','connector','elbow',point+[0,0,.060],2.,relative,np.eye(3))
    yield from carry('transport_connector','left','connector','elbow',CATCH+[-.035,0,.24],3.,relative,np.eye(3))
    yield from carry('lower_connector_to_catch_tray','left','connector','elbow',CATCH+[-.035,0,.026],3.,relative,np.eye(3))
    yield from release('left','connector')
    yield from park('left')
    if not task.measurements()['connector_removed']:
        raise RuntimeError('Connector is still on the liner')
    if stop_after=='connector': return

    # The plug remains free; its passive tether provides no robot attachment or seal latch.
    yield from grasp('left','liner','closure',[0,0,0],45,'closure')
    relative = grasp_transform('left','liner','closure')
    point = task.native('liner','closure')[0]
    yield from carry('lift_closure','left','liner','closure',point+[0,0,.045],1.8,relative,np.eye(3))
    def closure_target():
        p,q,_ = task.native('liner','lid_and_load')
        return p+quat_xyzw_to_matrix(q)@CAP_SEAT
    yield from carry('align_closure_above_port','left','liner','closure',lambda:closure_target()+[0,0,.025],2.5,relative,np.eye(3))
    yield from carry('shallow_closure','left','liner','closure',lambda:closure_target()+[0,0,.004],2.,relative,np.eye(3))
    yield from hold('verify_shallow_closure',.3)
    if task.measurements()['cap_seated']:
        raise RuntimeError('Shallow closure incorrectly measured as fully seated')
    # Lead with one edge of the rubber lip, then press the flange level.
    tipped = axis_angle_matrix([0.,1.,0.],math.radians(15))
    yield from carry('engage_closure_edge','left','liner','closure',
                     lambda:closure_target()+[0,0,.002],2.2,relative,tipped)
    yield from carry('seat_closure','left','liner','closure',closure_target,3.,relative,np.eye(3))
    yield from hold('verify_seated_closure',.6)
    yield from release('left','closure')
    yield from hold('verify_released_closure',.7)
    if not task.measurements()['cap_seated']:
        raise RuntimeError('Closure did not remain seated after release')
    yield from park('left')
    if stop_after=='closure': return

    yield from grasp('left','vacuum','lever',[.010,0,.012],60,'vacuum_switch')
    relative = grasp_transform('left','vacuum','lever')
    c,s = math.cos(1.10),math.sin(1.10)
    turn = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    yield from carry('switch_off_vacuum','left','vacuum','lever',CONTROL+[0,0,.018],3.,relative,turn)
    yield from release('left','vacuum_switch')
    yield from park('left')
    if task.measurements()['switch_angle_rad']<.95:
        raise RuntimeError('Vacuum control did not reach OFF')
    if stop_after=='vacuum': return

    # Both arms support the filled liner, with their pads on the leading ends of the tabs.
    yield from grasp('left','liner','lid_and_load',HANDLES['left']+[-.014,0,0],0,'left_lifting_tab')
    yield from grasp('right','liner','lid_and_load',HANDLES['right']+[-.014,0,0],0,'right_lifting_tab')
    lifted = task.native('liner','lid_and_load')[0].copy()
    lifted[2] = max(CANISTER[2]+CANISTER_HEIGHT-.004,RECEIVER[2]+BIN_HEIGHT)+LINER_DEPTH+.018
    yield from carry_both('extract_sealed_liner_with_both_arms',lifted,7.)
    if stop_after=='lift': return
    above = [RECEIVER[0],RECEIVER[1],lifted[2]]
    yield from carry_both('move_liner_above_receiver_with_both_arms',above,5.)
    if stop_after=='transfer': return
    # The liner body is inside the bin before release; the short remaining fall is contained.
    destination = RECEIVER+[0,0,BIN_HEIGHT+.042-HANDLES['left'][2]]
    yield from carry_both('lower_liner_into_receiver_with_both_arms',destination,6.)
    yield from hold('settle_liner_in_receiver',.7)
    yield from release_both()
    yield from park('left')
    yield from park('right')
    yield from hold('verify_released_liner',1.1)
    if not task.evaluate(base)['liner_inside']:
        raise RuntimeError('Liner is not fully inside the receiver')
    if stop_after=='deposit': return

    yield from close_bin()
    if task.evaluate(base)['score']!=1.:
        raise RuntimeError('Script did not reach the physical endpoint: '+repr(task.evaluate(base)))
