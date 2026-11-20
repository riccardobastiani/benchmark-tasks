"""Native scripted carriage stroke, slice transfer and fold-over package closure."""

import math
import numpy as np
from .geometry import (CARRIAGE_START, CARRIAGE_CUT, ENVELOPE, HANDLE, POWER_OFF, POWER_ON,
                       side_frame, ready_pose)


def actions(task, base, stop_after=None, max_steps=2400):
    from yam_arena import ik, station
    from yam_arena.runtime import arm_q, tcp_world
    from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix
    from resist_isaac.state import tensor
    from yam_arena.constants import FINGER_JOINTS, FINGER_OPEN

    action = task.initial_action()
    steps = 0
    open_grip = .85
    flap_grip, handle_grip, slice_grip = .05, .18, .01
    position_bias = {side: ik.tcp_pose(action[i:i+6], station.BASES[side])[0]
                     - ik.tcp_pose(arm_q(base.scene[side+'_robot']), station.BASES[side])[0]
                     for side, i in [('left', 0), ('right', 7)]}
    rotation_bias = {side: (ik.tcp_pose(action[i:i+6], station.BASES[side])[1]
                            @ ik.tcp_pose(arm_q(base.scene[side+'_robot']), station.BASES[side])[1].T)
                     for side, i in [('left', 0), ('right', 7)]}

    def actual_pose(side):
        return ik.tcp_pose(arm_q(base.scene[side+'_robot']), station.BASES[side])

    def set_target(side, position, rotation, grip, phase):
        index = 0 if side == 'left' else 7
        p, r = actual_pose(side)
        position_bias[side] = np.clip(position_bias[side] + np.clip(.15*(np.asarray(position)-p), -.0008, .0008), -.035, .035)
        error = ik._rot_error(rotation, r)
        angle = np.linalg.norm(error)
        if angle > 1e-9:
            rotation_bias[side] = axis_angle_matrix(error/angle, min(.004, .12*angle)) @ rotation_bias[side]
        bias_angle = np.linalg.norm(ik._rot_error(rotation_bias[side], np.eye(3)))
        if bias_angle > .20:
            rotation_bias[side] = ik.slerp_matrix(np.eye(3), rotation_bias[side], .20/bias_angle)
        for pscale, rscale in ((1., 1.), (1., .5), (1., 0.), (.5, 0.), (0., 0.)):
            rb = ik.slerp_matrix(np.eye(3), rotation_bias[side], rscale)
            pb = pscale*position_bias[side]
            q, pe, re = ik.solve(np.asarray(position)+pb,
                ik.matrix_to_quat_xyzw(rb@rotation), action[index:index+6],
                base_pos=station.BASES[side], iters=100)
            if pe <= .003 and re <= .04:
                position_bias[side], rotation_bias[side] = pb, rb
                break
        else:
            raise RuntimeError(f'Unreachable {phase}: {side}, {pe:.4f} m, {re:.3f} rad')
        action[index:index+6] = q
        action[index+6] = grip

    def emit(phase, side=None, position=None, rotation=None, grip=None):
        nonlocal steps
        if steps >= max_steps:
            raise RuntimeError('Script exceeds its control budget')
        if side is not None:
            index = 0 if side == 'left' else 7
            if position is not None:
                set_target(side, position, rotation, grip, phase)
            elif grip is not None:
                action[index+6] = grip
        steps += 1
        if task.unsafe_reason:
            raise RuntimeError('Unsafe native sequence: '+task.unsafe_reason)
        diagnostics = {}
        if side is not None and position is not None:
            diagnostics = {'side': side, 'tcp_m': actual_pose(side)[0].tolist(),
                           'target_m': np.asarray(position).tolist()}
        yield action.copy(), phase, diagnostics

    def hold(phase, seconds):
        for _ in range(round(seconds*20)):
            yield from emit(phase)

    def move(phase, side, position, rotation, grip, seconds=1.5):
        begin, r0 = actual_pose(side)
        index = 0 if side == 'left' else 7
        g0 = action[index+6]
        count = max(1, round(seconds*20))
        for tick in range(count):
            s = ik.smoothstep((tick+1)/count)
            yield from emit(phase, side, begin+s*(np.asarray(position)-begin),
                            ik.slerp_matrix(r0, rotation, s), g0+s*(grip-g0))

    def align(phase, side, position, rotation, grip):
        # Give the YAM servo a short settling window before testing alignment.
        for _ in range(40):
            yield from emit(phase, side, position, rotation, grip)
        actual = tcp_world(base.scene[side+'_robot'])
        delta = np.asarray(position) - actual
        error = np.linalg.norm(delta)
        if error > .009:
            raise RuntimeError(f'Grasp alignment blocked in {phase}: {error:.4f} m; '
                               f'target={np.asarray(position).tolist()}, tcp={actual.tolist()}, '
                               f'target_minus_tcp={delta.tolist()}')

    def grasp(side, name, part, local_offset, rotation, grip, label, opening=open_grip,
              clearance=.060, approach_above=0.):
        p, q, _ = task.native(name, part)
        point = p + quat_xyzw_to_matrix(q) @ np.asarray(local_offset)
        goal = point - .006*rotation[:, 2]
        pregrasp = goal-clearance*rotation[:, 2]
        if approach_above:
            yield from move('above_'+label, side, pregrasp+[0., 0., approach_above], rotation, opening, 2.)
        yield from move('approach_'+label, side, pregrasp, rotation, opening, 1.6)
        yield from align('settle_approach_'+label, side, pregrasp, rotation, opening)
        yield from move('reach_'+label, side, goal, rotation, opening, 1.2)
        yield from align('align_'+label, side, goal, rotation, opening)
        yield from move('grasp_'+label, side, actual_pose(side)[0], rotation, grip, .8)
        yield from hold('settle_grasp_'+label, .35)
        robot = base.scene[side+'_robot']
        ids = robot.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        gap = -float(tensor(robot.data.joint_pos)[0, ids].sum().cpu())
        if gap < -2 * FINGER_OPEN * grip + .0015:
            raise RuntimeError(f'Empty native grasp {label}: gap={gap:.4f} m')

    def relative(side, name, part):
        p, q, _ = task.native(name, part)
        tcp, r = actual_pose(side)
        return r.T@(p-tcp), r.T@quat_xyzw_to_matrix(q)

    def move_object(phase, side, name, part, position, rotation, seconds, grip, transform):
        offset, object_in_tool = transform
        p0, q0, _ = task.native(name, part)
        r0 = quat_xyzw_to_matrix(q0)
        count = max(1, round(seconds*20))
        for tick in range(count):
            s = ik.smoothstep((tick+1)/count)
            p = p0+s*(np.asarray(position)-p0)
            r = ik.slerp_matrix(r0, rotation, s)
            tool = r@object_in_tool.T
            error = p-task.native(name, part)[0]
            if tick > 6 and np.linalg.norm(error) > .025:
                raise RuntimeError('Stopped after native payload tracking loss: '+phase)
            target = p-tool@offset+np.clip(.20*error, -.004, .004)
            yield from emit(phase, side, target, tool, grip)
        if np.linalg.norm(task.native(name, part)[0]-position) > .018:
            raise RuntimeError('Lost native payload: '+phase)

    def release(side, label):
        p, r = actual_pose(side)
        yield from move('release_'+label, side, p, r, 1., .5)
        yield from move('withdraw_'+label, side, p-.060*r[:, 2]+[0., 0., .020], r, 1., 1.)

    def press_switch(name, position, side='right'):
        rotation = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
        goal = np.asarray(position)+[0., 0., .017]
        raised = goal+[0., 0., .055]
        yield from move('approach_'+name, side, raised, rotation, .05, 1.2)
        yield from align('align_'+name, side, raised, rotation, .05)
        yield from move(name, side, goal, rotation, .05, 1.25)
        for _ in range(24):
            yield from emit('confirm_'+name, side, goal, rotation, .05)
        yield from move('withdraw_'+name, side, raised, rotation, .05, .8)

    def home(side):
        p, r = ready_pose(side)
        yield from move('park_'+side, side, p, r, 1., 1.6)

    yield from hold('starting_layout', .4)

    yield from press_switch('slicer_power_on', POWER_ON)
    if not task.slicer_running:
        raise RuntimeError('Robot did not depress the start button')

    a = math.radians(25)
    x, z = np.array([1., 0., 0.]), np.array([0., math.cos(a), -math.sin(a)])
    pinch_frame = np.column_stack((x, np.cross(z, x), z))
    yield from grasp('right', 'workpiece', 'carriage', HANDLE, pinch_frame, handle_grip, 'carriage_handle', .55)
    carriage_transform = relative('right', 'workpiece', 'carriage')
    yield from move_object('cutting_stroke', 'right', 'workpiece', 'carriage', CARRIAGE_CUT,
                           np.eye(3), 2.4, handle_grip, carriage_transform)
    yield from hold('settle_carriage_at_blade', .35)
    if not task.portion_cut:
        raise RuntimeError('The powered carriage did not complete the cut')
    yield from hold('slice_falls_to_catcher', .6)
    yield from move_object('return_carriage', 'right', 'workpiece', 'carriage', CARRIAGE_START,
                           np.eye(3), 2.2, handle_grip, carriage_transform)
    yield from release('right', 'carriage_handle')
    yield from press_switch('slicer_power_off', POWER_OFF)
    if not task.slicer_stopped:
        raise RuntimeError('Robot did not depress the stop button')
    if stop_after == 'cut':
        yield from hold('observe_cut_slice', 1.)
        return

    # Transfer the separated portion only after the cutting motion has stopped.
    yield from move('raise_right_clear_of_slicer', 'right', [.24, -.18, .985],
                    side_frame('right'), .65, 1.6)
    _, slice_quat, _ = task.native('workpiece', 'portion')
    normal = quat_xyzw_to_matrix(slice_quat) @ [0., -1., 0.]
    approach = np.array([1., .17, 0.])
    approach -= normal*np.dot(normal, approach)
    approach /= np.linalg.norm(approach)
    slice_frame = np.column_stack((normal, np.cross(approach, normal), approach))
    yield from grasp('right', 'workpiece', 'portion', [-.026, 0., .025],
                     slice_frame, slice_grip, label='prosciutto_slice', opening=.19,
                     clearance=.028, approach_above=.140)
    portion_transform = relative('right', 'workpiece', 'portion')
    portion_position = task.native('workpiece', 'portion')[0]
    yield from move_object('lift_slice_clear_of_catcher', 'right', 'workpiece', 'portion',
                           portion_position + [0., .020, .085], np.eye(3), 1.6, slice_grip, portion_transform)
    # The catcher and envelope are on the same side of the blade plane.
    flat = axis_angle_matrix([1., 0., 0.], -math.pi/2)
    yield from move_object('carry_slice_above_envelope', 'right', 'workpiece', 'portion',
                           ENVELOPE + [0., -.025, .185], flat, 3.0, slice_grip, portion_transform)
    yield from move_object('lower_slice_clear_of_flap_tab', 'right', 'workpiece', 'portion',
                           ENVELOPE + [.010, -.025, .103], flat, 1.6, slice_grip, portion_transform)
    target = ENVELOPE + [.010, 0., .103]
    release_tilt = axis_angle_matrix([1., 0., 0.], math.radians(-25))
    yield from move_object('place_slice_in_envelope', 'right', 'workpiece', 'portion', target,
                           release_tilt, 1.8, slice_grip, portion_transform)
    p, r = actual_pose('right')
    yield from move('release_prosciutto_slice', 'right', p, r, 1., .5)
    yield from move('clear_fingers_from_slice', 'right', p+[0., -.045, 0.], r, 1., .7)
    yield from hold('slice_settles_in_pocket', .75)
    if not task.measurements()['portion_in_envelope']:
        raise RuntimeError('Released slice did not settle in the envelope')
    yield from move('withdraw_prosciutto_slice', 'right', p+[-.020, -.045, .040], r, 1., 1.)
    yield from home('right')

    # Give the right arm the entire packaging corridor before the left arm folds.
    left_frame = side_frame('left')
    yield from grasp('left', 'envelope', 'flap', [-.006, 0., .124],
                     left_frame, flap_grip, 'envelope_flap', .48, clearance=.034)
    flap_transform = relative('left', 'envelope', 'flap')
    flap_position = task.native('envelope', 'flap')[0]
    # Lower the flap far enough for gravity to finish without overextending the wrist.
    closed = axis_angle_matrix([1., 0., 0.], math.radians(65))
    yield from move_object('fold_envelope_closed', 'left', 'envelope', 'flap', flap_position,
                           closed, 2.0, flap_grip, flap_transform)
    yield from hold('press_folded_flap', .4)
    p, r = actual_pose('left')
    yield from move('release_envelope_flap', 'left', p, r, .32, .55)
    yield from move('withdraw_envelope_flap', 'left', ENVELOPE+[0., .020, .188],
                    left_frame, .32, 1.6)
    yield from home('left')
    yield from hold('observe_closed_food_envelope', 1.5)
    m = task.measurements()
    if m['packed_still_s'] < 1. - 1e-9:
        raise RuntimeError(f'Final packed envelope not reached: {m}')
