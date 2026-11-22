"""Scripted hem unfolding using ordinary YAM arm and finger targets."""

import numpy as np


def hem_actions(ready, variant='edge'):
    from yam_arena import ik, station
    from yam_arena.urdf import axis_angle_matrix

    if variant not in ('edge', 'deep', 'arc', 'roll', 'corner', 'corner_low'):
        raise ValueError(f'Unknown hem trajectory {variant!r}')
    deep = variant == 'deep'
    arc = variant in ('arc', 'roll')
    corner = variant in ('corner', 'corner_low')
    side_names = ('left',) if corner else ('left', 'right')
    offsets = (0,) if corner else (0, 7)
    bases = [station.BASES[side] for side in side_names]
    grasp, rotation, approach = [], [], []
    for sign, x in ((1., .273), (-1., .266)):
        forward = np.array([.476314, -sign*.723273, -.5])
        forward /= np.linalg.norm(forward)
        vertical = np.array([0., 0., 1.])
        closing = sign * (vertical - vertical.dot(forward)*forward)
        closing /= np.linalg.norm(closing)
        rotation.append(np.column_stack((closing, np.cross(forward, closing), forward)))
        grasp.append(np.array([x, sign*(.234 if deep else .246), .781]))
        approach.append(forward)
    grasp, rotation, approach = map(np.array, (grasp, rotation, approach))
    align = grasp - .038*approach
    away = grasp - .048*approach + [0., 0., .045]
    lift = grasp + [0., 0., .04]
    turn = np.array([[.174, .246, .870], [.174, -.246, .870]])
    pull = np.array([[.081, .246, .804], [.081, -.246, .804]])
    place = np.array([[.072, .246, .779], [.072, -.246, .779]])
    if deep:
        lift = grasp + [0., 0., .03]
        turn = np.array([[.216, .237, .826], [.216, -.237, .826]])
        pull = np.array([[.145, .239, .821], [.145, -.239, .821]])
        place = np.array([[.164, .239, .779], [.164, -.239, .779]])
    if arc:
        lift = grasp + [0., 0., .036]
        turn = np.array([[.237, .246, .831], [.237, -.246, .831]])
        pull = np.array([[.204, .246, .811], [.204, -.246, .811]])
        place = np.array([[.170, .246, .779], [.170, -.246, .779]])
    clear = np.array([[.166, .246, .866], [.166, -.246, .866]]) if arc else pull
    retreat = away if arc else turn
    stages = (
        ('hem_approach', away, .22, 35, True), ('hem_align', align, .22, 15, False),
        ('hem_insert', grasp, .22, 20, False), ('hem_pinch', grasp, 0., 35 if deep else 25, False),
        ('hem_lift', lift, 0., 25, False), ('hem_turn', turn, 0., 25, False),
        ('hem_pull', pull, 0., 40 if deep else 30, False), ('hem_place', place, 0., 20, False),
        ('hem_release', place, .22, 15, False), ('hem_clear', clear, .22, 20 if arc else 15, False),
        ('hem_retreat', retreat, .22, 25 if arc else 20, False),
    )
    tilts = {'hem_place': -.16, 'hem_release': -.16} if arc else {}
    if variant == 'roll':
        def pair(x, z):
            return np.array([[x, .246, z], [x, -.246, z]])

        stages = (*stages[:5],
                  ('hem_turn', turn, 0., 25, False),
                  ('hem_pull', pair(.204, .820), 0., 30, False),
                  ('hem_guide', pair(.185, .792), 0., 25, False),
                  ('hem_place', place, 0., 20, False),
                  ('hem_release', place, .22, 15, False),
                  ('hem_clear', pair(.177, .833), .22, 25, False),
                  ('hem_clear_up', pair(.177, .861), .22, 25, False),
                  ('hem_retreat', away, .22, 25, False))
        tilts = {'hem_turn': .35, 'hem_pull': 1.30, 'hem_guide': 1.80,
                 'hem_place': 1.60, 'hem_release': 1.60, 'hem_clear': 1.25}
    if corner:
        point = np.array([[.242, .273, .786]])
        align = point - .037*approach[:1]
        away = align + [0., 0., .045]
        lift = np.array([[.225, .278, .812]])
        high = np.array([[.185, .278, .833]])
        pull = np.array([[.143, .265, .808]])
        place = np.array([[.106, .248, .777]])
        clear_after_peel = high
        if variant == 'corner_low':
            # Roll just above the supported layer before drawing the released edge.
            lift = np.array([[.224, .277, .801]])
            high = np.array([[.196, .274, .810]])
            pull = np.array([[.151, .262, .795]])
            place = np.array([[.111, .248, .774]])
            clear_after_peel = np.array([[.182, .267, .856]])
        stages = (
            ('corner_approach', away, .22, 35, True), ('corner_align', align, .22, 15, False),
            ('corner_insert', point, .22, 20, False), ('corner_pinch', point, 0., 25, False),
            ('corner_lift', lift, 0., 25, False), ('corner_arc', high, 0., 25, False),
            ('corner_peel', pull, 0., 30, False), ('corner_place', place, 0., 25, False),
            ('corner_release', place, .22, 15, False), ('corner_clear', pull, .22, 20, False),
            ('corner_retreat', clear_after_peel, .22, 20, False),
        )
    motion = [(phase, target, grip, controls, joint_entry,
               axis_angle_matrix(np.array([0., 1., 0.]), tilts.get(phase, 0.)) @ rotation)
              for phase, target, grip, controls, joint_entry in stages]
    yield from _move(ready, side_names, offsets, bases, motion)


def _move(ready, side_names, offsets, bases, motion):
    from yam_arena import ik

    command = ready.copy()
    for phase, target, grip, controls, joint_entry, r1 in motion:
        q0 = [command[i:i+6].copy() for i in offsets]
        p0, r0 = zip(*(ik.tcp_pose(q, base) for q, base in zip(q0, bases, strict=True)), strict=True)
        grip0 = command[[i+6 for i in offsets]].copy()
        grip1 = np.broadcast_to(grip, grip0.shape)

        def solve(side, xyz, rot, initial):
            q, pe, re = ik.solve(xyz, ik.matrix_to_quat_xyzw(rot), initial, base_pos=bases[side], iters=160)
            if pe > .002 or re > .03:
                raise RuntimeError(f'Unreachable {side_names[side]} {phase}: {pe:.4f} m, {re:.4f} rad')
            return q

        if joint_entry:
            q_target = [solve(side, target[side], r1[side], q0[side]) for side in range(len(offsets))]
        for t in range(1, controls+1):
            a = ik.smoothstep(t/controls)
            for side, offset in enumerate(offsets):
                if joint_entry:
                    command[offset:offset+6] = q0[side] + a*(q_target[side] - q0[side])
                else:
                    xyz = p0[side] + a*(target[side]-p0[side])
                    rot = ik.slerp_matrix(r0[side], r1[side], a)
                    command[offset:offset+6] = solve(side, xyz, rot, command[offset:offset+6])
                command[offset+6] = grip0[side] + a*(grip1[side]-grip0[side])
            yield phase, command.copy()
    for _ in range(100):
        yield 'released_hold', command.copy()


def roll_push_actions(ready):
    """Guide the diagonal flap with a closed left hand while the right holds the torso."""
    from yam_arena import station
    from yam_arena.urdf import axis_angle_matrix

    forward = np.array([.476314, -.723273, -.5])
    forward /= np.linalg.norm(forward)
    closing = np.array([0., 0., 1.]) - forward[2]*forward
    closing /= np.linalg.norm(closing)
    left_base = np.column_stack((closing, np.cross(forward, closing), forward))
    right_down = axis_angle_matrix(np.array([0., 0., 1.]), -1.58) @ np.diag([1., -1., -1.])
    approach = (.283, .275, .830)
    right_approach = (.360, -.140, .860)
    right_high = (.455, -.140, .825)
    anchor = (.455, -.140, .7693)
    lifted = (.165, .257, .851)
    positions = (
        ('push_approach', approach, .50, right_approach, 35, True),
        ('anchor_approach', approach, .50, right_high, 22, False),
        ('anchor_place', approach, .50, anchor, 25, False),
        ('push_align', (.263, .274, .783), 1.05, anchor, 25, False),
        ('push_engage', (.245, .273, .779), 1.40, anchor, 25, False),
        ('push_turn', (.224, .269, .777), 1.40, anchor, 25, False),
        ('push_draw', (.184, .261, .775), 1.40, anchor, 25, False),
        ('push_smooth', (.147, .252, .774), 1.40, anchor, 25, False),
        ('push_finish', (.133, .246, .778), 1.40, anchor, 25, False),
        ('push_clear', lifted, .50, anchor, 25, False),
        ('anchor_clear', lifted, .50, right_high, 25, False),
        ('anchor_retreat', lifted, .50, right_approach, 22, False),
    )
    motion = [(phase, np.array([left, right]), 0., controls, joint_entry,
               np.stack([axis_angle_matrix(np.array([0., 1., 0.]), tilt) @ left_base, right_down]))
              for phase, left, tilt, right, controls, joint_entry in positions]
    yield from _move(ready, ('left', 'right'), (0, 7), [station.BASES['left'], station.BASES['right']], motion)


def board_grasp_actions(ready):
    """Lift and draw the exposed hem with the fingers outside the board lip."""
    from yam_arena import station

    yield from (('idle', ready.copy()) for _ in range(100))
    rotation = np.diag([-1., 1., -1.])[None, :, :]
    grasp = (.126, .189, .782)
    lifted = (.102, .189, .862)
    place = (.146, .319, .840)
    stages = (
        ('board_entry', (.122, .189, .905), .22, 30),
        ('board_align', (.126, .189, .801), .22, 30),
        ('board_insert', grasp, .22, 22),
        ('board_pinch', grasp, 0., 25),
        ('board_pull_clear', (.102, .189, .787), 0., 25),
        ('board_lift', lifted, 0., 30),
        ('board_hold', lifted, 0., 20),
        ('board_draw_mid', (.105, .265, .875), 0., 30),
        ('board_draw_end', (.112, .318, .865), 0., 30),
        ('board_place', place, 0., 25),
        ('board_release', place, .22, 18),
        ('board_retreat', (.115, .319, .900), .22, 25),
    )
    motion = [(phase, np.array([point]), grip, controls, False, rotation)
              for phase, point, grip, controls in stages]
    yield from _move(ready, ('left',), (0,), [station.BASES['left']], motion)


def board_scoop_actions(ready):
    """Lift the hanging hem from below on the two closed finger tips."""
    from yam_arena import station

    yield from (('idle', ready.copy()) for _ in range(100))
    angle = np.deg2rad(35.)
    forward = np.array([np.sin(angle), 0., -np.cos(angle)])
    closing = np.array([0., -1., 0.])
    rotation = np.column_stack((closing, np.cross(forward, closing), forward))[None]
    lifted = (.129, .192, .826)
    stages = (
        ('scoop_entry', (.107, .177, .902), 32),
        ('scoop_align', (.091, .187, .783), 30),
        ('scoop_under', (.131, .187, .767), 25),
        ('scoop_lift', lifted, 30),
        ('scoop_hold', lifted, 24),
        ('scoop_draw', (.130, .267, .867), 32),
        ('scoop_retract', (.115, .245, .880), 25),
        ('scoop_retreat', (.107, .177, .902), 25),
    )
    motion = [(phase, np.array([point]), 0., controls, False, rotation)
              for phase, point, controls in stages]
    yield from _move(ready, ('left',), (0,), [station.BASES['left']], motion)


def turnover_lift_actions(ready, *, roll=False):
    """Engage two spaced side-seam patches for a supported turnover."""
    from yam_arena import station

    yield from (('idle', ready.copy()) for _ in range(100))
    # The angled entry keeps the jaw shoulders farther from the upper drape.
    rotation = np.array([
        [[-.909542, .030653, .414480], [.163718, .943066, .289521], [-.382008, .331189, -.862777]],
        [[.876115, .137401, .462108], [.157701, -.987472, -.005377], [.455580, .077586, -.886807]],
    ])

    def pair(left_x, right_x, left_z, right_z):
        return np.array([[left_x, .1898, left_z], [right_x, -.0927, right_z]])

    high = pair(.1167, .1079, .902, .902)
    align = pair(.1367, .1279, .8091, .8003)
    grasp = pair(.1497, .1409, .7901, .7813)
    clear = pair(.1317, .1229, .8091, .8003)
    lift = pair(.1317, .1229, .880, .880)
    lower = pair(.1317, .1229, .821, .821)
    stages = (
        ('seam_entry', high, .24, 32), ('seam_align', align, .24, 25),
        ('seam_insert', grasp, .24, 25), ('seam_pinch_left', grasp, (0., .24), 24),
        ('seam_pinch_right', grasp, 0., 24), ('seam_clear', clear, 0., 25),
        ('seam_lift', lift, 0., 30), ('seam_hold', lift, 0., 30),
    )
    motion = [(phase, target, grip, controls, False, rotation) for phase, target, grip, controls in stages]
    if roll:
        from yam_arena.urdf import axis_angle_matrix

        down = np.array([np.diag([-1., 1., -1.]), np.diag([1., -1., -1.])])
        arc = (
            ('turn_clear', .157, .151, .952, -12., 0., 30),
            ('turn_rise', .20, .20, 1.022, -40., 0., 30),
            ('turn_above_board', .29, .29, 1.070, -65., 0., 35),
            ('turn_vertical', .37, .37, 1.080, -90., 0., 35),
            ('turn_cross', .47, .47, 1.060, -115., 0., 35),
            ('turn_drop_clear', .493, .493, 1.105, -121., 0., 25),
            ('turn_lower', .47, .47, .984, -94., 0., 30),
            ('turn_place', .47, .47, .920, -73., 0., 25),
            ('turn_release', .47, .47, .920, -73., .22, 22),
            ('turn_retreat', .39, .39, .960, -40., .22, 30),
        )
        for phase, left_x, right_x, z, degrees, grip, controls in arc:
            pitch = axis_angle_matrix(np.array([0., 1., 0.]), np.deg2rad(degrees))
            motion.append((phase, pair(left_x, right_x, z, z), grip, controls, False, pitch @ down))
    else:
        motion.extend((('seam_lower', lower, 0., 25, False, rotation),
                       ('seam_release', lower, .24, 20, False, rotation),
                       ('seam_retreat', high, .24, 30, False, rotation)))
    yield from _move(ready, ('left', 'right'), (0, 7), [station.BASES['left'], station.BASES['right']], motion)



def pressing_table_lift_actions(ready):
    """Peel two patches of the overhanging seam up from the near lip."""
    from yam_arena import station

    yield from (('idle', ready.copy()) for _ in range(100))
    # Approach farther along +X so the finger shoulders remain outside the deck.
    forward = np.array([[.55, -.70, -.50], [.55, .70, -.50]])
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    normal = np.tile([-.6, 0., .8], (2, 1))
    normal -= np.sum(normal * forward, axis=1, keepdims=True) * forward
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    normal[1] *= -1
    rotation = np.stack((normal, np.cross(forward, normal), forward), axis=2)
    grip = np.array([[.1393, .0185, .8304], [.1393, -.2076, .8304]])
    # Put the edge inside the 33 mm pad length, rather than beyond the pad tips.
    grip += .014 * forward
    high = grip + [.0607, 0., .0996]
    align = grip + [.0407, 0., .0696]
    pre = grip + [.0207, 0., .0396]
    # Lift across the pads rather than drawing the fabric out past their tips.
    up = np.tile([0., 0., 1.], (2, 1))
    lift_direction = (up - forward[:, 2, None]*forward) / (1.-forward[:, 2, None]**2)
    peel = grip + .016*lift_direction
    lift = grip + .051*lift_direction
    anchor = np.array([.25, .02, .855])
    anchor_high = anchor + [0., 0., .065]
    down = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
    anchor_rotation = np.stack((down, rotation[1]))
    stages = (
        ('anchor_entry', [anchor_high, high[1]], (0., .10), 50, anchor_rotation),
        ('anchor_land', [anchor, high[1]], (0., .10), 45, anchor_rotation),
        ('anchor_hold', [anchor, high[1]], (0., .10), 25, anchor_rotation),
        ('right_edge_align', [anchor, align[1]], (0., .10), 35, anchor_rotation),
        ('right_edge_pre', [anchor, pre[1]], (0., .10), 30, anchor_rotation),
        ('right_edge_insert', [anchor, grip[1]], (0., .10), 60, anchor_rotation),
        ('right_edge_align_cloth', [anchor, grip[1]], (0., .10), 40, anchor_rotation),
        ('right_edge_pinch', [anchor, grip[1]], 0., 8, anchor_rotation),
        ('right_edge_peel', [anchor, peel[1]], 0., 35, anchor_rotation),
        ('right_edge_hold', [anchor, peel[1]], 0., 25, anchor_rotation),
        ('left_anchor_clear', [anchor_high, peel[1]], 0., 35, anchor_rotation),
        ('left_edge_entry', [high[0], peel[1]], (.10, 0.), 45, rotation),
        ('left_edge_align', [align[0], peel[1]], (.10, 0.), 30, rotation),
        ('left_edge_pre', [pre[0], peel[1]], (.10, 0.), 25, rotation),
        ('left_edge_insert', [grip[0], peel[1]], (.10, 0.), 60, rotation),
        ('left_edge_align_cloth', [grip[0], peel[1]], (.10, 0.), 40, rotation),
        ('table_edge_pinch', [grip[0], peel[1]], 0., 8, rotation),
        ('table_edge_peel', peel, 0., 28, rotation),
        ('table_edge_lift', lift, 0., 35, rotation), ('table_edge_hold', lift, 0., 25, rotation),
        ('table_edge_lower', peel, 0., 25, rotation), ('table_edge_release', peel, .10, 20, rotation),
        ('table_edge_retract', high, .24, 27, rotation),
    )
    motion = [(phase, np.asarray(target), opening, controls, phase == 'anchor_entry', orientation)
              for phase, target, opening, controls, orientation in stages]
    yield from _move(ready, ('left', 'right'), (0, 7), [station.BASES['left'], station.BASES['right']], motion)


def table_grasp_feedback(command, cloth, base, phase, shifts, previous):
    """Align an open pinch to live cloth; subsequent motion keeps that measured offset."""
    from yam_arena import ik, station

    adjusted = command.copy()
    points = cloth.nodes(base)[0] if phase.endswith('edge_align_cloth') else None
    for side, offset, material_x in (('left', 0, .47), ('right', 7, .26)):
        aligning = phase in ('table_edge_align_cloth', f'{side}_edge_align_cloth')
        if not aligning and side not in shifts:
            continue
        pose, rotation = ik.tcp_pose(command[offset:offset+6], station.BASES[side])
        if aligning:
            patch = ((np.abs(cloth.rest[:, 0]-material_x) < .0051)
                     & (cloth.rest[:, 1] > .2449))
            goal = points[patch].mean(axis=0) - .006*rotation[:, 2]
            if np.linalg.norm(goal-pose) > .035:
                raise RuntimeError(f'{side} cloth has left the calibrated grasp neighbourhood')
            start, _ = ik.tcp_pose(previous[offset:offset+6], station.BASES[side])
            delta = goal-start
            target = start + delta*min(1., .002/max(np.linalg.norm(delta), 1e-9))
            shifts[side] = target-pose
        else:
            target = pose+shifts[side]
        q, pe, re = ik.solve(target, ik.matrix_to_quat_xyzw(rotation), previous[offset:offset+6],
                             base_pos=station.BASES[side], iters=200)
        if pe > .002 or re > .03:
            raise RuntimeError(f'Unreachable live {side} {phase}: {pe:.4f} m, {re:.4f} rad')
        adjusted[offset:offset+6] = q
    return adjusted


def short_board_hook_actions(ready, *, pinch=False, roll=False):
    """Lift the free hem and neckline from outside the board ends."""
    from yam_arena import station

    yield from (('idle', ready.copy()) for _ in range(100))
    rotation = np.array([
        [[.154508, -.866025, .475528], [-.267617, -.5, -.823639], [.951057, 0., -.309017]],
        [[-.237587, -.939693, .246029], [-.652765, .342020, .675958], [-.719340, 0., -.694658]],
    ])
    support_rotation = rotation.copy()
    support_rotation[1] = [[-.105690, -.939693, .325280], [-.290381, .342020, .893701],
                           [-.951057, 0., -.309017]]
    if pinch or roll:
        pinch_rotation = support_rotation.copy()
        pinch_rotation[1] = [[-.1307201, -.9199404, .3696240], [-.0435095, .3777871, .9248697],
                             [-.9904641, .1048169, -.0894105]]
        high = np.array([[.260, .369, .892], [.343, -.291, .909]])
        align = np.array([[.22489, .34997, .88572], [.31913, -.27620, .84215]])
        pre = np.array([[.22489, .34997, .87172], [.31913, -.27620, .82815]])
        grip = np.array([[.2363, .3302, .8643], [.328, -.254, .826]])
        lift = np.array([[.2363, .3362, .911], [.3336, -.2675, .908]])
        pickup = lift.copy()
        pickup[1, :2] = grip[1, :2]
        aperture = (.38, .54)
        stages = (
            ('pinch_entry', high, aperture, 32), ('pinch_align', align, aperture, 25),
            ('pinch_lower', pre, aperture, 22), ('pinch_insert', grip, aperture, 24),
            ('pinch_left', grip, (0., aperture[1]), 30), ('pinch_right', grip, 0., 40),
            ('pinch_lift', pickup, 0., 36), ('pinch_shift', lift, 0., 18), ('pinch_hold', lift, 0., 35),
        )
        motion = [(phase, target, opening, controls, False, pinch_rotation)
                  for phase, target, opening, controls in stages]
        if roll:
            from yam_arena.urdf import axis_angle_matrix

            down = np.array([np.diag([-1., 1., -1.]), np.diag([1., -1., -1.])])
            rotation_steps = (
                ('rotate_clear', .282, .340, .977, -18., 0., 45, True),
                ('rotate_rise', .330, .398, .978, -48., 0., 32, False),
                ('rotate_across', .391, .444, 1.016, -70., 0., 35, False),
                ('rotate_over', .463, .482, 1.011, -89., 0., 36, False),
                ('rotate_lower', .451, .474, .954, -92., 0., 30, False),
                ('rotate_place', .437, .460, .902, -83., 0., 32, False),
                ('rotate_hold', .437, .460, .902, -83., 0., 22, False),
                ('rotate_release', .437, .460, .902, -83., .30, 22, False),
                ('rotate_retreat', .385, .411, .959, -45., .30, 28, False),
            )
            for phase, left, right, height, degrees, opening, controls, joint_entry in rotation_steps:
                target = np.array([[left, lift[0, 1], height], [right, lift[1, 1], height]])
                r = axis_angle_matrix(np.array([0., 1., 0.]), np.deg2rad(degrees)) @ down
                motion.append((phase, target, opening, controls, joint_entry, r))
        else:
            # Test retained contact without the turnover's abrupt wrist reorientation.
            from yam_arena.urdf import axis_angle_matrix

            rock = axis_angle_matrix(np.array([0., 1., 0.]), np.deg2rad(15.))
            # Preserve the short TCP-to-fabric offset while rotating each grasp.
            offset = .004*pinch_rotation[:, :, 2]
            rocked = lift-offset + (rock @ offset.T).T
            motion.extend((('pinch_rock', rocked, 0., 50, False, rock @ pinch_rotation),
                           ('pinch_rock_hold', rocked, 0., 40, False, rock @ pinch_rotation),
                           ('pinch_unrock', lift, 0., 50, False, pinch_rotation)))
            motion.extend((('pinch_place', grip, 0., 32, False, pinch_rotation),
                           ('pinch_release', grip, aperture, 20, False, pinch_rotation),
                           ('pinch_retract', pre, aperture, 25, False, pinch_rotation),
                           ('pinch_retreat', high, aperture, 25, False, pinch_rotation)))
        yield from _move(ready, ('left', 'right'), (0, 7), [station.BASES['left'], station.BASES['right']], motion)
        return
    stages = (
        ('end_entry', (.364, .890), (-.263, .909), 32),
        ('end_lower', (.364, .841), (-.263, .787), 30),
        ('end_approach', (.349, .841), (-.248, .787), 25),
        ('end_under', (.335, .841), (-.234, .787), 22),
        ('end_scoop', (.338, .876), (-.230, .822), 30),
        ('end_lift', (.342, .912), (-.230, .909), 32),
        ('end_hold', (.342, .912), (-.230, .909), 25),
        ('end_retract', (.364, .890), (-.263, .909), 30),
    )
    supporting = ('end_scoop', 'end_lift', 'end_hold')
    motion = [(phase, np.array([[.248, *left], [.333, *right]]), 0., controls, False,
               support_rotation if phase in supporting else rotation)
              for phase, left, right, controls in stages]
    yield from _move(ready, ('left', 'right'), (0, 7), [station.BASES['left'], station.BASES['right']], motion)
