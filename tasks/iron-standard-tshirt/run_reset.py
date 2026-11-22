#!/usr/bin/env python3
"""Record the native idle, cold-iron pass, or hem manipulation on visnu."""

import argparse
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import socket
import subprocess

import numpy as np


def stroke_actions(ready, iron_rest, measure_load=False):
    """Move the right gripper through one grasp, sweep and return."""
    from yam_arena import ik, station

    pose, rotation = ik.tcp_pose(ready[7:13], station.BASES["right"])
    down = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
    rest = np.asarray(iron_rest) + [-.006, 0., .100]
    rest_above = rest + [0., 0., .0665]
    start_above = (.34, -.18, .92)
    start = (.34, -.18, .855)
    end = (.25, -.18, .855)
    stages = (
        ("approach", rest_above, 1., 25), ("lower", rest, 1., 15),
        ("grasp", rest, .20, 15), ("lift", rest_above, .20, 20),
        ("carry", start_above, .20, 25), ("land", start, .20, 20),
        ("sweep", end, .20, 30), ("withdraw", (.25, -.18, .92), .20, 20),
        ("return", rest_above, .20, 25), ("park", rest, .20, 20),
        ("release", rest, 1., 15), ("clear", rest_above, 1., 20),
    )
    if measure_load:
        hover = (.34, -.18, .881)
        bare_above = (.26, -.38, .92)
        bare = (.26, -.38, .853)
        stages = (("empty_approach", hover, .20, 30), ("empty_hover", hover, .20, 30),
                  *stages[:5], ("hover_approach", hover, .20, 20),
                  ("loaded_hover", hover, .20, 35), stages[5],
                  ("cloth_hold", start, .20, 35), *stages[6:8],
                  ("bare_approach", bare_above, .20, 25), ("bare_land", bare, .20, 20),
                  ("bare_hold", bare, .20, 35), ("bare_withdraw", bare_above, .20, 20), *stages[8:])
    command = ready.copy()
    for phase, target, grip, controls in stages:
        opening = command[13]
        target = np.asarray(target)
        for t in range(1, controls + 1):
            ratio = ik.smoothstep(t / controls)
            xyz = pose + ratio * (target - pose)
            rot = ik.slerp_matrix(rotation, down, ratio)
            q, pe, re = ik.solve(xyz, ik.matrix_to_quat_xyzw(rot), command[7:13],
                                 base_pos=station.BASES["right"], iters=80)
            if pe > .002 or re > .03:
                raise RuntimeError(f"Unreachable {phase} target: {pe:.4f} m, {re:.4f} rad")
            command[7:13] = q
            command[13] = opening + ratio * (grip - opening)
            yield phase, command.copy()
        pose, rotation = target, down
    for _ in range(100):
        yield "released_hold", command.copy()


def buckle_stroke_actions(ready, iron_rest, *, hover, wide=False, table=False, table_top=.7775,
                          stabilize=False):
    """Move the pointed end of the sole along the settled wrinkle toward the side seam."""
    from yam_arena import station
    from unfold import _move

    yield from (('idle', ready.copy()) for _ in range(100))
    left_down = np.diag([-1., 1., -1.])[None]
    if stabilize:
        support_high = np.array([[.310, .200, .840]])
        support_low = np.array([[.310, .200, .7635]])
        supporting = (
            ('stabilize_approach', support_high, 0., 35, False, left_down),
            ('stabilize_lower', support_low, 0., 35, False, left_down),
            ('stabilize_hold', support_low, 0., 20, False, left_down),
        )
        for phase, command in _move(ready, ('left',), (0,), [station.BASES['left']], supporting):
            if phase != 'released_hold':
                ready = command.copy()
                yield phase, command
    down = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
    yaw = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    toe_first = yaw @ down
    rest = np.asarray(iron_rest) + [-.006, 0., .100]
    above_rest = rest + [0., 0., .0665]
    # The held iron was released about 12 mm back and 14 mm inward in both trials.
    park = rest + [.012, -.014, 0.]
    above_park = park + [0., 0., .0665]
    turn = (.275, -.433, .955)
    z = .898 if hover else .857
    start = (.370, -.092, z)
    end = (.377, -.226, z) if wide else (.370, -.161, z)
    stages = (
        ('approach', above_rest, 1., 25, down), ('lower', rest, 1., 15, down),
        ('grasp', rest, .20, 15, down), ('lift', above_rest, .20, 20, down),
        ('carry_clear', turn, .20, 22, down),
    )
    if table:
        table_offset = table_top - .7775
        near, far = (.200, -.175, .879 + table_offset), (.465, -.175, .879 + table_offset)
        stages += (
            ('carry', (.223, -.175, .925 + table_offset), .20, 30, down), ('land', near, .20, 30, down),
            ('press_hold', near, .20, 22, down), ('sweep', far, .20, 120, down),
            ('sweep', near, .20, 120, down), ('withdraw', (.223, -.175, .925 + table_offset), .20, 25, down),
        )
        return_orientation = down
    else:
        stages += (
            ('aim_toe', turn, .20, 30, toe_first),
            ('carry', (.370, -.092, .898), .20, 30, toe_first), ('land', start, .20, 30, toe_first),
            ('press_hold', start, .20, 22, toe_first), ('sweep', end, .20, 88 if wide else 56, toe_first),
        )
        if wide:
            stages += (('press_hold', end, .20, 16, toe_first),)
        if stabilize:
            stages += (('sweep', start, .20, 56, toe_first),
                       ('sweep', end, .20, 56, toe_first),
                       ('press_hold', end, .20, 16, toe_first))
        stages += (('withdraw', (*end[:2], .898), .20, 23, toe_first),)
        return_orientation = toe_first
    if wide:
        from yam_arena.urdf import axis_angle_matrix

        across = axis_angle_matrix(np.array([0., 0., 1.]), np.deg2rad(48.)) @ down
        outer, inner = (.311, -.036, z), (.311, .038, z)
        across_high = (.311, -.036, .898)
        stages += (
            ('shift_clear', turn, .20, 24, toe_first), ('aim_center', turn, .20, 30, down),
            ('aim_across', turn, .20, 30, across), ('carry_across', across_high, .20, 34, across),
            ('land', outer, .20, 30, across), ('press_hold', outer, .20, 22, across),
            ('sweep', inner, .20, 65, across), ('press_hold', inner, .20, 16, across),
            ('sweep', outer, .20, 65, across), ('withdraw_across', across_high, .20, 28, across),
        )
        return_orientation = across
    stages += (
        ('return_clear', turn, .20, 24, return_orientation), ('aim_rest', turn, .20, 30, down),
        ('return', above_park, .20, 22, down), ('park', park, .20, 20, down),
        ('release', park, 1., 15, down), ('clear', above_park, 1., 20, down),
    )
    motion = [(phase, np.array([point]), grip, controls, False, rotation[None])
              for phase, point, grip, controls, rotation in stages]
    if stabilize:
        # Park forward of the rest so the forearm no longer hides the iron.
        clear_rotation = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        motion.append(('finish_right_clear', np.array([[.450, -.305, 1.000]]),
                       1., 40, False, clear_rotation[None]))
    for phase, command in _move(ready, ('right',), (7,), [station.BASES['right']], motion):
        if not stabilize or phase != 'released_hold':
            ready = command.copy()
            yield phase, command
    if stabilize:
        release_support = (
            ('stabilize_lift_clear', support_high, 0., 30, False, left_down),
            ('stabilize_retreat', np.array([[.220, .260, .960]]), 1., 35, False, left_down),
        )
        yield from _move(ready, ('left',), (0,), [station.BASES['left']], release_support)


def main():
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from resist_isaac.runtime import ResistanceEmbodiment, _task_module
    from resist_isaac.state import pose_state, tensor
    from yam_arena.constants import FINGER_JOINTS, FINGER_OPEN
    from yam_arena.runtime import tcp_world
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0", help="Native simulation and rendering GPU")
    parser.add_argument("--stop-after-phase", help="End a diagnostic at a completed controller phase")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stroke", action="store_true", help="Run an experimental single cold pass")
    mode.add_argument("--press-wrench", action="store_true", help="Compare native wrist load in hover and contact")
    mode.add_argument("--unfold", action="store_true", help="Run a released bimanual hem unfolding control")
    mode.add_argument("--unfold-deep", action="store_true", help="Try a deeper pinch and lower draw")
    mode.add_argument("--unfold-arc", action="store_true", help="Roll the flap on a shorter arc near the supporting panel")
    mode.add_argument("--unfold-roll", action="store_true", help="Guide the loose hem toward the mat with a wrist roll")
    resolution = parser.add_mutually_exclusive_group()
    resolution.add_argument("--diagnostic-resolution", action="store_true", help="Use 640 x 360 at the calibrated camera poses")
    resolution.add_argument("--full-hd", action="store_true", help="Render native 1920 x 1080 top frames")
    parser.add_argument("--record-top", action="store_true", help="Record every calibrated top-camera frame as H.264 MP4")
    mode.add_argument("--diagonal-reset", action="store_true", help="Idle control for the diagonal fold geometry")
    mode.add_argument("--diagonal-peel", action="store_true", help="Grasp the exposed corner of the diagonal fold")
    mode.add_argument("--diagonal-peel-low", action="store_true", help="Peel the corner close to the board")
    mode.add_argument("--diagonal-refined-peel", action="store_true", help="Repeat the low peel with a finer cloth corner")
    mode.add_argument("--diagonal-contact-peel", action="store_true", help="Repeat the low peel with finer native contact timing")
    mode.add_argument("--diagonal-roll-push", action="store_true", help="Support the torso and roll the free edge with the other hand")
    mode.add_argument("--board-reset", action="store_true", help="Check the draped reset on a raised ironing board")
    mode.add_argument("--board-grasp", action="store_true", help="Lift and draw the draped hem outside the board lip")
    mode.add_argument("--board-scoop", action="store_true", help="Guide the draped hem from below with the closed jaws")
    mode.add_argument("--turnover-reset", action="store_true", help="Check the spread shirt's seam access on the board")
    mode.add_argument("--short-board-reset", action="store_true", help="Check hanging hem and collar access on a short deck")
    mode.add_argument("--short-board-hook", action="store_true", help="Lift the hanging hem and collar from beneath")
    mode.add_argument("--short-board-pinch", action="store_true", help="Grip the exposed hem and collar for turnover")
    mode.add_argument("--short-board-turn", action="store_true", help="Turn the shirt using the exposed hem and collar")
    mode.add_argument("--pressing-table-reset", action="store_true", help="Check the full-width pressing surface and edge access")
    mode.add_argument("--pressing-table-stroke", action="store_true", help="Press the exposed crease on the full-width surface")
    mode.add_argument("--pressing-table-lift", action="store_true", help="Pick up two overhanging side-seam patches")
    mode.add_argument("--turnover-lift", action="store_true", help="Lift the spread shirt by two separated side-seam patches")
    mode.add_argument("--turnover-roll", action="store_true", help="Roll the lifted shirt across the board and release")
    mode.add_argument("--z-buckle-reset", action="store_true", help="Check idle stability of a short torso wrinkle")
    mode.add_argument("--settled-reset", action="store_true", help="Check the fixed initial pose for a short torso wrinkle")
    mode.add_argument("--z-buckle-stroke", action="store_true", help="Run a supported toe-first stroke on the settled torso wrinkle")
    mode.add_argument("--stabilized-front-press", action="store_true", help="Steady fabric with the left hand while the right presses its front")
    mode.add_argument("--wide-stroke", action="store_true", help="Run two overlapping press paths on the front crease")
    mode.add_argument("--z-buckle-hover", action="store_true", help="Replay the toe-first stroke above the cloth")
    args = parser.parse_args()
    task = Path(__file__).resolve().parent
    out = args.output.resolve()
    if task == out or task in out.parents:
        raise ValueError("Use a new output directory outside the task source")
    out.mkdir(parents=True, exist_ok=False)
    _, fingerprint = _task_module(task)
    scope = "native single cold-iron pass diagnostic" if args.stroke else "native reset and idle only"
    if args.press_wrench:
        scope = "native wrist load diagnostic with known iron mass, cloth and bare mat"
    hem_variant = next((name for enabled, name in ((args.unfold_roll, 'roll'), (args.unfold_arc, 'arc'),
                       (args.unfold_deep, 'deep'), (args.unfold, 'edge'),
                       (args.diagonal_peel, 'corner'), (args.diagonal_peel_low or args.diagonal_refined_peel or args.diagonal_contact_peel,
                        'corner_low')) if enabled), None)
    if hem_variant:
        scope = "native bimanual hem unfolding control"
    if args.diagonal_roll_push:
        scope = "native supported push of the diagonal hem with standard YAM jaws"
    width, height = ((640, 360) if args.diagnostic_resolution else
                     (1920, 1080) if args.full_hd else (1280, 720))
    diagonal = args.diagonal_reset or args.diagonal_roll_push or hem_variant in ('corner', 'corner_low')
    condition = 'diagonal_probe' if diagonal else 'recoverable'
    if args.diagonal_refined_peel:
        condition = 'diagonal_refined'
    if args.diagonal_contact_peel:
        condition = 'diagonal_contact'
    board_motion = args.board_grasp or args.board_scoop
    if args.board_reset or board_motion:
        condition = 'raised_board'
        scope = 'native raised-board hem lift and draw' if args.board_grasp else 'native raised ironing board reset and idle'
        if args.board_scoop:
            scope = 'native raised-board scoop and guide from beneath the hem'
    turnover_motion = args.turnover_lift or args.turnover_roll
    if args.turnover_reset or turnover_motion:
        condition = 'turnover'
        scope = 'native two-seam turnover lift' if turnover_motion else 'native board reset for the spread garment turnover'
        if args.turnover_roll:
            scope = 'native two-seam rollover and release'
    short_board_motion = args.short_board_hook or args.short_board_pinch or args.short_board_turn
    if args.short_board_reset or short_board_motion:
        condition = 'settled_short_turnover' if short_board_motion else 'short_turnover'
        scope = ('native short-board end lift' if short_board_motion
                 else 'native short ironing board reset for end access')
        if args.short_board_pinch:
            scope = 'native short-board end pinch and release'
        if args.short_board_turn:
            scope = 'native short-board complete turnover attempt'
    table_modes = ((args.pressing_table_reset, 'reset and idle'), (args.pressing_table_stroke, 'press'),
                   (args.pressing_table_lift, 'two-point pickup'))
    for enabled, operation in table_modes:
        if enabled:
            condition = 'pressing_table'
            scope = f'native full-width pressing table {operation}'
    buckle_stroke = args.z_buckle_stroke or args.z_buckle_hover or args.wide_stroke or args.stabilized_front_press
    record_wrench = args.press_wrench or buckle_stroke or args.pressing_table_stroke
    if args.z_buckle_reset:
        condition = 'z_buckle'
        scope = 'native short torso wrinkle reset and idle'
    if args.settled_reset or buckle_stroke:
        condition = 'settled_buckle'
        scope = 'native fixed torso wrinkle reset and idle'
    if buckle_stroke:
        scope = 'native toe-first torso wrinkle sweep with ' + ('hover' if args.z_buckle_hover else 'plate support')
    if args.wide_stroke:
        condition = 'wide_buckle'
        scope = 'native overlapping front strokes with plate support'
    if args.stabilized_front_press:
        scope = 'native bimanual supported front pressing; turnover and final ironing score unverified'
    receipt = dict(status="running", hostname=socket.gethostname(), scope=scope, hem_variant=hem_variant,
                   condition=condition,
                   task_fingerprint=fingerprint, runner_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                   seed=0, camera_resolution=[width, height], device=args.device,
                   stop_after_phase=args.stop_after_phase,
                   gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version",
                                                       "--format=csv,noheader"], text=True).strip(),
                   versions={name: version(name) for name in ("isaacsim", "isaaclab", "isaaclab-arena", "yam-arena")})
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    emb = ResistanceEmbodiment(task=str(task), condition=condition, device=args.device, cam_width=width, cam_height=height)
    video = None
    unfinished_video = out / 'top-camera-incomplete.mp4'
    try:
        obs = emb.reset(Scene(id=task.name, instruction="Iron both sides of the T-shirt until smooth and crease-free."), seed=0)
        cloth = emb.task
        native_rgb = tensor(emb._base.scene['top_cam'].data.output['rgb'])
        receipt['native_top_camera_resolution'] = [int(native_rgb.shape[2]), int(native_rgb.shape[1])]
        if receipt['native_top_camera_resolution'] != [width, height]:
            raise RuntimeError('Native top-camera dimensions differ from the requested recording; refusing an upscale')
        receipt["physics_dt_s"] = float(emb._base.physics_dt)
        receipt["control_dt_s"] = float(emb._base.step_dt)
        receipt["use_fabric"] = bool(emb._base.sim.cfg.use_fabric)
        receipt["render_kit_args"] = cloth.render_kit_args
        receipt["robot_contact_offset_m"] = getattr(cloth, "robot_contact_offset", None)
        receipt["finger_pad_material_colliders"] = getattr(cloth, "finger_pad_material_colliders", None)
        from pxr import Usd, UsdPhysics
        from isaaclab.sim.utils.stage import get_current_stage

        offsets = []
        stage = get_current_stage()
        receipt['articulation_contact_last'] = {
            str(prim.GetPath()): prim.GetAttribute('physxScene:solveArticulationContactLast').Get()
            for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)
        }
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if "tip_" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
                attr = prim.GetAttribute("physxCollision:contactOffset")
                if attr and attr.Get() is not None:
                    offsets.append(float(attr.Get()))
        receipt["finger_collider_contact_offsets_m"] = sorted(set(offsets))
        segments = []
        nodes = []
        render_flags = []
        ready = cloth.initial_action()
        if board_motion:
            from unfold import board_grasp_actions, board_scoop_actions

            motion = list((board_scoop_actions if args.board_scoop else board_grasp_actions)(ready))
        elif turnover_motion:
            from unfold import turnover_lift_actions

            motion = list(turnover_lift_actions(ready, roll=args.turnover_roll))
        elif short_board_motion:
            from unfold import short_board_hook_actions

            motion = list(short_board_hook_actions(ready, pinch=args.short_board_pinch, roll=args.short_board_turn))
        elif args.pressing_table_lift:
            from unfold import pressing_table_lift_actions

            motion = list(pressing_table_lift_actions(ready))
        elif args.diagonal_roll_push:
            from unfold import roll_push_actions

            motion = list(roll_push_actions(ready))
        elif hem_variant:
            from unfold import hem_actions

            motion = list(hem_actions(ready, variant=hem_variant))
        elif buckle_stroke or args.pressing_table_stroke:
            motion = list(buckle_stroke_actions(ready, cloth.iron_rest, hover=args.z_buckle_hover,
                                                wide=args.wide_stroke, table=args.pressing_table_stroke,
                                                table_top=cloth.support_top, stabilize=args.stabilized_front_press))
        elif args.stroke or args.press_wrench:
            motion = list(stroke_actions(ready, cloth.iron_rest, measure_load=args.press_wrench))
        else:
            motion = [("idle", ready)] * 100
        if args.stop_after_phase:
            matches = [i for i, (phase, _) in enumerate(motion) if phase == args.stop_after_phase]
            if not matches:
                raise ValueError(f'Unknown stop phase: {args.stop_after_phase}')
            motion = motion[:matches[-1] + 1]
        timeline = [("reset", None), *motion]
        if args.record_top:
            video = subprocess.Popen(['ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'rawvideo',
                                      '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                                      '-framerate', str(1/receipt['control_dt_s']), '-i', 'pipe:0', '-an',
                                      '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p',
                                      '-movflags', '+faststart', str(unfinished_video)], stdin=subprocess.PIPE)
        arms = {side: emb._base.scene[f'{side}_robot'] for side in ('left', 'right')}
        fingers = {side: arm.find_joints(FINGER_JOINTS, preserve_order=True)[0]
                   for side, arm in arms.items()}
        if record_wrench:
            wrist = arms['right']
            wrench_link = wrist.backend_body_names.index('gripper')
        last_phase = None
        top_reference = None
        track_grasps = short_board_motion or args.pressing_table_lift or args.stabilized_front_press
        sample_period = 5 if record_wrench or turnover_motion or track_grasps else 10 if board_motion else 20
        grasp_shifts = {}
        previous_command = ready.copy()
        for step, (phase, command) in enumerate(timeline):
            if step:
                if args.pressing_table_lift and (phase.endswith('edge_align_cloth') or grasp_shifts):
                    from unfold import table_grasp_feedback

                    command = table_grasp_feedback(command, cloth, emb._base, phase,
                                                   grasp_shifts, previous_command)
                result = emb.step(Action(data=command))
                obs = result.observation
                previous_command = command.copy()
            if step == 100:
                import warp as wp

                receipt['native_robot_materials'] = {}
                for side, arm in arms.items():
                    values = wp.to_torch(arm.root_view.get_material_properties()).cpu().numpy().reshape(-1, 3)
                    unique, counts = np.unique(values, axis=0, return_counts=True)
                    receipt['native_robot_materials'][side] = {
                        'static_dynamic_restitution': unique.tolist(), 'shape_counts': counts.tolist()
                    }
                    if np.count_nonzero(np.isclose(values[:, 1], 1.0)) < 8:
                        raise RuntimeError(f'{side} pad friction is not active in native PhysX shapes')
            if video is not None:
                frame = obs.images['top_cam']
                video.stdin.write(np.ascontiguousarray(frame).tobytes())
                lo, hi = np.percentile(frame[::8, ::8], [5, 95])
                if hi - lo < 12:
                    render_flags.append(dict(step=step, camera='top_cam', issue='near-uniform frame'))
            transition = step == len(motion) or timeline[step + 1][0] != phase
            if step % sample_period == 0 or transition:
                pos, quat, vel = pose_state(emb._base.scene["iron"])
                sample = dict(step=step, phase=phase, iron_position_m=pos.cpu().tolist(),
                              iron_orientation_native=quat.cpu().tolist(),
                              tcp_m={side: np.asarray(tcp_world(arm)).tolist() for side, arm in arms.items()},
                              opening={side: float(tensor(arm.data.joint_pos)[0, fingers[side]].mean())/FINGER_OPEN
                                       for side, arm in arms.items()},
                              iron_velocity=vel.cpu().tolist(), **cloth.evaluate(emb._base))
                if grasp_shifts:
                    sample['grasp_target_offset_m'] = {side: shift.tolist() for side, shift in grasp_shifts.items()}
                if track_grasps:
                    sample['gripper_orientation_native'] = {
                        side: pose_state(arm, 'gripper')[1].cpu().tolist() for side, arm in arms.items()
                    }
                    if args.pressing_table_lift or args.stabilized_front_press:
                        sample['grasp_wrist_force_world_N'] = {
                            side: cloth.gripper_wrench(arm, arm.backend_body_names.index('gripper'))[1].tolist()
                            for side, arm in arms.items()
                        }
                if record_wrench:
                    wrench, force = cloth.gripper_wrench(wrist, wrench_link)
                    sample['wrist_force_world_N'] = force.tolist()
                    sample['wrist_wrench_link'] = wrench.tolist()
                segments.append(sample)
                with (out/'progress.jsonl').open('a') as log:
                    log.write(json.dumps(sample) + '\n')
                nodes.append(cloth.nodes(emb._base)[0])
                if transition:
                    partial = out / "cloth-partial.tmp.npz"
                    np.savez_compressed(partial, positions=np.asarray(nodes), rest=cloth.rest, faces=cloth.faces)
                    partial.replace(out / "cloth-partial.npz")
            if step in (0, len(motion)) or (phase not in ('idle', 'reset') and phase != last_phase):
                for camera, frame in obs.images.items():
                    if (video is None or camera != 'top_cam') and np.percentile(frame, 95) - np.percentile(frame, 5) < 12:
                        # Keep the raw frame for diagnosis while marking it unsuitable as evidence.
                        render_flags.append(dict(step=step, camera=camera, issue='near-uniform frame'))
                    if camera == 'top_cam':
                        iron_position, _, _ = pose_state(emb._base.scene['iron'])
                        landmarks = np.array([iron_position.cpu().numpy(), *(tcp_world(arm) for arm in arms.values())])
                        motion_m = (0. if top_reference is None else
                                    float(np.linalg.norm(landmarks-top_reference[0], axis=1).max()))
                        if top_reference is None or motion_m >= .08:
                            current = frame.astype(np.int16)
                            if top_reference is not None:
                                changed = float(np.mean(np.max(np.abs(current-top_reference[1]), axis=2) > 20))
                                if changed < .006:
                                    render_flags.append(dict(step=step, camera=camera, issue='possibly stale frame',
                                                             native_displacement_m=motion_m, changed_fraction=changed))
                            top_reference = (landmarks, current)
                    if not cv2.imwrite(str(out / f"{step:03d}-{camera}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError(f"Failed writing {camera} at control {step}")
            last_phase = phase
        if video is not None:
            video.stdin.close()
            if video.wait(timeout=60) != 0:
                raise RuntimeError('The calibrated top-camera video failed to encode')
            probe = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                '-show_entries', 'stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames', '-of', 'json',
                str(unfinished_video),
            ], text=True))['streams'][0]
            if (probe['codec_name'] != 'h264' or probe['pix_fmt'] != 'yuv420p'
                    or [probe['width'], probe['height']] != [width, height]
                    or abs(float(Fraction(probe['avg_frame_rate']))*receipt['control_dt_s']-1.) > 1e-5
                    or int(probe['nb_read_frames']) != len(timeline)):
                raise RuntimeError(f'Top-camera video does not match the native action timeline: {probe}')
            video_path = unfinished_video.replace(out / 'top-camera.mp4')
            receipt['top_camera_video'] = dict(path=video_path.name, size_bytes=video_path.stat().st_size, **probe)
        np.savez_compressed(out / "cloth.npz", positions=np.asarray(nodes), rest=cloth.rest, faces=cloth.faces)
        (out / "cloth-partial.npz").unlink(missing_ok=True)
        receipt.update(status="completed", observations=segments, render_flags=render_flags,
                       source_unchanged=_task_module(task)[1] == fingerprint)
    except Exception as exc:
        receipt.update(status="error", error=repr(exc))
        raise
    finally:
        if video is not None and video.poll() is None:
            video.terminate()
            video.wait(timeout=10)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        emb.close()


if __name__ == "__main__":
    main()
