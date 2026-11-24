"""Supported passes across the fold; ordinary robot actions only."""
import numpy as np


def actions(ready, iron_rest, *, hover=False):
    from yam_arena import station
    from unfold import _move

    ready = ready.copy()
    yield from (('idle', ready.copy()) for _ in range(100))
    left_down = np.diag([-1., 1., -1.])[None]
    high = np.array([[.310, .200, .840]])
    low = np.array([[.310, .200, .7635]])
    support = (('stabilize_approach', high, 0., 35, False, left_down),
               ('stabilize_lower', low, 0., 35, False, left_down),
               ('stabilize_hold', low, 0., 20, False, left_down))
    for phase, command in _move(ready, ('left',), (0,), [station.BASES['left']], support):
        if phase != 'released_hold':
            ready = command.copy()
            yield phase, command
    down = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
    rest = np.asarray(iron_rest) + [-.006, 0., .100]
    above_rest = rest + [0., 0., .0665]
    park = rest + [.012, -.014, 0.]
    above_park = park + [0., 0., .0665]
    turn = (.275, -.433, .955)
    z = .898 if hover else .8565
    stages = [
        ('approach', above_rest, 1., 25, down), ('lower', rest, 1., 15, down),
        ('grasp', rest, .20, 15, down), ('lift', above_rest, .20, 20, down),
        ('carry_clear', turn, .20, 22, down),
    ]
    for label, y in (('outer', -.180), ('middle', -.110), ('inner', -.040)):
        stages += [
            (label+'_carry', (.250, y, .920), .20, 35, down),
            (label+'_land', (.250, y, z), .20, 30, down),
            (label+'_press', (.250, y, z), .20, 22, down),
            (label+'_cross_fold', ({'outer': .400, 'middle': .380, 'inner': .345}[label], y, z), .20, 100, down),
            (label+'_withdraw', ({'outer': .380, 'middle': .340, 'inner': .280}[label], y, .898), .20, 25, down),
        ]
    # Turn the sole sideways so its wider section follows the raised crest.
    across = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]) @ down
    crest_x = .35 if hover else .40  # Raised wrist cannot reach the loaded x target.
    stages += [
        ('crest_turn', (.29, -.21, .92), .20, 40, across),
        ('crest_carry', (.35, -.06, .90), .20, 32, across),
        ('crest_land', (crest_x, -.06, z), .20, 25, across),
        ('crest_sweep_outward', (crest_x, -.22, z), .20, 85, across),
        ('crest_lift', (.35, -.22, .90), .20, 25, across),
        ('crest_reset_inner', (.35, -.06, .90), .20, 30, across),
        ('crest_second_land', (crest_x, -.06, z if hover else z-.0015), .20, 25, across),
        ('crest_second_sweep', (crest_x, -.22, z if hover else z-.0015), .20, 100, across),
        ('crest_second_lift', (.35, -.22, .90), .20, 25, across),
    ]
    clear_rotation = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    stages += [
        ('return_clear', turn, .20, 45, down), ('return', above_park, .20, 22, down),
        ('park', park, .20, 20, down), ('release', park, 1., 15, down),
        ('clear', above_park, 1., 20, down),
        ('finish_right_clear', (.450, -.305, 1.), 1., 40, clear_rotation),
    ]
    motion = [(phase, np.array([point]), grip, controls, False, rotation[None])
              for phase, point, grip, controls, rotation in stages]
    for phase, command in _move(ready, ('right',), (7,), [station.BASES['right']], motion):
        if phase != 'released_hold':
            ready = command.copy()
            yield phase, command
    release = (('stabilize_lift_clear', high, 0., 30, False, left_down),
               ('stabilize_retreat', np.array([[.220, .260, .960]]), 1., 35, False, left_down))
    yield from _move(ready, ('left',), (0,), [station.BASES['left']], release)
