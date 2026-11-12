"""Flexible wire, passive clips and an open two-post routing gate."""
import math
from pathlib import Path

import numpy as np
from resist_isaac.recovery import body, box, description

FOLDER = Path(__file__).parent
TABLE_Z = .7475
PLATE_Z = TABLE_Z + .008
WIRE_RADIUS = .004
WIRE_Z = PLATE_Z + .0087
OFFSET = np.array([-.12, .08, 0.])
UPSTREAM = np.array([.32, .145, PLATE_Z])+OFFSET
TARGET = np.array([.49, -.13, PLATE_Z])+OFFSET
GATE_X = .320
POST_Y = (0., .100)
POST_RADIUS = .010
POST_HEIGHT = .055
GATE_LOW = POST_Y[0]+POST_RADIUS
GATE_HIGH = POST_Y[1]-POST_RADIUS
GATE_WIRE_HEIGHT = POST_HEIGHT-WIRE_RADIUS
JAW_SPAN = .0065
JAW_TRAVEL = .004
JAW_STIFFNESS = 650.
JAW_DAMPING = 1.0
WIRE_STIFFNESS = .0004
WIRE_DAMPING = .00003


def wire_nodes():
    """A reproducible slack loop, resampled at approximately 12 mm spacing."""
    controls = np.array([
        [.32, .235], [.32, .18], [.32, .13], [.31, .105],
        [.275, .085], [.255, .04], [.275, -.025],
        [.335, -.075], [.395, -.095], [.43, -.13], [.43, -.19],
    ])
    padded = np.vstack([controls[0], controls, controls[-1]])
    dense = []
    for a, b, c, d in zip(padded, padded[1:], padded[2:], padded[3:]):
        for t in np.linspace(0, 1, 30, endpoint=False):
            dense.append(.5 * ((2*b) + (-a+c)*t
                         + (2*a-5*b+4*c-d)*t*t + (-a+3*b-3*c+d)*t*t*t))
    dense = np.vstack([dense, controls[-1]])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    samples = np.linspace(0., arc[-1], math.ceil(arc[-1]/.012)+1)
    xy = np.column_stack([np.interp(samples, arc, dense[:, i]) for i in range(2)])
    return np.column_stack([xy, np.full(len(xy), WIRE_Z)])+OFFSET


def wire_geometry():
    nodes = wire_nodes()
    parts, joints = [], []
    for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        name = f'link{i:02d}'
        length = float(np.linalg.norm(b-a))
        shape = dict(type='capsule', size=[WIRE_RADIUS, length/2],
                     color=[1., .36, .035, 1.], friction=[.6, .45], contact_offset=.0001)
        part = body(name, [shape], mass=.16*length, position=(a+b)/2)
        tangent = (b-a)/length
        q = np.r_[1+tangent[2], -tangent[1], tangent[0], 0.]
        part['orientation'] = (q/np.linalg.norm(q)).tolist()
        parts.append(part)
        if i:
            previous_length = float(np.linalg.norm(nodes[i]-nodes[i-1]))
            joints.append(dict(name=f'bend{i:02d}', type='spherical',
                parent=f'link{i-1:02d}', child=name,
                position_parent=[0, 0, previous_length/2], position_child=[0, 0, -length/2],
                locked_axes=['rotZ'], angular_limit=math.radians(65),
                stiffness=WIRE_STIFFNESS, damping=WIRE_DAMPING, collision=False))
    return description('Wire', parts, joints=joints,
                       solver_position_iterations=64, solver_velocity_iterations=16)


def clip_geometry(name, centre, color):
    """Two passive spring jaws with rounded, narrower-than-wire retaining lips."""
    x, y, z = centre
    parts = [body('housing', [box((.022, .022, .0015), (x, y, z+.0015), color),
             box((.0047, .015, .0015), (x, y, z+.003), (.14, .16, .18, 1))], static=True)]
    joints = []
    for sign, side in [(-1, 'left'), (1, 'right')]:
        shapes = [box((.0015, .014, .0065), (0, 0, .0065), (.16, .18, .20, 1)),
            dict(type='capsule', size=[.0022], fromto=[-sign*.0025, -.012, .013,
                -sign*.0025, .012, .013], color=[.22, .24, .26, 1.],
                friction=[.55, .4], contact_offset=.0001)]
        position = [x+sign*JAW_SPAN, y, z+.003]
        parts.append(body(side, shapes, .006, position))
        joints.append(dict(name=side+'_slide', type='prismatic', child=side, axis='X',
            position_parent=position, lower=-JAW_TRAVEL if sign<0 else 0.,
            upper=0. if sign<0 else JAW_TRAVEL, stiffness=JAW_STIFFNESS,
            damping=JAW_DAMPING, target=-sign*.0003, collision=False))
    return description(name, parts, joints=joints,
                       solver_position_iterations=64, solver_velocity_iterations=16)


def geometry():
    fixture = [box((.20, .255, .004), (.270, .080, TABLE_Z+.004), (.23, .26, .29, 1)),
        # Painted passage: visible goal geometry, without an additional collider.
        box((.018, (GATE_HIGH-GATE_LOW)/2, .00006),
            (GATE_X, (GATE_LOW+GATE_HIGH)/2, PLATE_Z+.00007),
            (.78, .86, .90, 1.), collision=False)]
    for y in POST_Y:
        fixture.append(dict(type='cylinder', size=[POST_RADIUS, POST_HEIGHT/2],
            position=[GATE_X, y, PLATE_Z+POST_HEIGHT/2], color=[.52, .55, .59, 1.],
            roughness=.25, contact_offset=.0001))
    return dict(fixture=description('Fixture', [body('plate', fixture, static=True)]),
        wire=wire_geometry(),
        upstream=clip_geometry('UpstreamClip', UPSTREAM, (.10, .48, .24, 1)),
        target=clip_geometry('TargetClip', TARGET, (.08, .38, .70, 1)))
