"""A guarded deli slicer, a one-portion cut proxy and a fold-over food envelope."""

import math
import numpy as np
from resist_isaac.recovery import body, box, description
from .assets import cylinder, rod

TABLE_Z = .7475
CARRIAGE_START = np.array([.29, -.02, TABLE_Z + .049])
CARRIAGE_CUT = np.array([.37, -.02, TABLE_Z + .049])
CARRIAGE_TRAVEL_LIMIT = .0735
HANDLE = np.array([0., -.130, .060])
HAM_OFFSET = np.array([.028, -.006, .039])
PORTION_OFFSET = np.array([.028, .0655, .039])
HAM_START = CARRIAGE_START + HAM_OFFSET
PORTION_START = CARRIAGE_START + PORTION_OFFSET
BLADE_CENTER = np.array([.451, .042, HAM_START[2]])
BLADE_RADIUS = .100
BLADE_HALF_THICKNESS = .001
BLADE_GUARD_RADIUS = .116
GUARD_NORMAL_OFFSET = -.008
GUARD_ROD_RADIUS = .007
HAM_REAR, HAM_FRONT = -.071, .065
HAM_COLLIDER_CENTER = np.array([0., (HAM_FRONT + HAM_REAR)/2, 0.])
HAM_HALF_SIZE = np.array([.032, (HAM_FRONT - HAM_REAR)/2, .032])
PORTION_HALF_SIZE = np.array([.032, .0017, .030])
ENVELOPE = np.array([.21, .18, TABLE_Z + .004])
FLAP_HINGE = ENVELOPE + np.array([0., .058, .026])
POWER_OFF = np.array([.224, -.235, TABLE_Z + .105])
POWER_ON = POWER_OFF + [.034, 0., 0.]


def _arc_xz(y, cx, cz, radius, start_deg, end_deg, thickness, color):
    shapes = []
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), 33)
    for a, b in zip(angles[:-1], angles[1:]):
        p0 = [cx + radius * math.cos(a), y, cz + radius * math.sin(a)]
        p1 = [cx + radius * math.cos(b), y, cz + radius * math.sin(b)]
        shapes.append(rod(p0, p1, thickness, color, collision=False))
    return shapes


def _cotto_skin(stations, y_radius, z_radius):
    """Rounded cooked ham with a lighter edge and pink cut faces."""
    theta = np.arange(48) * 2 * math.pi / 48
    c, s = np.cos(theta), np.sin(theta)
    fullness = 1. - .035*np.sin(3*theta) + .025*np.cos(5*theta)
    profile = np.column_stack((y_radius * np.sign(c) * abs(c)**.81 * fullness,
                               z_radius * np.sign(s) * abs(s)**.89 * fullness))
    n = len(theta)

    def loop(x, scale):
        return [[-y*scale, x, z*scale] for y, z in profile]

    def mesh(points, faces, color):
        return dict(type='mesh', collision=False, color=color, roughness=.62,
                    mesh=dict(points=points, faces=faces))

    def ring_faces(start):
        return [[start+i, start+(i+1)%n, start+n+(i+1)%n, start+n+i] for i in range(n)]

    skin = [mesh([p for x, scale in stations for p in loop(x, scale)],
                 [f for row in range(len(stations)-1) for f in ring_faces(row*n)],
                 [.70, .37, .33, 1.])]
    for sign, (x, scale) in ((-1, stations[0]), (1, stations[-1])):
        center = loop(x, scale*.92)
        edge = loop(x, scale)
        faces = ring_faces(0)
        if sign < 0:
            faces = [f[::-1] for f in faces]
        skin.append(mesh(edge+center, faces, [.94, .72, .58, 1.]))
        face = list(range(n))
        skin.append(mesh(center, [face if sign > 0 else face[::-1]], [.84, .43, .49, 1.]))
    return skin


def geometry():
    blade_center = BLADE_CENTER
    blade_marks = dict(type='mesh', collision=False, blade_spin=True, roughness=.4,
                       color=[.52,.56,.60,1.], position=(blade_center+[0., .0011, 0.]).tolist(),
                       mesh=dict(points=[[.079,0.,.002],[.091,0.,.002],[.091,0.,-.002],[.079,0.,-.002],
                                         [-.048,0.,.063],[-.055,0.,.073],[-.052,0.,.075],[-.045,0.,.065]],
                                 faces=[[0,1,2,3],[4,5,6,7]]))
    machine_shapes = [
        box((.22, .112, .018), (.41, -.055, TABLE_Z + .022), (.24, .27, .29, 1)),
        # Motor housing sits behind the product path, leaving the sliding
        # carriage clear of static machine collision geometry.
        box((.055, .047, .049), (.50, .115, TABLE_Z + .058), (.13, .16, .17, 1)),
        # A continuous bed supports the carriage throughout its full stroke.
        box((.120, .091, .004), (.35, -.043, TABLE_Z + .038), (.61, .63, .64, 1)),
        cylinder(BLADE_RADIUS, BLADE_HALF_THICKNESS, blade_center, (.77, .79, .80, 1),
                 orientation=[math.sqrt(.5), math.sqrt(.5), 0., 0.], collision=False),
        cylinder(.014, .0004, blade_center + np.array([0., .0012, 0.]),
                 (.36, .39, .41, 1), orientation=[math.sqrt(.5), math.sqrt(.5), 0., 0.], collision=False),
        blade_marks,
        # A high guard covers the rear/top of the blade. The front cutting arc
        # remains visible while the robot stays on the carriage handle.
        *_arc_xz(blade_center[1] + GUARD_NORMAL_OFFSET, blade_center[0], blade_center[2], .109,
                 -65., 110., GUARD_ROD_RADIUS, (.19, .22, .24, 1)),
        rod((.28, -.112, TABLE_Z + .049), (.47, -.112, TABLE_Z + .049), .006, (.32, .36, .38, 1), collision=False),
        rod((.28, .072, TABLE_Z + .049), (.47, .072, TABLE_Z + .049), .006, (.32, .36, .38, 1), collision=False),
        # The shallow receiving slot leaves the slice's upper edge exposed.
        box((.051, .030, .004), (.416, .083, TABLE_Z + .032), (.72, .75, .76, 1)),
        box((.048, .0015, .004), (.416, .062, TABLE_Z + .035), (.44, .47, .48, 1)),
        box((.048, .002, .014), (.416, .076, TABLE_Z + .050), (.44, .47, .48, 1)),
        # Put the controls on a low front-left pedestal. The earlier elevated
        # panel overlapped the carriage handle's approach corridor and blocked
        # the robot's grasp.
        box((.045, .025, .049), (.241, -.24, TABLE_Z + .049), (.20, .23, .25, 1)),
        box((.045, .025, .005), (.241, -.24, TABLE_Z + .097), (.28, .30, .32, 1)),
        cylinder(.011, .004, POWER_OFF, (.82, .12, .10, 1)),
        cylinder(.011, .004, POWER_ON, (.13, .66, .24, 1)),
    ]
    carriage_shapes = [
        box((.092, .067, .006), (0., -.008, 0.), (.72, .74, .75, 1)),
        box((.065, .055, .002), (.005, 0., .008), (.82, .83, .83, 1), collision=False),
        box((.013, .033, .005), (0., -.102, .002), (.24, .27, .29, 1)),
        rod(HANDLE-[0., 0., .056], HANDLE-[0., 0., .005], .010, (.24, .27, .29, 1)),
        cylinder(.016, .009, HANDLE, (.29, .32, .34, 1)),
        # Pusher plate fixed to the carriage holds the remaining ham safely.
        box((.006, .050, .038), (-.049, 0., .045), (.83, .84, .84, 1), collision=False),
        box((.008, .050, .006), (-.042, 0., .085), (.72, .74, .75, 1)),
    ]
    carriage = body('carriage', carriage_shapes, .85, CARRIAGE_START)

    ham_profile = [(HAM_REAR, .68), (-.060, .92), (-.039, 1.), (.044, 1.)]
    whole_tip = PORTION_OFFSET[1] - HAM_OFFSET[1] + PORTION_HALF_SIZE[1]
    ham_shapes = [box(HAM_HALF_SIZE, HAM_COLLIDER_CENTER, visible=False),
                  *[dict(s, cut_visual='whole') for s in _cotto_skin(
                      ham_profile + [(whole_tip, 1.)], .031, .031)],
                  *[dict(s, cut_visual='sliced', visible=False) for s in _cotto_skin(
                      ham_profile + [(HAM_FRONT, 1.)], .031, .031)]]
    ham = body('ham', ham_shapes, .32, HAM_START)

    # The 3.4 mm strip shares the whole piece's silhouette until the blade
    # edge sweeps its cross-section. The released part settles in PhysX.
    portion_shapes = [
        box(PORTION_HALF_SIZE, visible=False),
        *[dict(s, cut_visual='sliced', visible=False) for s in _cotto_skin(
            [(-PORTION_HALF_SIZE[1], 1.), (PORTION_HALF_SIZE[1], 1.)], .031, .029)],
    ]
    portion = body('portion', portion_shapes, .019, PORTION_START)
    machine_body = body('machine', machine_shapes, static=True)
    workpiece = description('DeliSlicerWorkpiece', [machine_body, carriage, ham, portion], joints=[
        # PhysX constrains the carriage to the slicer's real one-axis rail.
        dict(name='carriage_rail', type='prismatic', parent='machine', child='carriage', axis='X',
             position_parent=CARRIAGE_START.tolist(), position_child=[0., 0., 0.],
             lower=-.005, upper=CARRIAGE_TRAVEL_LIMIT, damping=5., collision=False),
        dict(name='ham_mount', type='fixed', parent='carriage', child='ham',
             position_parent=HAM_OFFSET.tolist(), position_child=[0., 0., 0.], collision=False),
        dict(name='cut_link', type='fixed', parent='carriage', child='portion',
             position_parent=PORTION_OFFSET.tolist(), position_child=[0., 0., 0.], collision=False),
    ], solver_position_iterations=48, solver_velocity_iterations=12)

    pocket_shapes = [
        box((.071, .068, .002), (0., -.007, 0.), (.91, .89, .81, 1)),
        box((.003, .064, .010), (-.068, -.006, .010), (.91, .89, .81, 1)),
        box((.003, .064, .010), (.068, -.006, .010), (.91, .89, .81, 1)),
        box((.066, .003, .005), (0., -.068, .005), (.91, .89, .81, 1)),
        box((.023, .006, .001), (0., 0., .0025), (.72, .20, .16, 1), collision=False),
    ]
    pocket = body('pocket', pocket_shapes, static=True, position=ENVELOPE)
    flap_shapes = [
        # Open upright, hinged along the rear edge; a quarter turn lays it
        # across the envelope mouth instead of sliding it along the tabletop.
        box((.071, .0015, .062), (0., 0., .062), (.92, .90, .83, 1)),
        box((.024, .010, .014), (0., 0., .116), (.77, .28, .21, 1)),
    ]
    flap = body('flap', flap_shapes, .018, FLAP_HINGE)
    envelope = description('FoodPaperEnvelope', [pocket, flap], joints=[
        dict(name='fold_hinge', type='revolute', parent='pocket', child='flap', axis='X',
             position_parent=[0., .058, .026], position_child=[0., 0., 0.],
             lower=0., upper=math.pi/2, stiffness=0., damping=.02, collision=False),
    ], solver_position_iterations=48, solver_velocity_iterations=12)
    return {'workpiece': workpiece, 'envelope': envelope}


def side_frame(side):
    a = math.radians(25)
    z, x = np.array([math.cos(a), 0., -math.sin(a)]), np.array([0., -1., 0.])
    return np.column_stack((x, np.cross(z, x), z))


def ready_pose(side):
    from yam_arena import ik, station
    from yam_arena.constants import READY
    return ik.tcp_pose(READY, station.BASES[side])
