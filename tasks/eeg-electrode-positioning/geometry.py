"""Procedural, fixed EEG training-head and electrode geometry."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from yam_arena.constants import TABLE_HEIGHT

from resist_isaac.recovery import body, box


FOLDER = Path(__file__).parent
TABLE_TOP = float(TABLE_HEIGHT)
HEAD_CENTER = np.array([0.44, 0.0, TABLE_TOP + 0.10])
# This enlarged robotic training head gives the YAM gripper more lateral room
# while keeping the frontal sites at the previously reachable x coordinates.
HEAD_RADII = np.array([0.12, 0.1275, 0.094])
PUCK_RADIUS = 0.005
PUCK_HALF_HEIGHT = 0.0015
GRIP_CENTER_HEIGHT = 0.010
GEL_THICKNESS = 0.002
PAD_TIP_PAST_TCP = 0.013

SITES = (
    ("Fp1", "red",    (0.88, 0.12, 0.14, 1.0), (-0.0624, 0.0320)),
    ("Fp2", "blue",   (0.10, 0.32, 0.86, 1.0), (-0.0624, -0.0320)),
    ("C3",  "yellow", (0.96, 0.72, 0.08, 1.0), (0.0000, 0.0900)),
    ("C4",  "green",  (0.10, 0.62, 0.30, 1.0), (0.0000, -0.0900)),
    # Posterior centers are laterally separated and calibrated to the measured
    # released-cap positions at the YAM reach used by this prototype.
    ("O1",  "white",  (0.92, 0.94, 0.94, 1.0), (0.0624, 0.0478)),
    ("O2",  "black",  (0.13, 0.15, 0.18, 1.0), (0.0624, -0.0478)),
)
ORDER = tuple(site for site, *_ in SITES)

# Three pickups per side: left arm handles Fp1/C3/O1; right handles Fp2/C4/O2.
STARTS = {
    "Fp1": (0.245, 0.190, TABLE_TOP + 0.004 + PUCK_HALF_HEIGHT + GEL_THICKNESS),
    "C3":  (0.310, 0.190, TABLE_TOP + 0.004 + PUCK_HALF_HEIGHT + GEL_THICKNESS),
    "O1":  (0.375, 0.190, TABLE_TOP + 0.004 + PUCK_HALF_HEIGHT + GEL_THICKNESS),
    "Fp2": (0.245, -0.190, TABLE_TOP + 0.004 + PUCK_HALF_HEIGHT + GEL_THICKNESS),
    "C4":  (0.310, -0.190, TABLE_TOP + 0.004 + PUCK_HALF_HEIGHT + GEL_THICKNESS),
    "O2":  (0.375, -0.190, TABLE_TOP + 0.004 + PUCK_HALF_HEIGHT + GEL_THICKNESS),
}


def surface_z(dx: float, dy: float) -> float:
    radial = 1.0 - (dx / HEAD_RADII[0]) ** 2 - (dy / HEAD_RADII[1]) ** 2
    if radial <= 0:
        raise ValueError("EEG site falls outside the authored head")
    return float(HEAD_CENTER[2] + HEAD_RADII[2] * math.sqrt(radial))


def site_position(site: str) -> np.ndarray:
    for name, _, _, (dx, dy) in SITES:
        if name == site:
            return np.array([HEAD_CENTER[0] + dx, dy, surface_z(dx, dy)])
    raise KeyError(site)


def target_position(site: str) -> np.ndarray:
    point = site_position(site)
    point[2] += GEL_THICKNESS + PUCK_HALF_HEIGHT
    return point


def _head_collision_mesh():
    """Closed ellipsoid mesh so PhysX preserves the visible head dimensions."""
    rings, sectors = 24, 48
    points = [HEAD_CENTER + np.array([0.0, 0.0, HEAD_RADII[2]])]
    for ring in range(1, rings):
        theta = math.pi * ring / rings
        for sector in range(sectors):
            phi = 2 * math.pi * sector / sectors
            points.append(HEAD_CENTER + np.array([
                HEAD_RADII[0] * math.sin(theta) * math.cos(phi),
                HEAD_RADII[1] * math.sin(theta) * math.sin(phi),
                HEAD_RADII[2] * math.cos(theta),
            ]))
    bottom = len(points)
    points.append(HEAD_CENTER + np.array([0.0, 0.0, -HEAD_RADII[2]]))
    faces = []
    for sector in range(sectors):
        nxt = (sector + 1) % sectors
        faces.append([0, 1 + sector, 1 + nxt])
    for ring in range(1, rings - 1):
        a0, b0 = 1 + (ring - 1) * sectors, 1 + ring * sectors
        for sector in range(sectors):
            nxt = (sector + 1) % sectors
            faces.append([a0 + sector, b0 + sector, b0 + nxt, a0 + nxt])
    last = 1 + (rings - 2) * sectors
    for sector in range(sectors):
        nxt = (sector + 1) % sectors
        faces.append([last + sector, bottom, last + nxt])
    return {"points": [p.tolist() for p in points], "faces": faces}


def _segment_shapes(character: str, centre_x: float, centre_y: float, z: float,
                    color=(0.08, 0.10, 0.13, 1.0)):
    """Small raised seven-segment glyphs for a readable physical work card."""
    glyph = {
        "F": "afge", "P": "abefg", "C": "adef", "O": "abcdef",
        "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
        "5": "afgcd", "6": "afgecd",
    }[character]
    width, height, thick = 0.0050, 0.0105, 0.0012
    dx = width / 2 - thick / 2
    dy = height / 2 - thick / 2
    segments = {
        "a": ((0, dy, 0), (width / 2, thick / 2, 0.00045)),
        "g": ((0, 0, 0), (width / 2, thick / 2, 0.00045)),
        "d": ((0, -dy, 0), (width / 2, thick / 2, 0.00045)),
        "f": ((-dx, dy / 2, 0), (thick / 2, height / 4, 0.00045)),
        "b": ((dx, dy / 2, 0), (thick / 2, height / 4, 0.00045)),
        "e": ((-dx, -dy / 2, 0), (thick / 2, height / 4, 0.00045)),
        "c": ((dx, -dy / 2, 0), (thick / 2, height / 4, 0.00045)),
    }
    shapes = []
    for segment in glyph:
        (sx, sy, sz), size = segments[segment]
        shapes.append(box(size, (centre_x + sx, centre_y + sy, z + sz), color))
    return shapes


def geometry():
    """Return task-owned assets and world-space initial poses."""
    fixture_shapes = [
        dict(type="cylinder", size=[0.039, 0.004], position=[HEAD_CENTER[0], 0, TABLE_TOP + 0.004],
             color=[0.27, 0.30, 0.33, 1.0]),
        dict(type="cylinder", size=[0.030, 0.034], position=[HEAD_CENTER[0], 0, TABLE_TOP + 0.042],
             color=[0.56, 0.58, 0.60, 1.0]),
        dict(type="cylinder", size=[0.055, 0.003], position=[HEAD_CENTER[0], 0, TABLE_TOP + 0.079],
             color=[0.62, 0.64, 0.65, 1.0]),
        dict(type="cylinder", size=[0.054, 0.004], position=[HEAD_CENTER[0], 0, TABLE_TOP + 0.086],
             color=[0.74, 0.54, 0.40, 1.0]),
        dict(type="sphere", size=[1.0], scale=HEAD_RADII.tolist(), position=HEAD_CENTER.tolist(),
             color=[0.73, 0.53, 0.40, 1.0], roughness=0.82, collision=False),
        dict(type="mesh", mesh=_head_collision_mesh(), color=[0.73, 0.53, 0.40, 1.0],
             visible=False, collision=True, collision_approximation="convexHull"),
        dict(type="sphere", size=[1.0], scale=[0.019, 0.0165, 0.027],
             position=(HEAD_CENTER + [0.006, 0.129, 0.008]).tolist(), color=[0.69, 0.48, 0.37, 1.0]),
        dict(type="sphere", size=[1.0], scale=[0.019, 0.0165, 0.027],
             position=(HEAD_CENTER + [0.006, -0.129, 0.008]).tolist(), color=[0.69, 0.48, 0.37, 1.0]),
        dict(type="sphere", size=[1.0], scale=[0.0144, 0.010, 0.015],
             position=(HEAD_CENTER + [-0.1092, 0, -0.007]).tolist(), color=[0.74, 0.54, 0.41, 1.0]),
        dict(type="sphere", size=[1.0], scale=[0.004, 0.004, 0.004],
             position=(HEAD_CENTER + [-0.096, 0.039, -0.020]).tolist(), color=[0.10, 0.08, 0.07, 1.0]),
        dict(type="sphere", size=[1.0], scale=[0.004, 0.004, 0.004],
             position=(HEAD_CENTER + [-0.096, -0.039, -0.020]).tolist(), color=[0.10, 0.08, 0.07, 1.0]),
    ]

    # Two shallow electrode staging mats.
    for y in (0.190, -0.190):
        fixture_shapes.append(box((0.112, 0.054, 0.002), (0.310, y, TABLE_TOP + 0.002),
                                  (0.78, 0.80, 0.78, 1.0), friction=[0.65, 0.55]))
        fixture_shapes.append(box((0.112, 0.002, 0.002), (0.310, y + (0.052 if y > 0 else -0.052),
                                                               TABLE_TOP + 0.004),
                                  (0.31, 0.36, 0.39, 1.0)))

    # A compact printed-style card: numbered order, site code and matching color chip.
    card_x, card_y, card_z = 0.600, 0.205, TABLE_TOP + 0.004
    fixture_shapes.append(box((0.086, 0.078, 0.002), (card_x, card_y, card_z),
                              (0.94, 0.94, 0.91, 1.0)))
    for index, (site, _, color, _) in enumerate(SITES):
        row_y = card_y + 0.058 - index * 0.021
        fixture_shapes.append(box((0.006, 0.006, 0.001), (card_x - 0.064, row_y, card_z + 0.002), color))
        x0 = card_x - 0.046
        for char_index, character in enumerate(site.upper()):
            fixture_shapes.extend(_segment_shapes(character, x0 + char_index * 0.011,
                                                 row_y, card_z + 0.002))
        fixture_shapes.extend(_segment_shapes(str(index + 1), card_x + 0.060, row_y, card_z + 0.002))

    assets = {
        "fixture": {"name": "EEGTrainingHead", "bodies": [{
            "name": "fixture", "position": [0, 0, 0], "mass": 0.0, "static": True,
            "shapes": fixture_shapes,
        }]},
    }
    poses = {"fixture": [0.0, 0.0, 0.0]}
    for site, color_name, color, _ in SITES:
        lower = site.lower()
        shapes = [
            # Conductive paste travels with the cup; nothing is pre-mounted on
            # the scalp. This rigid, compliant-looking pad is a visual proxy.
            dict(type="cylinder", size=[0.006, GEL_THICKNESS / 2],
                 position=[0, 0, -PUCK_HALF_HEIGHT - GEL_THICKNESS / 2], color=[0.90, 0.87, 0.78, 1.0],
                 roughness=0.18, friction=[1.0, 0.9]),
            dict(type="cylinder", size=[PUCK_RADIUS, PUCK_HALF_HEIGHT],
                 color=[0.76, 0.76, 0.72, 1.0], roughness=0.23, friction=[1.0, 0.9]),
            dict(type="cylinder", size=[0.0038, 0.0005], position=[0, 0, 0.0018],
                 color=list(color), friction=[0.8, 0.7]),
            # Insulated connector/grasp tab, enlarged for the YAM fingertips.
            dict(type="cylinder", size=[0.0032, 0.006], position=[0, 0, 0.008],
                 color=list(color), friction=[1.0, 0.9]),
            dict(type="cylinder", size=[0.004, 0.001], position=[0, 0, 0.014],
                 color=list(color), friction=[1.0, 0.9]),
            dict(type="capsule", size=[0.0009, 0.014],
                 fromto=[-0.004, 0.0, 0.002, -0.032, 0.0, 0.002],
                 color=list(color), collision=False),
        ]
        electrode_body = body("body", shapes, mass=0.004)
        electrode_body.update(linear_damping=1.5, angular_damping=4.0)
        assets["electrode_" + lower] = {
            "name": "Electrode_" + site,
            "bodies": [electrode_body],
            "solver_position_iterations": 64,
            "solver_velocity_iterations": 16,
        }
        poses["electrode_" + lower] = STARTS[site]
    return assets, poses

