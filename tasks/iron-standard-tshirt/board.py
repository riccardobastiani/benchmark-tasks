"""Raised 730 × 320 mm tabletop ironing board and a draped cloth reset."""

import numpy as np


BOARD_CENTER_X = 0.34
BOARD_HALF_WIDTH = 0.16
BOARD_HALF_LENGTH = 0.365
BOARD_HEIGHT = 0.10
EDGE_RADIUS = 0.012


def board_shapes(table_top):
    top = table_top + BOARD_HEIGHT
    width = BOARD_HALF_WIDTH
    radius = EDGE_RADIUS
    # Extrude the rounded cross section along the board length.
    angles = np.linspace(0., np.pi / 2, 7)
    right = [(width - radius + radius*np.cos(a), top - radius + radius*np.sin(a))
             for a in angles]
    outline = [(-width, top-.02), (width, top-.02), *right,
               *((-x, z) for x, z in reversed(right))]
    points = [(BOARD_CENTER_X+x, y, z) for y in (-BOARD_HALF_LENGTH, BOARD_HALF_LENGTH)
              for x, z in outline]
    n = len(outline)
    faces = [list(range(n)), list(reversed(range(n, 2*n)))]
    for i in range(n):
        j = (i + 1) % n
        faces.append([j, i, i+n, j+n])
    deck = dict(type='mesh', mesh=dict(points=points, faces=faces),
                color=(.38, .40, .38, 1.), friction=(.65, .55), contact_offset=.0005)
    legs = [dict(type='box', size=(.09, .009, .04), position=(BOARD_CENTER_X, y, table_top+.04),
                 color=(.71, .71, .70, 1.)) for y in (-.27, .27)]
    return [deck, *legs]


def drape_on_board(points, old_mat_top, table_top):
    """Rotate the shirt clockwise and wrap the overhang down each side."""
    height = points[:, 2] - old_mat_top
    across = points[:, 1]
    distance = np.abs(across)
    edge = BOARD_HALF_WIDTH - EDGE_RADIUS
    length = np.maximum(distance - edge, 0.)

    # Each layer follows its own turn radius, using the same arc length as the
    # flat fabric. A second turn lays excess sleeve material on the station.
    top_radius = EDGE_RADIUS + height
    top_arc = top_radius * (np.pi / 2)
    top_turn = np.minimum(length / top_radius, np.pi / 2)
    bottom_radius = .034 - height
    straight = BOARD_HEIGHT - EDGE_RADIUS - .034
    vertical = np.clip(length - top_arc, 0., straight)
    bottom_length = np.maximum(length - top_arc - straight, 0.)
    bottom_turn = np.minimum(bottom_length / bottom_radius, np.pi / 2)
    tail = np.maximum(bottom_length - bottom_radius*(np.pi / 2), 0.)

    width = (np.minimum(distance, edge) + top_radius*np.sin(top_turn)
             + bottom_radius*(1. - np.cos(bottom_turn)) + tail)
    z = (table_top + BOARD_HEIGHT + height - top_radius*(1. - np.cos(top_turn))
         - vertical - bottom_radius*np.sin(bottom_turn))
    return np.column_stack((BOARD_CENTER_X + np.sign(across)*width, .43 - points[:, 0], z))
