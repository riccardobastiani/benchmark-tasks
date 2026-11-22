"""A full-size padded tabletop pressing surface with an accessible front lip."""

import numpy as np

from .board import EDGE_RADIUS
from .short_board import rounded_deck


PRESS_HEIGHT = .100
PRESS_CENTER = np.array([.555, -.025])
PRESS_HALF_SIZE = np.array([.390, .450])


def pressing_table_shapes(table_top):
    top = table_top + PRESS_HEIGHT
    bottom = top - .016
    deck = rounded_deck(PRESS_CENTER, PRESS_HALF_SIZE - EDGE_RADIUS, top, bottom)
    foot_height = (bottom - table_top)/2
    skids = [dict(type='box', size=(.024, .355, foot_height),
                  position=(x, PRESS_CENTER[1], table_top + foot_height), color=(.58, .57, .55, 1.))
             for x in (.33, .73)]
    return [deck, *skids]


def drape_front_lip(points, mat_top, table_top):
    """Set down the tee with a graspable near seam and its torso on the padded deck."""
    x = PRESS_CENTER[0] - .171 - points[:, 1]
    y = points[:, 0] - .550
    front_tangent = PRESS_CENTER[0] - PRESS_HALF_SIZE[0] + EDGE_RADIUS
    overhang = np.maximum(front_tangent - x, 0.)
    layer = np.maximum(points[:, 2] - mat_top, 0.) + .0015
    lip_radius = EDGE_RADIUS + layer
    deck_turn = np.minimum(overhang/lip_radius, np.pi/2)
    beyond_lip = np.maximum(overhang - lip_radius*np.pi/2, 0.)
    foot_radius = .014 - layer
    vertical = PRESS_HEIGHT - EDGE_RADIUS - .014
    fall = np.minimum(beyond_lip, vertical)
    foot_turn = np.minimum(np.maximum(beyond_lip - vertical, 0.)/foot_radius, np.pi/2)
    tail = np.maximum(beyond_lip - vertical - foot_radius*np.pi/2, 0.)
    reach = lip_radius*np.sin(deck_turn) + foot_radius*(1. - np.cos(foot_turn)) + tail
    drop = lip_radius*(1. - np.cos(deck_turn)) + fall + foot_radius*np.sin(foot_turn)
    return np.column_stack((np.maximum(x, front_tangent) - reach, y,
                            table_top + PRESS_HEIGHT + layer - drop))
