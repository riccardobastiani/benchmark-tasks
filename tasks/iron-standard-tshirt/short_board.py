"""A 480 × 320 mm travel ironing board with freely hanging garment ends."""

import numpy as np

from .board import BOARD_CENTER_X, BOARD_HALF_WIDTH, EDGE_RADIUS


SHORT_BOARD_HEIGHT = .13
CENTER = np.array([BOARD_CENTER_X, .045])
INNER_HALF_SIZE = np.array([BOARD_HALF_WIDTH, .24]) - EDGE_RADIUS


def rounded_deck(center, inner_half_size, top, bottom):
    corners = center + inner_half_size * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    corner_angles = np.linspace(0., np.pi / 2, 5)
    angles = np.arange(4)[:, None] * (np.pi / 2) + corner_angles
    outward = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    rings = [(EDGE_RADIUS, bottom)]
    rings.extend((EDGE_RADIUS*np.cos(a), top - EDGE_RADIUS + EDGE_RADIUS*np.sin(a))
                 for a in np.linspace(0., np.pi / 2, 7)[:-1])
    points = [(x, y, z) for radius, z in rings
              for x, y in (corners[:, None, :] + radius*outward).reshape(-1, 2)]
    ring_size = len(corners) * len(corner_angles)
    top_start = len(points)
    points.extend((x, y, top) for x, y in corners)
    faces = [list(reversed(range(ring_size))), list(range(top_start, len(points)))]
    for lower in range(0, top_start - ring_size, ring_size):
        for index in range(ring_size):
            following = (index + 1) % ring_size
            faces.append([lower + index, lower + following,
                          lower + ring_size + following, lower + ring_size + index])
    for index in range(ring_size):
        following = (index + 1) % ring_size
        corner = top_start + index // len(corner_angles)
        next_corner = top_start + following // len(corner_angles)
        face = [top_start - ring_size + index, top_start - ring_size + following, next_corner]
        if corner != next_corner:
            face.append(corner)
        faces.append(face)

    return dict(type='mesh', mesh=dict(points=points, faces=faces),
                color=(.38, .40, .38, 1.), friction=(.65, .55), contact_offset=.0005)


def short_board_shapes(table_top):
    top = table_top + SHORT_BOARD_HEIGHT
    bottom = top - .02
    deck = rounded_deck(CENTER, INNER_HALF_SIZE, top, bottom)
    leg_half_height = (bottom - table_top) / 2
    legs = [dict(type='box', size=(.09, .009, leg_half_height),
                 position=(BOARD_CENTER_X, y, table_top + leg_half_height),
                 color=(.71, .71, .70, 1.)) for y in (-.12, .22)]
    return [deck, *legs]


def drape_on_short_board(points, old_mat_top, table_top):
    """Wrap the rotated shirt around the deck edges without moving seated points off the top."""
    flat = np.column_stack((BOARD_CENTER_X + points[:, 1], .43 - points[:, 0]))
    supported = np.clip(flat, CENTER - INNER_HALF_SIZE, CENTER + INNER_HALF_SIZE)
    excess = flat - supported
    length = np.linalg.norm(excess, axis=1)
    direction = np.divide(excess, length[:, None], out=np.zeros_like(excess), where=length[:, None] > 0)
    # Free material leaves the rounded corners more gradually than a straight edge.
    corner = (2 * direction[:, 0] * direction[:, 1])**2
    slope = np.pi / 2 - np.deg2rad(40.) * corner
    # Re-seat off-mat edges and leave room for mesh chords across the rounded support.
    height = np.maximum(points[:, 2] - old_mat_top, 0.) + .002
    top_radius = EDGE_RADIUS + height
    bottom_radius = .034 - height

    top_arc = top_radius * slope
    top_turn = np.minimum(length / top_radius, slope)
    straight_limit = (SHORT_BOARD_HEIGHT - (top_radius + bottom_radius) * (1 - np.cos(slope))) / np.sin(slope)
    straight = np.clip(length - top_arc, 0., straight_limit)
    bottom_length = np.maximum(length - top_arc - straight, 0.)
    bottom_turn = np.minimum(bottom_length / bottom_radius, slope)
    tail = np.maximum(bottom_length - bottom_radius * slope, 0.)

    run = (top_radius * np.sin(top_turn) + straight * np.cos(slope)
           + bottom_radius * (np.sin(slope) - np.sin(slope - bottom_turn)) + tail)
    drop = (top_radius * (1 - np.cos(top_turn)) + straight * np.sin(slope)
            + bottom_radius * (np.cos(slope - bottom_turn) - np.cos(slope)))
    return np.column_stack((supported + direction * run[:, None],
                            table_top + SHORT_BOARD_HEIGHT + height - drop))
