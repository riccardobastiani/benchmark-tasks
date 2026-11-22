"""Measure cloth under the native iron soleplate, without inferring force."""

import numpy as np


OUTLINE = ((-.125, -.046), (-.108, -.059), (-.041, -.062), (.022, -.051),
           (.095, -.027), (.132, 0.), (.095, .027), (.022, .051),
           (-.041, .062), (-.108, .059), (-.125, .046))


def _rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or abs(norm - 1.) > .001:
        raise ValueError('Expected the native unit quaternion in XYZW order')
    x, y, z, w = q/norm
    return np.array([
        [1.-2.*(y*y+z*z), 2.*(x*y-z*w), 2.*(x*z+y*w)],
        [2.*(x*y+z*w), 1.-2.*(x*x+z*z), 2.*(y*z-x*w)],
        [2.*(x*z-y*w), 2.*(y*z+x*w), 1.-2.*(x*x+y*y)],
    ])


def _soleplate_frame(points, position, quaternion):
    rotation = _rotation(quaternion)
    return np.einsum('ni,ij->nj', points - position, rotation), rotation


def _inside_soleplate(xy):
    polygon = np.asarray(OUTLINE)
    edges = np.roll(polygon, -1, axis=0) - polygon
    delta = xy[:, None] - polygon
    signed = edges[:, 0]*delta[..., 1] - edges[:, 1]*delta[..., 0]
    return np.all(signed >= 0., axis=1)


def measure_soleplate(points, position, quaternion):
    local, rotation = _soleplate_frame(points, position, quaternion)
    inside = _inside_soleplate(local[:, :2])
    gap = -local[inside, 2]
    return dict(soleplate_tilt_deg=float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1., 1.)))),
                nodes_under_soleplate=int(len(gap)),
                minimum_gap_m=float(gap.min()) if len(gap) else None,
                nodes_near_soleplate=int(np.count_nonzero((-0.0005 <= gap) & (gap <= .0015))),
                nodes_above_soleplate=int(np.count_nonzero(gap < -0.0005)))


def measure_upward_patch(points, panels, position, quaternion):
    """Return projected areas and material triangles facing the sole."""
    local, _ = _soleplate_frame(points, position, quaternion)
    front, back = (local[face] for face in panels)
    upper = front.mean(axis=1)[:, 2] > back.mean(axis=1)[:, 2]
    areas, masks = [], []
    for triangles, exposed in ((front, upper), (back, ~upper)):
        centers = triangles.mean(axis=1)
        n = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        nearly_parallel = n[:, 2] >= np.cos(np.deg2rad(25.)) * np.linalg.norm(n, axis=1)
        under = _inside_soleplate(centers[:, :2]) & (-.0015 <= centers[:, 2]) & (centers[:, 2] <= .0005)
        mask = exposed & under & nearly_parallel
        masks.append(mask)
        areas.append(float(n[mask, 2].sum() / 2.))
    return dict(zip(('front_patch_m2', 'back_patch_m2'), areas, strict=True)), masks


def measure_rest(position, quaternion, rest, half_extents):
    """Measure the actual sole footprint and height relative to the silicone pad."""
    rotation = _rotation(quaternion)
    sole = np.c_[OUTLINE, np.zeros(len(OUTLINE))] @ rotation.T + position
    clearance = half_extents - np.abs(sole[:, :2] - rest[:2])
    return dict(iron_rest_xy_clearance_mm=float(clearance.min() * 1000.),
                iron_rest_lowest_gap_mm=float((sole[:, 2].min()-rest[2]) * 1000.),
                iron_rest_tilt_deg=float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1., 1.)))))
