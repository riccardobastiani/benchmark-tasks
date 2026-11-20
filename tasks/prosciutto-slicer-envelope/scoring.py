"""Native geometry checks for the blade and food envelope."""

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import BLADE_CENTER, BLADE_HALF_THICKNESS, BLADE_RADIUS, ENVELOPE


def blade_zone_intersects(lower, upper):
    """Check each box against the blade's disk, with 2 mm clearance."""
    offset = np.maximum(lower - BLADE_CENTER, BLADE_CENTER - upper)
    radial = np.linalg.norm(np.maximum(offset[:, [0, 2]], 0.), axis=1)
    return bool(np.any((offset[:, 1] <= BLADE_HALF_THICKNESS + .002)
                       & (radial <= BLADE_RADIUS + .002)))


def slice_inside_envelope(lower, upper):
    # Inner wall edges, floor top, and underside of the flat flap.
    inner_min = ENVELOPE + [-.065, -.065, .002]
    inner_max = ENVELOPE + [.065, .058, .0245]
    tolerance = .0005
    return bool(np.all(lower >= inner_min - tolerance) and np.all(upper <= inner_max + tolerance))


class ArmBounds:
    """World bounds of each URDF collision hull, using native link transforms."""

    def __init__(self, base):
        from yam_arena.analysis import link_hulls

        self.arms = []
        hulls = link_hulls()
        for side in ('left', 'right'):
            robot = base.scene[side + '_robot']
            index, centers, extents = [], [], []
            for link, pieces in hulls.items():
                for piece in pieces:
                    lower, upper = piece.vertices.min(0), piece.vertices.max(0)
                    index.append(robot.body_names.index(link))
                    centers.append((lower + upper) / 2)
                    extents.append((upper - lower) / 2)
            self.arms.append((robot.root_view, np.asarray(index),
                              np.asarray(centers), np.asarray(extents), len(robot.body_names)))

    def world(self):
        lower, upper = [], []
        for view, index, local_center, local_extent, link_count in self.arms:
            poses = np.asarray(view.get_link_transforms().numpy()).reshape(link_count, 7)
            if not np.isfinite(poses).all() or np.any(abs(np.linalg.norm(poses[:, 3:], axis=1) - 1.) > .001):
                raise RuntimeError('Invalid native arm pose')
            rotation = Rotation.from_quat(poses[:, 3:]).as_matrix()[index]
            center = poses[index, :3] + (rotation @ local_center[..., None])[..., 0]
            extent = (abs(rotation) @ local_extent[..., None])[..., 0]
            lower.append(center - extent)
            upper.append(center + extent)
        return np.concatenate(lower), np.concatenate(upper)

    def clear_of_slice(self, lower, upper):
        arm_lower, arm_upper = self.world()
        separate = np.any((arm_upper < lower - .005) | (arm_lower > upper + .005), axis=1)
        return bool(np.all(separate))
