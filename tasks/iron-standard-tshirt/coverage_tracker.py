"""Accumulate physically supported contact on each exposed material face."""

from collections import deque

import numpy as np

from .soleplate import _rotation, measure_upward_patch


HANDLE_CENTER = np.array([-.006, 0., .100])
HANDLE_TCP_TOLERANCE = np.array([.060, .006, .006])
# The 28 mm handle holds the 96 mm-aperture jaws at 0.297 in the native runs.
GRASP_OPENING = (.26, .34)
# Runs 031/032 measured 6.0–7.6 N in the press and 16.5–17.1 N in the hover.
MAX_PRESS_WRIST_N = 12.
# The settled press covered at least 177 cm² per sample and the hover peaked at 1 cm².
MIN_PATCH_M2 = .004
MIN_FACE_UP = .90
MAX_SOLE_TILT_DEG = 5.
CONTACT_CONTROLS = 3
SIDES = ('front', 'back')


class CoverageTracker:
    """Track unique material area at 20 Hz. Construct a fresh tracker for each trial."""

    def __init__(self, rest, panels, *, crease_span=(.266, .3095)):
        self._panels = tuple(panels)
        if len(self._panels) != 2 or self._panels[0].shape != self._panels[1].shape:
            raise ValueError('Expected paired front and back material triangles')
        self._weights, self._target = [], []
        for panel in self._panels:
            triangles = rest[panel]
            a, b, c = triangles.transpose(1, 0, 2)
            self._weights.append(.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1))
            x, y = triangles.mean(axis=1)[:, :2].T
            self._target.append((crease_span[0] <= x) & (x <= crease_span[1]) & (-.22 <= y) & (y <= -.02))
        self._area = [weights.sum() for weights in self._weights]
        self._crease_area = [weights[target].sum() for weights, target in zip(self._weights, self._target, strict=True)]
        if any(area <= 0 for area in self._crease_area):
            raise ValueError('The crease band must contain material on each face')
        self._visited = [np.zeros(len(panel), dtype=bool) for panel in self._panels]
        self._loaded_controls = [0, 0]
        self._step = -1
        self._contact = None
        self._recent = deque(maxlen=CONTACT_CONTROLS)

    def observe(self, step, points, *, tcp_m, opening, wrist_force_z_N,
                iron_position_m, iron_orientation_native):
        """Consume one completed control's native measurements. Jaw opening is normalized."""
        measurements = [points, iron_position_m, iron_orientation_native,
                        *tcp_m.values(), list(opening.values()), list(wrist_force_z_N.values())]
        if any(not np.isfinite(value).all() for value in measurements):
            raise ValueError('Invalid native supported-contact measurement')
        if step <= self._step:
            return
        if step != self._step + 1:
            self._recent.clear()
        self._step = step

        contact = self._contact_patch(points, tcp_m, opening, wrist_force_z_N,
                                      iron_position_m, iron_orientation_native)
        key, patch = contact if contact is not None else (None, None)
        if key != self._contact:
            self._recent.clear()
            self._contact = key
        if patch is None:
            return
        self._recent.append(patch)
        if len(self._recent) == CONTACT_CONTROLS:
            sustained = np.logical_and.reduce(self._recent)
            face = key[1]
            self._visited[face] |= sustained
            self._loaded_controls[face] += int(sustained.any())

    def _contact_patch(self, points, tcp_m, opening, wrist_force_z_N, position, quaternion):
        rotation = _rotation(quaternion)
        if rotation[2, 2] < np.cos(np.deg2rad(MAX_SOLE_TILT_DEG)):
            return None
        grasp = []
        for hand in ('left', 'right'):
            local = (np.asarray(tcp_m[hand]) - position) @ rotation - HANDLE_CENTER
            if GRASP_OPENING[0] <= opening[hand] <= GRASP_OPENING[1] and np.all(np.abs(local) <= HANDLE_TCP_TOLERANCE):
                grasp.append(hand)
        if not grasp:
            return None
        # Include the additional empty-wrist tare when both hands share the handle.
        # Downward pressure can make the net vertical wrench negative.
        load = sum(wrist_force_z_N[hand] for hand in grasp)
        if (not np.isfinite(load) or load > MAX_PRESS_WRIST_N + 6.8*(len(grasp)-1)
                or not np.isfinite(points).all()):
            return None

        areas, masks = measure_upward_patch(points, self._panels, position, quaternion)
        face = int(areas['back_patch_m2'] > areas['front_patch_m2'])
        if areas[f'{SIDES[face]}_patch_m2'] < MIN_PATCH_M2:
            return None
        for i, panel in enumerate(self._panels):
            a, b, c = points[panel, :2].transpose(1, 0, 2)
            u, v = b - a, c - a
            signed = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
            oriented = signed > 0 if i == face else signed < 0
            if self._weights[i][oriented].sum() < MIN_FACE_UP * self._area[i]:
                return None
        return (tuple(grasp), face), masks[face]

    def diagnostics(self):
        """Report actual covered rest area and accepted control counts for each face."""
        out = {}
        for i, side in enumerate(SIDES):
            weights, target, visited = self._weights[i], self._target[i], self._visited[i]
            covered = weights[visited].sum()
            crease_covered = weights[target & visited].sum()
            out[f'{side}_loaded_area_m2'] = float(covered)
            out[f'{side}_loaded_coverage'] = float(covered / self._area[i])
            out[f'{side}_loaded_crease_coverage'] = float(crease_covered / self._crease_area[i])
            out[f'{side}_loaded_controls'] = self._loaded_controls[i]
        return out
