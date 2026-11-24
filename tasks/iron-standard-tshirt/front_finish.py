"""Released front-strip finish, measured with five 40 mm straightedge bridges.

This measures cold mechanical flattening, not heat-set ironing quality.
"""

from collections import deque

import numpy as np

from .soleplate import _rotation


STATIONS_Y = (-.22, -.17, -.12, -.07, -.02)
BRIDGE_X = (.2675, .3075)
GAP_MM = 2.
HOLD_CONTROLS = 20


class FrontFinish:
    def __init__(self, rest, front_faces, *, pass_based=False):
        self.pass_based = pass_based
        self.rest = rest
        self.faces = front_faces
        front = np.unique(front_faces)
        grid = dict(zip(map(tuple, np.rint(rest[front, :2] * 400).astype(int)), front, strict=True))
        self.rows = []
        for y in STATIONS_Y:
            row = [grid[x, round(y*400)] for x in range(107, 124)
                   if (x, round(y*400)) in grid]
            if len(row) != 17:
                raise ValueError('Finish bridge does not match the refined front mesh')
            self.rows.append(row)
        landmarks = ((0., -.25), (0., .25), (.57, -.37), (.57, .37), (.62, 0.))
        self.landmarks = [front[np.linalg.norm(rest[front, :2]-xy, axis=1).argmin()]
                          for xy in landmarks]
        self.history = deque(maxlen=HOLD_CONTROLS)
        self.step = None
        self.last = None

    def gaps(self, points):
        profiles = points[self.rows]
        a, b = profiles[:, 0], profiles[:, -1]
        chord = b-a
        lengths = np.linalg.norm(chord, axis=1)
        gap = np.linalg.norm(np.cross(profiles-a[:, None], chord[:, None]), axis=2).max(axis=1)
        return gap / np.maximum(lengths, 1e-12)*1000., lengths*1000.

    def observe(self, step, points, *, geometry, coverage, rest_metrics, tcp_m,
                iron_position, iron_orientation):
        if step == self.step:
            return self.last
        if self.step is not None and step != self.step+1:
            self.history.clear()
        self.step = step
        if points.shape != self.rest.shape or not np.isfinite(points).all():
            raise ValueError('Invalid native garment measurement')
        if any(not np.isfinite(value).all() for value in [iron_position, *tcp_m.values()]):
            raise ValueError('Invalid native endpoint measurement')
        gaps, chords = self.gaps(points)
        passed = (gaps <= GAP_MM) & (chords >= 36.) & (chords <= 44.)
        if self.pass_based:
            passed = coverage["pressed_stations"].copy()
        rotation = _rotation(iron_orientation)
        hand_clearance = []
        for tcp in tcp_m.values():
            # An 80 mm TCP exclusion radius conservatively encloses the open jaws.
            cloth_gap = np.linalg.norm(points-tcp, axis=1).min()
            local = (np.asarray(tcp)-iron_position) @ rotation
            body_gap = np.linalg.norm(np.maximum(np.maximum(
                np.array([-.125, -.062, 0.])-local, local-np.array([.132, .062, .114])), 0.))
            hand_clearance.append(min(cloth_gap, body_gap))
        spread = (geometry['front_signed_projected_area_ratio'] >= .96
                  and geometry['front_outward_up_fraction'] >= .99
                  and geometry['front_reversed_fraction'] <= .01
                  and np.all((.02-.003 <= points[:, 0]) & (points[:, 0] <= .82+.003))
                  and np.all(np.abs(points[:, 1]) <= .503)
                  and np.all((.7525 <= points[:, 2]) & (points[:, 2] <= .7805)))
        parked = (rest_metrics['iron_rest_xy_clearance_mm'] >= 0.
                  and -1. <= rest_metrics['iron_rest_lowest_gap_mm'] <= 2.
                  and rest_metrics['iron_rest_tilt_deg'] <= 5.)
        used = bool(passed.any()) if self.pass_based else coverage.get('front_loaded_crease_coverage', 0.) >= .5
        eligible = spread and parked and min(hand_clearance) >= .08 and used
        self.history.append(((points[self.landmarks] if self.pass_based else points).copy(), np.asarray(iron_position).copy(), rotation, passed, eligible))
        motion_mm = 0.
        iron_motion_mm = 0.
        iron_motion_deg = 0.
        score = 0.
        if len(self.history) == HOLD_CONTROLS:
            first = self.history[0]
            motion = np.max if self.pass_based else lambda values: np.percentile(values, 95)
            motion_mm = float(max(motion(np.linalg.norm(row[0]-first[0], axis=1))
                                  for row in self.history)*1000.)
            iron_motion_mm = float(max(np.linalg.norm(row[1]-first[1]) for row in self.history)*1000.)
            iron_motion_deg = float(max(np.degrees(np.arccos(np.clip(
                (np.trace(row[2] @ first[2].T)-1.)/2., -1., 1.))) for row in self.history))
            if (all(row[4] for row in self.history) and motion_mm <= 3.
                    and iron_motion_mm <= 1. and iron_motion_deg <= 2.):
                score = float(np.logical_and.reduce([row[3] for row in self.history]).sum()/5.)
        out = dict(score=score, success=score == 1., finished_stations=int(passed.sum()),
                   finish_spread=bool(spread), finish_parked=bool(parked), finish_iron_used=bool(used),
                   finish_hands_clear=bool(min(hand_clearance) >= .08),
                   finish_cloth_motion_mm=motion_mm, finish_iron_motion_mm=iron_motion_mm,
                   finish_iron_motion_deg=iron_motion_deg)
        out.update({f'bridge_{i+1}_gap_mm': float(gap) for i, gap in enumerate(gaps)})
        out.update({f'bridge_{i+1}_span_mm': float(span) for i, span in enumerate(chords)})
        self.last = out
        return out
