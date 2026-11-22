#!/usr/bin/env python3
"""Inspect saved garment geometry after the robot lets go."""

import argparse
import json
from pathlib import Path

import numpy as np

from soleplate import measure_soleplate, measure_upward_patch


PRESS_CONDITIONS = ('z_buckle', 'settled_buckle', 'wide_buckle', 'pressing_table')
TURNOVER_CONDITIONS = ('turnover', 'short_turnover', 'settled_short_turnover')


def panel_metrics(rest, points, faces, *, crease_span=None):
    """Measure native mesh geometry using fixed material coordinates."""
    nominal = rest[faces]
    observed = points[faces]
    flat_area = np.linalg.norm(np.cross(nominal[:, 1]-nominal[:, 0], nominal[:, 2]-nominal[:, 0]), axis=1).sum()
    cross = np.cross(observed[:, 1]-observed[:, 0], observed[:, 2]-observed[:, 0])
    projection = cross[:, 2]
    projected_area = np.abs(projection).sum()
    signed_area = abs(projection.sum())

    # 20 mm on each side gives a 40 mm material span. The 3D offset from
    # the chord detects a reversal even when the fold sits nearly level.
    panel = np.unique(faces)
    grid = np.rint(rest[panel, :2]*400).astype(int)
    node = dict(zip(map(tuple, grid), panel, strict=True))
    triplets = []
    for (x, y), middle in node.items():
        for dx, dy in ((8, 0), (0, 8)):
            before = node.get((x-dx, y-dy))
            after = node.get((x+dx, y+dy))
            if before is not None and after is not None:
                triplets.append((before, middle, after))
    before, middle, after = points[triplets].transpose(1, 0, 2)
    deflection_mm = np.linalg.norm(middle - (before+after)/2, axis=1)*1000
    result = dict(projected_area_ratio=float(signed_area/flat_area),
                  outward_up_fraction=float(projection[projection > 0].sum()/projected_area) if projected_area else 0.,
                  reversed_projected_fraction=float(.5*(1-signed_area/projected_area)) if projected_area else 1.,
                  local_deflection_p99_mm=float(np.percentile(deflection_mm, 99)),
                  local_deflection_max_mm=float(deflection_mm.max()))
    if crease_span is not None:
        a, m, b = rest[triplets].transpose(1, 0, 2)
        across_crease = (b[:, 0] > a[:, 0]) & (crease_span[0] <= m[:, 0]) & (m[:, 0] <= crease_span[1])
        across_crease &= np.abs(m[:, 1]) < .22
        result['crease_band_deflection_p95_mm'] = float(np.percentile(deflection_mm[across_crease], 95))
        swept_half = across_crease & (m[:, 1] <= .03)
        result['crease_swept_half_deflection_p95_mm'] = float(np.percentile(deflection_mm[swept_half], 95))
        result['crease_unswept_half_deflection_p95_mm'] = float(np.percentile(deflection_mm[across_crease & ~swept_half], 95))
    return result


def evaluate(path):
    receipt = json.loads((path/'receipt.json').read_text())
    if (receipt.get('status') != 'completed'
            or receipt.get('source_unchanged') is not True):
        raise ValueError('Expected a completed native diagnostic with unchanged source')
    with np.load(path/'cloth.npz') as trace:
        rest, points, faces = trace['rest'], trace['positions'], trace['faces']
    if points.shape != (len(receipt['observations']), *rest.shape) or not np.isfinite(points).all():
        raise ValueError('Invalid native trace or receipt length')
    half = len(rest)//2
    panels = (faces[(faces < half).all(axis=1)], faces[(faces >= half).all(axis=1)])
    out = {}
    released = receipt['observations'][-1]
    if 'front_loaded_area_m2' in released:
        out['loaded_coverage'] = {
            side: {metric: released[f'{side}_loaded_{metric}']
                   for metric in ('area_m2', 'coverage', 'crease_coverage', 'controls')}
            for side in ('front', 'back')
        }
    if 'iron_rest_xy_clearance_mm' in released:
        out['iron_rest'] = {key: released[f'iron_rest_{key}']
                            for key in ('xy_clearance_mm', 'lowest_gap_mm', 'tilt_deg')}
    condition = receipt.get('condition')
    crease_span = (.266, .3095) if condition in (*PRESS_CONDITIONS, *TURNOVER_CONDITIONS) else None
    frames = [('reset', points[0]), ('final', points[-1])]
    if crease_span is not None:
        settled = next((i for i, row in enumerate(receipt['observations']) if row['step'] == 100), None)
        if settled is not None:
            frames.insert(1, ('settled', points[settled]))
    for label, frame in frames:
        out[label] = dict(zip(('front', 'back'), (panel_metrics(rest, frame, f, crease_span=crease_span)
                                                for f in panels), strict=True))
    if condition in TURNOVER_CONDITIONS:
        x, y = rest[:, :2].T
        if condition == 'turnover':
            seams = {'left': (abs(x-.15) <= .011) & (-.231 <= y) & (y <= -.219),
                     'right': (abs(x-.49) <= .011) & (-.241 <= y) & (y <= -.229)}
        else:
            seams = {'left': (x <= .011) & (abs(y+.085) <= .012),
                     'right': (abs(x-.62) <= .011) & (abs(y+.01) <= .012)}
        baseline = points[settled] if settled is not None else points[0]
        reference = {side: baseline[mask].mean(axis=0) for side, mask in seams.items()}
        observations = receipt['observations']
        out['seam_track'] = []
        for index, (row, frame) in enumerate(zip(observations, points, strict=True)):
            if index+1 < len(observations) and observations[index+1]['phase'] == row['phase']:
                continue
            patches = {}
            for side, mask in seams.items():
                patch = frame[mask]
                delta = patch.mean(axis=0) - reference[side]
                nearest = np.linalg.norm(patch - row['tcp_m'][side], axis=1).min()
                patches[side] = dict(displacement_mm=(delta*1000).tolist(), nearest_tcp_mm=float(nearest*1000))
            out['seam_track'].append(dict(step=row['step'], phase=row['phase'], **patches))
    if 'iron_orientation_native' in receipt['observations'][0]:
        out['sweep_proximity'] = [dict(step=row['step'], phase=row['phase'], **measure_soleplate(frame,
                                  row['iron_position_m'], row['iron_orientation_native']))
                                  for row, frame in zip(receipt['observations'], points, strict=True)
                                  if row['phase'] in ('land', 'press_hold', 'sweep')]
    if 'wrist_force_world_N' in receipt['observations'][0] and condition in PRESS_CONDITIONS:
        force = out['sweep_wrist_z_N'] = []
        sole = out['upward_face_soleplate'] = []
        visited = np.zeros((2, len(panels[0])), dtype=bool)
        for row, frame in zip(receipt['observations'], points, strict=True):
            if row['phase'] not in ('press_hold', 'sweep'):
                continue
            force.append(dict(step=row['step'], phase=row['phase'], force_N=row['wrist_force_world_N'][2]))
            area, masks = measure_upward_patch(frame, panels, row['iron_position_m'], row['iron_orientation_native'])
            sole.append(dict(step=row['step'], phase=row['phase'], **area))
            visited |= masks
        coverage = {}
        for side, panel, touched in zip(('front', 'back'), panels, visited, strict=True):
            tri = rest[panel]
            x, y = tri.mean(axis=1)[:, :2].T
            target = (crease_span[0] <= x) & (x <= crease_span[1]) & (np.abs(y) < .22)
            weights = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
            coverage[side] = float(weights[target & touched].sum()/weights[target].sum())
        out['near_sole_crease_coverage'] = coverage
    if any(row.get('phase') == 'bare_hold' for row in receipt['observations']):
        phases = ('empty_hover', 'loaded_hover', 'cloth_hold', 'bare_hold')
        loads = {}
        for phase in phases:
            samples = [(row, frame) for row, frame in zip(receipt['observations'], points, strict=True)
                       if row['phase'] == phase]
            if len(samples) < 4:
                raise ValueError(f'Missing settled samples for the {phase} control')
            samples = samples[len(samples)//2:]
            forces = np.array([row['wrist_force_world_N'][2] for row, _ in samples])
            near = [measure_soleplate(frame, row['iron_position_m'], row['iron_orientation_native'])
                    ['nodes_near_soleplate'] for row, frame in samples]
            loads[phase] = dict(samples=len(samples), median_wrist_force_z_N=float(np.median(forces)),
                                force_span_N=float(np.ptp(forces)), nodes_near_soleplate=int(np.median(near)))
        empty, loaded, cloth, bare = (loads[phase]['median_wrist_force_z_N'] for phase in phases)
        out['load_diagnostic'] = dict(phases=loads, hover_added_weight_N=loaded-empty,
                                     cloth_support_estimate_N=loaded-cloth, bare_support_estimate_N=loaded-bare)
    return out


def compare_strokes(loaded, hover):
    """Report released deformation alongside the measured load and clearance controls."""
    a, b = (json.loads((p/'receipt.json').read_text()) for p in (loaded, hover))
    common = ('condition', 'task_fingerprint', 'runner_sha256', 'seed', 'versions', 'physics_dt_s',
              'control_dt_s', 'camera_resolution', 'use_fabric', 'render_kit_args')
    if (a.get('condition') not in ('z_buckle', 'settled_buckle')
            or not a['scope'].endswith('plate support') or not b['scope'].endswith('hover')
            or any(a[key] != b[key] for key in common)):
        raise ValueError('Compare one loaded and one hover run from the same frozen setup')
    for run in (a, b):
        if run['status'] != 'completed' or run['observations'][-1]['phase'] != 'released_hold':
            raise ValueError('Expected two completed release holds')
    if [(r['step'], r['phase']) for r in a['observations']] != [(r['step'], r['phase']) for r in b['observations']]:
        raise ValueError('The loaded and hover observations must use the same control schedule')

    summary = {}
    for label, path in (('loaded', loaded), ('hover', hover)):
        audit = evaluate(path)
        phases = ('press_hold', 'sweep')
        gaps = [sample['minimum_gap_m'] for sample in audit['sweep_proximity']
                if sample['phase'] in phases and sample['minimum_gap_m'] is not None]
        summary[label] = dict(
            settled_front_projected_ratio=audit['settled']['front']['projected_area_ratio'],
            minimum_gap_mm=min(gaps)*1000,
            median_wrist_z_N=float(np.median([sample['force_N'] for sample in audit['sweep_wrist_z_N']])),
            maximum_front_patch_cm2=max(sample['front_patch_m2'] for sample in audit['upward_face_soleplate'])*1e4,
            maximum_back_patch_cm2=max(sample['back_patch_m2'] for sample in audit['upward_face_soleplate'])*1e4,
            crease_coverage=audit['near_sole_crease_coverage'],
        )
        for region in ('swept', 'unswept'):
            key = f'crease_{region}_half_deflection_p95_mm'
            initial = audit['settled']['front'][key]
            released = audit['final']['front'][key]
            summary[label][region] = dict(initial_mm=initial, released_mm=released, decrease_mm=initial-released)
    summary['additional_swept_decrease_mm'] = summary['loaded']['swept']['decrease_mm'] - summary['hover']['swept']['decrease_mm']
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--hover', type=Path, help='Compare a loaded run with a matched hover control')
    args = parser.parse_args()
    print(json.dumps(compare_strokes(args.run, args.hover) if args.hover else evaluate(args.run), indent=2))
