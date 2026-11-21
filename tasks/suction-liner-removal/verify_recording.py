"""Audit revised native closure evidence; never award a score to old media."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def audit(directory, package, *, full_demo=False):
    receipt = json.loads((directory / 'receipt.json').read_text())
    assert receipt['status'] in ('completed', 'completed_checkpoint'), receipt.get('error')
    assert receipt['hostname'] == 'visnu'
    assert receipt['physics_dt_s'] == 1 / 240 and receipt['control_dt_s'] == 1 / 20
    assert not any(k in receipt for k in ('error', 'video_error', 'final_frame_error'))
    for path in package.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = str(path.relative_to(package))
        if path.suffix == '.py' or path.name in ('pilot.json', 'pyproject.toml'):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['source_sha256'][relative], relative
    assert receipt.get('source_unchanged', True), receipt.get('source_changed_files')
    rows = [json.loads(s) for s in (directory / 'trace.jsonl').read_text().splitlines()]
    if receipt['policy'] == 'dream':
        transcript = json.loads((directory / 'dream-transcript.json').read_text())
        assert not transcript.get('dry_run', False) and transcript.get('requests')
        assert all(row['phase'] == 'dream' and len(row['diagnostics']['raw_action']) == 14 for row in rows)
    events = [json.loads(s) for s in (directory / 'native-events.jsonl').read_text().splitlines()]
    assert len(rows) == receipt['steps'] and len(events) == receipt['steps'] * 12
    isolated = extracted = False
    retained = []
    violations = 0
    for event in events:
        isolated |= event['switch_angle_rad'] > .90
        extracted |= event['liner_lift_m'] > .012
        if receipt['policy'] == 'scripted':
            assert not event['unsafe_reason'], event['unsafe_reason']
        if isolated or extracted:
            engaged = (event['cap_seated'] and -.0008 < event['cap_seat_gap_m'] < .0010
                       and event['cap_radial_error_m'] < .0012 and event['cap_up'] > .99)
            violations += not engaged
            if receipt['policy'] == 'scripted':
                assert engaged, event
            retained.append(event)
    end = receipt['final_metrics']
    if full_demo:
        assert receipt['status'] == 'completed' and receipt['policy'] == 'scripted'
        assert end['score'] == 1.0 and end['success']
        assert isolated and extracted and retained
        assert end['cap_seated'] and end['liner_inside'] and end['hands_clear']
        assert end['canister_retained'] and end['connector_parked']
        assert all(not event['arm_contact'] for event in events)
    elif receipt['policy'] == 'scripted':
        assert end['cap_seated'], end
    videos = {}
    for name in ('top_cam', 'close_cam', 'overview_cam'):
        path = directory / (name + '.mp4')
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                          '-show_format', '-of', 'json', str(path)]))
        stream = info['streams'][0]
        assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
        assert int(stream['nb_frames']) == receipt['steps'] + 1
        assert [stream['width'], stream['height']] == receipt['resolution']
        assert path.stat().st_size < 50_000_000
        result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                                capture_output=True)
        assert result.returncode == 0 and not result.stderr, result.stderr[:200]
        videos[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            frames=int(stream['nb_frames']), bytes=path.stat().st_size)
    return dict(run=str(directory), policy=receipt['policy'], status=receipt['status'],
                score=end['score'], steps=receipt['steps'], resolution=receipt['resolution'],
                retained_physics_samples=len(retained),
                retention_violations=violations,
                retained_gap_range_mm=([min(e['cap_seat_gap_m'] for e in retained)*1000,
                                        max(e['cap_seat_gap_m'] for e in retained)*1000]
                                       if retained else None),
                released_final_gap_mm=end['cap_seat_gap_m']*1000, videos=videos,
                scope='Native revised trainer evidence; not physical bench validation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--full-demo', action='store_true')
    args = parser.parse_args()
    result = audit(args.directory, args.package, full_demo=args.full_demo)
    (args.directory / 'revision-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
