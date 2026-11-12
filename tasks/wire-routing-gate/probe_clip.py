"""Native mechanical bench test: guided rigid wire section, not a robot demo."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
source = Path(__file__).parent
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [source/'geometry.py', source/'probe_clip.py']}
receipt = dict(scope='Actual native PhysX clip bench test; guided rigid wire section; no robot task score',
               hostname=socket.gethostname(), status='starting', source_sha256=hashes)
(output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, device='cpu', multi_gpu=False).app

try:
    import importlib.util
    import sys
    import numpy as np
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from pxr import UsdPhysics
    from resist_isaac.recovery import body, description
    from resist_isaac.state import tensor
    from resist_isaac.usd_assets import create_asset
    spec = importlib.util.spec_from_file_location('_clip_bench_geometry', source/'geometry.py')
    g = importlib.util.module_from_spec(spec); sys.modules[spec.name] = g; spec.loader.exec_module(g)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device='cpu', dt=1/240))
    stage = omni.usd.get_context().get_stage()
    clip = create_asset(g.clip_geometry('BenchClip', g.TARGET, (.08, .38, .70, 1)),
                        output/'clip.usda', source)
    stage.DefinePrim('/World/Clip', 'Xform').GetReferences().AddReference(str(clip))
    x, y, z = map(float, g.TARGET)
    start = [x, y, z+.035]
    section = body('section', [dict(type='capsule', size=[g.WIRE_RADIUS],
        fromto=[0., -.025, 0., 0., .025, 0.], color=[1., .36, .035, 1.],
        friction=[.6, .45], contact_offset=.0001)], .01, start)
    probe = create_asset(description('Probe', [section], joints=[dict(name='test_press',
        type='prismatic', child='section', axis='Z', position_parent=start,
        lower=-.05, upper=.05, stiffness=1500., damping=4.)],
        solver_position_iterations=64, solver_velocity_iterations=16), output/'probe.usda', source)
    stage.DefinePrim('/World/Probe', 'Xform').GetReferences().AddReference(str(probe))
    objects = {name: RigidObject(RigidObjectCfg(prim_path=path, spawn=None))
        for name, path in [('wire', '/World/Probe/section'), ('left', '/World/Clip/left'),
                           ('right', '/World/Clip/right')]}
    drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath('/World/Probe/test_press'), 'linear')
    drive.CreateMaxForceAttr(3.)
    sim.reset()
    dt = sim.get_physics_dt()
    phases = [('approach', .5), ('press', 1.2), ('release', .7), ('light_pull', 1.), ('pull_out', 1.)]
    records = []
    started = time.monotonic()
    with (output/'trace.jsonl').open('w') as trace:
        for phase, duration in phases:
            drive.GetStiffnessAttr().Set(0. if phase=='release' else 10. if phase=='light_pull' else 1500.)
            drive.GetDampingAttr().Set(0. if phase=='release' else 4.)
            for tick in range(round(duration/dt)):
                target = -.027*(tick+1)/round(duration/dt) if phase=='press' else 0.
                drive.GetTargetPositionAttr().Set(target)
                sim.step(render=False)
                for obj in objects.values(): obj.update(dt)
                poses = {name: tensor(obj.data.root_link_pose_w)[0, :3].cpu().numpy()
                         for name, obj in objects.items()}
                velocity = tensor(objects['wire'].data.root_com_vel_w)[0].cpu().numpy()
                if not all(np.isfinite(p).all() for p in [*poses.values(), velocity]):
                    raise RuntimeError('Invalid native bench measurement')
                row = dict(phase=phase, time_s=(len(records)+1)*dt,
                    wire_z_m=float(poses['wire'][2]), wire_speed_m_s=float(np.linalg.norm(velocity[:3])),
                    jaw_opening_m=float(max(abs(poses['left'][0]-(x-g.JAW_SPAN)),
                                              abs(poses['right'][0]-(x+g.JAW_SPAN)))),
                    commanded_guide_target_m=target)
                trace.write(json.dumps(row)+'\n'); records.append(row)
            print('Native phase endpoint:', json.dumps(records[-1]), flush=True)
    end = {phase: [r for r in records if r['phase']==phase][-1] for phase, _ in phases}
    peak = {phase: max(r['jaw_opening_m'] for r in records if r['phase']==phase) for phase, _ in phases}
    checks = dict(snap_in=end['press']['wire_z_m']<z+.012 and peak['press']>.001,
                  retained_after_release=z+.007<end['release']['wire_z_m']<z+.012,
                  retained_under_light_pull=end['light_pull']['wire_z_m']<z+.016,
                  released_under_strong_pull=end['pull_out']['wire_z_m']>z+.027 and peak['pull_out']>.001)
    receipt.update(status='completed', checks=checks, endpoints=end, peak_jaw_opening_m=peak,
                   dt_s=dt, steps=len(records), elapsed_wall_s=time.monotonic()-started,
                   force_scope='Guided prismatic test press with native spring drive, maximum effort 3 N; no measured force claim',
                   source_unchanged=all(hashlib.sha256((source/name).read_bytes()).hexdigest()==h
                                        for name, h in hashes.items()))
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if not all(checks.values()): raise RuntimeError('Native clip bench checks failed: '+str(checks))
    print('Native clip bench checks passed:', checks, flush=True)
except Exception as error:
    receipt.update(status='error', error=str(error))
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    raise
finally:
    app.close()
