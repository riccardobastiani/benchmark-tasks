"""Render and check the native starting layout; this is not a scripted task demo."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent.resolve()
    receipt = dict(scope='Native stationary scene-design preview; no benchmark score',
        hostname=socket.gethostname(), status='starting',
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version',
                                     '--format=csv,noheader'], text=True),
        versions={name:importlib.metadata.version(name) for name in
                  ('isaacsim','isaaclab','isaaclab-arena','yam-arena')},
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=source, text=True).strip())
    start, emb = time.monotonic(), None
    try:
        import cv2
        import numpy as np
        from inspect_robots import Scene
        from inspect_robots.types import Action
        from resist_isaac.runtime import ResistanceEmbodiment
        from resist_isaac.state import pose_state, tensor
        emb = ResistanceEmbodiment(task=str(source), condition='design_preview', device='cpu',
            render_width=1920, render_height=1080, cam_width=320, cam_height=180)
        emb.reset(Scene(id='suction-liner-design', instruction='Scene design preview.'), seed=0)
        mechanism = emb._mechanism
        action = Action(data=mechanism.initial_action())
        samples = []
        for tick in range(60):
            emb.step(action)
            if tick >= 40:
                samples.append(mechanism.evaluate(emb._base))
        receipt['metrics'] = mechanism.evaluate(emb._base)
        receipt['native_render_settings'] = mechanism.render_settings
        receipt['settled_metrics'] = samples
        if any(m['canister_drift_m'] > .01 or m['liner_drift_m'] > .01
               or m['connector_drift_m'] > .008 or m['max_linear_speed_m_s'] > .02
               or m['max_angular_speed_rad_s'] > .25
               or m['bin_lid_angle_error_rad'] > .02 for m in samples):
            raise RuntimeError('Starting assembly failed native stability check')
        for name in ('top_cam', 'close_cam', 'overview_cam'):
            frame = tensor(emb._base.scene[name].data.output['rgb'])[0,:,:,:3].cpu().numpy()
            if frame.shape != (1080,1920,3):
                raise RuntimeError('Unexpected native camera dimensions')
            if not cv2.imwrite(str(output/(name+'.png')), cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)):
                raise RuntimeError('Could not save native image')
        states = {}
        for name, data in mechanism.descriptions.items():
            for part in data['bodies']:
                if part.get('static',False):
                    continue
                values = [v.cpu().tolist() for v in pose_state(emb._base.scene[name+'_'+part['name']])]
                if not all(np.isfinite(value).all() for value in values):
                    raise RuntimeError('Non-finite native body state')
                states[name+'/'+part['name']] = values
        receipt.update(status='completed', native_device=str(emb._base.device),
                       physics_dt_s=emb._base.physics_dt, control_dt_s=emb._base.step_dt,
                       stationary_control_steps=60, object_states=states)
        import omni.usd
        omni.usd.get_context().get_stage().GetRootLayer().Export(str(output/'scene.usda'))
        # Export retained apparatus separately: original runtime temporary assets expire on exit.
        from resist_isaac.usd_assets import create_asset
        from resist_isaac.runtime import _task_module
        task_module,_ = _task_module(source)
        for name, data in mechanism.descriptions.items():
            path = create_asset(data, output/'assets'/(name+'.usda'), source)
            if name in ('canister','liner'):
                task_module.clear_plastic(path,data)
        # Keep the exported scene's apparatus references valid after temporary files expire.
        scene_path = output/'scene.usda'
        scene_path.write_text(scene_path.read_text().replace(mechanism._temporary.name,
                                                            str(output/'assets')))
        receipt['source_sha256'] = {str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
        receipt['runtime_sha256'] = {str(path.name):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (source/'resist_isaac').glob('*.py')}
    except Exception as error:
        receipt.update(status='error', error=repr(error))
        raise
    finally:
        receipt['wall_seconds'] = time.monotonic()-start
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if emb is not None:
            emb.close()


if __name__ == '__main__':
    main()
