# Slice and wrap a portion of cooked ham

This local submission draft contains the Isaac Lab scripted prototype. It depicts a deli worker's core sequence with a guarded gravity slicer: feed the product using the carriage and pusher, stop the slicing motion, collect the cut portion, place it in a food-grade paper envelope, and fold the envelope closed. OSHA's slicer guidance identifies the rotary blade, on/off control, thickness adjustment, product carriage and pusher as common components, and stresses using the provided pusher and guarding. The prototype keeps the robot on the carriage handle and away from the blade during the cut, then stops the slicer before packaging. [OSHA slicer safety sheet](https://www.osha.gov/sites/default/files/publications/OSHA3794.pdf), [slicer operator manual](https://www.williams-sonoma.com/netstorage/pdf/Chefs_Choice_Meat_Slicer_615_Manual.pdf).

## Scripted sequence

1. The right arm taps the slicer power control and grasps the outboard carriage knob.
2. It slides the whole piece along the face of the visibly rotating blade. The edge first meets the leading side, then sweeps the full cut face. Only then does the scripted cut event release a 3.4 mm slice into the catcher.
3. It returns the carriage and taps the stop control.
4. It pinches the slice at its exposed edge and carries it from the catcher to the paper envelope on the blade's exit side. A conservative guard volume is checked against the released slice's bounds at every control step.
5. The right arm parks and the left arm folds the flap over the slice, then parks.

## Fidelity boundary

The robot, carriage, ham, released portion, catcher, and envelope flap use native Isaac Sim/PhysX rigid bodies and contacts. The blade rotation is visual. The sweep is checked geometrically before scripted separation, with a release velocity to represent an outward deflector. The simulation does not model deformation, fracture, cutting force or interlocks. The ham is fixed to the carriage and the envelope base is fixed to the table. The envelope flap is rigid with a hinge and a raised tab. These are simulator fixtures. There is no validated physical replica or real-trial procedure.

The scene uses the calibrated YAM station tabletop directly. It does not add a second coplanar tabletop plate. The recorder produces separate `top-view.mp4` and `close-view.mp4` files at 1920 × 1080. The review copy places the synchronized top view on the left and the close view on the right.

## Native run

Use the installed Isaac Lab environment on verified `visnu`, with the existing accepted NVIDIA terms. Store output and caches on NVMe. From this task folder:

```bash
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME=/media/nvme/rbasti/isaaclab-arena-runtime/xdg-cache
export PYTHONPATH="$PWD"
ISAAC_PY=/media/nvme/rbasti/isaaclab-arena-runtime/source/.venv/bin/python
"$ISAAC_PY" run_scripted.py /media/nvme/rbasti/isaaclab-arena-runtime/outputs/prosciutto-slicer-new --preview --max-steps 1400 --fps 20 --renderer RaytracedLighting
```

The run uses 20 Hz control, 240 Hz CPU PhysX and GPU rendering. The two camera MP4s are written from the same native simulation run. Use `--physics-only` from the task folder for a diagnostic without camera recording. The demo uses seed 0 and takes 1,218 control steps (60.9 s of simulation). Rendering both views on an RTX 3090 took about six minutes. The last row of `trace.jsonl` records the score and measured endpoint.

```bash
tail -n 1 /path/to/run/trace.jsonl | python3 -c 'import json, sys; row = json.load(sys.stdin); print(row["step"], row["metrics"]["score"])'
```

The current prototype recorded 1.0 at step 1,218. The center of the one slice was inside the envelope, the flap was flat, and the slicer was stopped. In the recording, the slice lands in the catcher around 00:14. The scripted controller directly invokes start, stop and cut release. Before Dream-1 evaluation, route those events through policy-accessible controls and scene-state checks. Then validate the hazard and completion scorer for arbitrary actions, rerun the scripted demo, and record the Dream-1 trial and its measured score.
