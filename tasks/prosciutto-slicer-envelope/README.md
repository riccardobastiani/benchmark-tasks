# Slice and wrap a portion of cooked ham

This local submission draft contains the Isaac Lab scripted prototype. It depicts a deli worker's core sequence with a guarded gravity slicer: feed the product using the carriage and pusher, stop the slicing motion, collect the cut portion, place it in a food-grade paper envelope, and fold the envelope closed. OSHA's slicer guidance identifies the rotary blade, on/off control, thickness adjustment, product carriage and pusher as common components, and stresses using the provided pusher and guarding. The prototype keeps the robot on the carriage handle and away from the blade during the cut, then stops the slicer before packaging. [OSHA slicer safety sheet](https://www.osha.gov/sites/default/files/publications/OSHA3794.pdf), [slicer operator manual](https://www.williams-sonoma.com/netstorage/pdf/Chefs_Choice_Meat_Slicer_615_Manual.pdf).

## Scripted sequence

1. The right arm presses the spring-return start button and grasps the outboard carriage knob.
2. It slides the whole piece along the face of the visibly rotating blade. The edge first meets the leading side, then sweeps the full cut face. The task releases the 3.4 mm slice near the carriage travel stop after a powered pass across the full cut face.
3. It returns the carriage and presses the stop button.
4. It pinches the slice at its exposed edge and carries it from the catcher to the paper envelope on the blade's exit side. A conservative guard volume is checked against the released slice's bounds at every control step.
5. The right arm parks and the left arm folds the flap over the slice, then parks.

## Fidelity boundary

The robot, carriage, ham, released portion, catcher, and envelope flap use native Isaac Sim/PhysX rigid bodies and contacts. The blade rotation is visual. Button travel of 4 mm latches the native task power state. A geometric blade sweep releases a pre-modeled slice, with a release velocity representing the deflector. The simulation does not model deformation, fracture, cutting force or interlocks. The ham is fixed to the carriage and the envelope base is fixed to the table. The envelope flap is rigid with a hinge and a raised tab. These are simulator fixtures. There is no validated physical replica or real-trial procedure.

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

The run uses 20 Hz control, 240 Hz CPU PhysX and GPU rendering. The two camera MP4s are written from the same native simulation run. Use `--physics-only` from the task folder for a diagnostic without camera recording. The demo uses fixed seed 0 at 20 Hz. The proposed evaluation limit is 1,400 control steps (70 s). The native controls demo reached 1.0 at step 1,360 (68 s). The last row of `trace.jsonl` records the score and measured endpoint.

```bash
tail -n 1 /path/to/run/trace.jsonl | python3 -c 'import json, sys; row = json.load(sys.stdin); print(row["step"], row["metrics"]["score"])'
```

Scripted and policy-driven actions now use the same power and slice-release logic. The scorer checks a stopped blade, the slice center in the envelope and the flap angle. Full arm and fingertip clearance checks, whole-slice containment and release checks still need validation before submission. The existing side-by-side video documents the earlier prototype and must be refreshed for a submission.
