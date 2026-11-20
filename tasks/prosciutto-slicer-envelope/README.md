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

Activate an Isaac Lab-Arena Python environment with the YAM station and Inspect Robots installed. The local validation used Arena `c7edf3a9` and Isaac Lab `91a0906f9`. Set `OMNI_KIT_ACCEPT_EULA=YES` in the environment after accepting the NVIDIA terms. From this task folder:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
python run_scripted.py "$PWD/../prosciutto-slicer-run" --preview --max-steps 1400 --fps 20 --renderer RaytracedLighting
```

The run uses 20 Hz control, 240 Hz CPU PhysX and GPU rendering. The two camera MP4s are written from the same native simulation run. Use `--physics-only` from the task folder for a diagnostic without camera recording. The demo uses fixed seed 0 at 20 Hz. The evaluation limit is 1,400 control steps (70 s). The recorded native run uses this fixed layout and seed. The score appears in the final row of the trace and in `receipt.json`. The last row of `trace.jsonl` records the score and measured endpoint.

```bash
tail -n 1 /path/to/run/trace.jsonl | python3 -c 'import json, sys; row = json.load(sys.stdin); print(row["step"], row["metrics"]["score"])'
```

## Run Dream-1

With the same environment and configured Dream Labs credentials, install `inspect-robots-dream` and `dreamlabs-client`.

```bash
python run_scripted.py "$PWD/../prosciutto-dream-run" --policy dream --max-steps 1400 --fps 20 --renderer RaytracedLighting
```

This uses the fixed seed-0 reset, the instruction in `task.json` and the standard Inspect Robots action guardrails. The run writes one top-view MP4, the policy transcript, and the final native score. A trial ends at success or at the 1,400-step limit. The recorded Dream-1 trial reached the limit in about 11 minutes of wall-clock time on an RTX 3090 and scored 0.0. The slicer remained off and no slice was released. The native scripted demo scored 1.0.

## Score a trial

Both robot entry points use the same task scorer. A blade incursion latches zero for the trial. At every 240 Hz physics substep, use the world-aligned bounds of each YAM URDF collision hull, including the fingertips. The check conservatively treats the whole blade disk as off limits, with a 2 mm margin. It stays active with the motor off. At 20 Hz, also check that the remaining ham stays behind the feed face and the separated slice stays out of the guard zone.

Full credit requires all completion conditions to hold for one second. The motor must be stopped after actual release of a slice by a powered carriage pass. The entire slice collision box must fit within the envelope interior, with 0.5 mm tolerance. The slice must be separated by at least 5 mm from the arm hull bounds and below 10 mm/s linear speed and 0.20 rad/s angular speed. The flap must be within 10 degrees of flat and turn no faster than 0.20 rad/s. A correct endpoint can be inspected using the motor state, the released portion and the flap position. The trace contains the native measurements.

## Equipment reference

Use the Berkel Home Line Plus 200 as a reference for the mechanism. Its data sheet gives a 195 mm blade and a 0–18 mm thickness range. The manual shows the product holder, guard, gauge plate and separate start and stop controls. The simulated 200 mm blade, 73.5 mm rail travel, 8 mm button travel and catcher dimensions are authored replica dimensions. The model has no functioning gauge plate. A guarded inert physical trainer preserving the carriage, pusher, gauge and safe slice exit has yet to be built and validated. [Berkel data sheet](https://s3.eu-west-1.amazonaws.com/berkel-s3/Manuali/HL-PLUS-200-250-data-sheet.pdf), [Berkel manual](https://s3.eu-west-1.amazonaws.com/berkel-s3/Manuali/manuale-home-line-plus-2022.pdf#page=14).
