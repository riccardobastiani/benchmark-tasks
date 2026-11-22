# Press a front crease strip on a T-shirt

Use the iron to press the right half of the front crease flat. Leave the shirt spread out and released, with the iron fully on its rest.

This submission scope is the localized **front strip** shown in the approved review video. It does not require turnover or pressing the back. The simulation represents cold mechanical flattening; it does not establish heat-set wrinkle removal or safe heated-iron behavior. Material parameters have not been calibrated on a physical garment.

## Run

Use Linux, an NVIDIA CUDA GPU, Python 3.12, Isaac Sim 6.1.0.0, Isaac Lab 17.0.2 and Arena 0.3.0. Install the pinned YAM and Inspect Robots dependencies in `pyproject.toml` into that existing Isaac environment. Isaac Sim requires its license acceptance. All task geometry and the initial cloth mesh are included; no task asset download is needed. Use the same physical GPU for CUDA and Vulkan. Replace the GPU index below if needed.

```bash
cd tasks/iron-standard-tshirt
export CUDA_VISIBLE_DEVICES=0
export ROBOCURVE_TSHIRT_RENDER_GPU=0
export ROBOCURVE_TSHIRT_CPU_NORMALS=1
export ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0
python run_trial.py /tmp/tshirt-demo --policy scripted
python run_trial.py /tmp/tshirt-hover --policy hover --width 640 --height 360
python run_trial.py /tmp/tshirt-idle --policy idle --width 640 --height 360
# DREAM_API_KEY must already be configured privately; do not put it in this folder.
python run_trial.py /tmp/tshirt-dream --policy dream
```

Each output path must be new and outside the source folder. The runner records only the native top view, defaults to 1920 x 1080 H.264/yuv420p at 20 fps, and gives Dream-1 calibrated top and wrist RGB plus joint state at 320 x 180. Model actions pass through the stock Inspect Robots guards; the scripted controller never supplies actions during Dream trials. The two policies use the same scene, instruction, reset and evaluator. The scripted trajectory takes 1,229 controls (61.45 simulated seconds); the model limit is 1,600 controls (80 seconds). Wall-clock runtime depends on GPU contention and hosted policy latency and is saved in each receipt.

## Score and evidence

`front_finish.py` implements the five straightedge profile checks and final released hold. `coverage_tracker.py` checks physically supported use of the iron on the target band. `task.json` states the complete numeric rubric. No phase label, prescribed arm, simultaneous bimanual contact or fixed stroke order is used by the evaluator.

The five stations are 30, 80, 130, 180 and 230 mm inboard from the right torso seam (material y = -220, -170, -120, -70, -20 mm). At each, a 40 mm bridge runs from material x = 267.5 to 307.5 mm, measured from the hem. Use a ruler/straightedge and 2 mm feeler gauge on a physical trainer; endpoints must remain 36–44 mm apart. The simulation measures the greatest perpendicular profile distance to the endpoint chord. This also detects sideways buckling. A station passes only throughout the final one-second released and quiet window. Credit requires supported iron use over at least half the target band; an untouched already-flat station can count only after this common condition is met.

The supported-contact proxy requires the handle within 60 x 6 x 6 mm of a grasping TCP and normalized jaw aperture 0.26–0.34 (28 mm handle), sole within 5 degrees of level, an exposed front patch of at least 40 cm² within -1.5 to +0.5 mm of the sole plane and within 25 degrees of its normal for three successive controls. The grasping wrist vertical load must be at most 12 N (18.8 N total if both hands hold the iron, including the extra wrist tare). These estimates distinguish supported pressing from carrying the 1 kg iron. Physical checking uses visible sustained sole/cloth contact and released final geometry; a calibrated wrist load cell is needed to reproduce the native force threshold exactly.

The run writes a source snapshot, source hashes, seed, runtime versions, timing, setup/final PNGs, `trace.jsonl` (every scored control), `cloth.npz` (sampled actual FEM nodes) and `receipt.json`. Dream runs also save a transcript. Read the final score without starting Isaac:

```bash
python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="completed" and r["source_unchanged"]; print(r["final_metrics"])' /tmp/tshirt-demo/receipt.json
python test_finish.py
```

A completed script with `validation_error` is not a full-score demonstration. Keep all old prototype recordings separate: they used an unconditional-zero evaluator and are not qualifying demo/model evidence for this revision. Fresh native results are in `validation_results.json`; the task metadata names the uploaded recordings.

## Implementation and limitations

`cloth.py` authors native surface FEM, sewn front/back panels and openings; `settled_buckle.npz` stores only the initial positions. The submission reset raises the inherited cache's below-mat neckline vertices above the support before physics begins. After reset, the ordinary robot, gravity, collision and FEM solver determine all cloth and iron motion. `interaction.py` makes three overlapping passes across the fold. `run_reset.py` and the board helpers retain the earlier research diagnostics, but `run_trial.py` with condition `recoverable` is the canonical scored entry point. `recoverable` now selects the approved broad-mat settled crease layout. Outputs must remain outside the package.

The rest pad is represented as a rigid support and cloth is elastic, so transient simulated flattening is not proof of permanent real cotton ironing. Damage, steam and thermal safety are outside the task claim. Attribution for the cloth construction is included in `LICENSE.cloth`; native YAM assets come from the pinned runtime dependency.

## Recorded results

Dream-1 run: 0.0; demo: 1.0. The matched hover control scored 0.0.

- Demo: 1229 controls; 25.0 minutes wall time on a shared RTX 3090.
- Hover: 1229 controls; 23.8 minutes wall time on a shared RTX 3090.
- Dream: 1600 controls; 31.7 minutes wall time on a shared RTX 3090.

All three recordings used identical frozen task sources. Code, geometry, materials, instruction and scoring remain unchanged in this package; result metadata and this README were completed afterward. The earlier prototype recordings remain historical evidence only.

The same saved reset was used for all runs. After native settling, 99% of nodes differed by at most 0.45 mm per axis from the demo, with isolated differences up to 3.54 mm. See the recorded measurements in `validation_results.json`.
