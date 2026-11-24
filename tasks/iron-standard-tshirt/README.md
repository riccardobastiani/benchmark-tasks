# Iron the front strip of a T-shirt

Iron the marked right half of the front of the T-shirt. Leave the shirt spread out and released, and return the iron to its rest.

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

Each output path must be new and outside the source folder. The runner records only the native top view, defaults to 1920 x 1080 H.264/yuv420p at 20 fps, and gives Dream-1 calibrated top and wrist RGB plus joint state at 320 x 180. Model actions pass through the stock Inspect Robots guards; the scripted controller never supplies actions during Dream trials. The two policies use the same scene, instruction, reset and evaluator. The scripted trajectory takes 1,637 controls (81.85 simulated seconds); the model limit is 1,800 controls (90 seconds). Wall-clock runtime depends on GPU contention and hosted policy latency and is saved in each receipt.

## Physical reference and proposed referee procedure

The reference garment is a [UNIQLO SUPIMA Cotton Crew Neck T-shirt, E455365, size M](https://www.uniqlo.com/sg/en/products/E455365-000/00?colorCode=COL12&sizeCode=SMA003). The manufacturer's garment chart lists 100% cotton, 680 mm back length, 515 mm body width laid flat, 430 mm shoulder width, and 420 mm sleeve length from the center back. The simulated garment has a 500 mm torso width and an estimated mass of 133 g. [Laurastar's T-shirt demonstration](https://www.youtube.com/watch?v=lQHrj4i4M3A) shows garment handling and iron motion with an ironing system. [Philips' cotton ironing demonstration](https://www.youtube.com/watch?v=vjQaX3PnhP0) shows heated ironing. The room-temperature reset and finish measurements below are proposed and await physical validation.

Mark five station centers 287.5 mm from the hem and 30, 80, 130, 180 and 230 mm inboard of the robot-right torso edge. In a referee video, identify a pass from overhead sole coverage and low oblique footage of the sole sitting flush on the cloth. At 20 Hz the simulator credits each exposed front station after three consecutive samples under the level sole in its contact tolerance. Each station earns 0.2. At the end, verify the front is up, the shirt stays spread on the mat, the hands are clear and the iron is flat and fully on its rest for one second. Track the hem corners, sleeve tips and neckline center for stillness. Each may move at most 3 mm. The five crease profiles remain available for diagnostics.

The powered-iron representation uses a visual cord and outlet. The native cloth and sole contact determine the cloth movement. The model omits thermal effects and cord forces. The physical garment, reset and finish remain unverified.

The earlier RoboCurve recordings use the previous mechanical finish rubric. The revised recordings and score are listed below.

## Revised results

Scripted demo 1.0 at 1637 controls. Dream-1 0.0 after 1800 controls. Both videos use native 1080p at 20 fps. RoboCurve media ID M-ADCDTA.

The mean crest height across six diagnostic profiles fell from 7.03 mm to 4.91 mm after release. The treated strip retained some visible creasing. See `validation_results.json` for the run records.

For a proposed physical reset, locate the crease line using the hem and torso edge. Form a shallow raised ridge with a removable smooth 4 mm rod and feed a small amount of fabric slack around it. Pull the rod out before the trial and photograph the settled ridge with a ruler in view. The resulting setup and its response to pressing require physical validation.
