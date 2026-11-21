# Place six EEG cup electrodes on a mannequin scalp

A bimanual YAM robot places six labeled cup-style sensors on a bare training mannequin scalp in the order shown on a work card, then withdraws. The scene uses native Isaac Lab/Isaac Sim/PhysX. The task measures a fixed layout of labeled surface placements and release; the six positions do not constitute a clinical EEG examination.

Exact model instruction:

> Place the six EEG electrodes on the mannequin in the order shown on the card.

The card order is Fp1, Fp2, C3, C4, O1, O2. Colors are task-specific. Either arm can earn credit for any electrode; the scripted demo alternates left and right for reach. See [SETUP.md](SETUP.md) for every object, target, pose and material approximation, and [SCORING.md](SCORING.md) for the implemented numeric scorer.

## Recorded evidence

The fixed seed-0 scripted run reached score **1.0** at step **2,671**, **133.55 s** of simulated control, with all six paste bonds, all six targets, the full ordered prefix and both hands clear. Maximum final cup-center error: **5.624 mm**. All six cup centers stayed unchanged at recorded precision after the first release/settling second. The native capture took **1,699.13 s (28.32 minutes)** of wall time on an RTX 3090 while recording 1920 x 1080 at 20 fps. Timing is one observed run, not a performance guarantee.

The top-view demo is H.264/yuv420p MP4, 133.6 s, under 50 MB. The setup screenshot shows the settled initial layout, before robot interaction, and is under 10 MB. The final image is only a review aid; it is not the setup screenshot.

**Dream-1 completed 3,000 actions and scored 0.0; the scripted demo scored 1.0.** Both videos and the initial setup screenshot are uploaded as media **M-HY4TM8**, recorded in `task.json`.

## Software and run command

Use Python 3.12 with an operational Isaac Lab-Arena installation, YAM Arena, Inspect Robots, NumPy, OpenCV, USD/PhysX and `ffmpeg`/`ffprobe`. Task-owned assets are generated procedurally; no additional EEG asset download is required. The task folder includes the used RESIST helper modules and their MIT license. The YAM model, station textures and framework dependencies still come from the installed environment.

The validated development checkout was RESIST `f001a38fdf2c62f0d264544e403b4572f7d8cf7c` plus local task/runtime changes. The archived original run fingerprint was `849644f2223afba6901e97a12318d959cb9c34694603094cf8f88351e59ba12c`. The copied geometry and scorer match both demonstrated trials; the updated recorder was exercised by the Dream trial; adding documentation and vendoring helpers changes the package content fingerprint. The prepared package has not itself been rerun. Do not represent the commit alone as containing all executed local changes.

From this task folder inside a Git checkout, using the installed Isaac Lab interpreter:

```bash
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES
python run_video.py /absolute/path/to/new-eeg-run --fps 20 --max-steps 3000 --gpu cuda:0
```

Accept the NVIDIA terms before setting the EULA variable. GPU 1 is mapped to `cuda:0` by the example. The output folder must not already exist. The recorder records the actual hostname, GPU inventory and Git revision. The development-host restriction was removed from the submitted recorder so another configured GPU computer can run it; no additional native run on another host is claimed. The geometry and scorer remain byte-identical to both recorded trials. Use the installed Isaac Lab interpreter and execute from a Git checkout. The command describes the tested development workflow; a fresh Brev installation has not been verified.

The recorder writes `top-view.mp4`, `setup-top.png`, `final-top.png`, `trace.jsonl`, copied sources and `receipt.json`. It starts after 0.5 s settling and camera warmup. Seed is 0; jitter is disabled. Physics runs at 240 Hz; control at 20 Hz. Maximum trial length is **3,000 actions / 150 simulated seconds**, with early success after the one-second endpoint hold. The recorder selects the scripted policy by default; use `--policy dream` for hosted Dream-1.

## Read a trial score

`EEGPlacementTask.evaluate()` in `task.py` reads native PhysX state for both scripted and hosted-policy execution. The runner's receipt records the final score; every trace row records current metrics.

```bash
python - /absolute/path/to/run/receipt.json <<'PYCODE'
import json, sys
r = json.load(open(sys.argv[1]))
print(r['status'], r['steps'], r['final_metrics'])
PYCODE
```

Keep error and interrupted receipts distinct from completed trials. A native measurement failure must remain an errored/invalid run, not be relabeled as a valid score. Historical diagnostic probes are not full-score demonstrations of the six-cup task.

## Dream-1 run

The recorder supports hosted Dream-1 through `inspect-robots-dream` and `dreamlabs-client`, using the exact instruction above, the same seed, the same 410 mm arm spacing, the same native scorer and a 3,000-action limit. The native hosted trial completed all 3,000 actions (150 simulated seconds) with score **0.0** and zero electrodes seated or bonded. It made no target placement. The recording lasts 150.05 s including its initial frame. Observed wall time was **4,478.72 s (74.65 minutes)** on the shared RTX 3090 host, including rendering and hosted observation/inference handling. This is a valid completed failed trial, not an infrastructure error or a dry run. The transcript records 62 successful action chunks over 123 attempts, including client-managed expired-session retries. Installed versions were inspect-robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.1.3. Active action guardrails were clamp and delta-limit. The model used its standard 32-control preroll.

```bash
python run_video.py /absolute/path/to/new-eeg-dream-run --policy dream --fps 20 --max-steps 3000 --gpu cuda:0
```

The model is `dreamlabs/dream-1-beta`. The runner uses the installed Inspect Robots action guardrails, saves raw and approved actions in the native trace, and writes `dream-transcript.json`. Policy observations are RGB top/left-wrist/right-wrist images resized to 320 x 180 plus the measured 14-D joint/gripper state; the saved top-view video remains 1920 x 1080. The client performs its standard frame preroll and pauses simulation for inference. It does not receive electrode positions or scorer metrics as policy inputs. Keep the API key in the configured credential store/environment. Do not place credentials in this task or the PR. Record the actual result, including a failed valid trial; do not infer its score from the scripted demo. The scripted recorder's wrist sensors are 320 x 180 and its trial setup changes station base positions, so the model's observation/action description must match this actual configuration.

The benchmark repository's `tools/upload_media.py` uploaded the Dream video, full-score scripted demo and **initial** setup screenshot as M-HY4TM8. The final image is not used as the setup screenshot.

## Real equipment references and fidelity

[Natus Grass stamped-cup electrodes](https://natus.com/neuro/grass-reusable-stamped-cup-eeg-electrodes/) provides the 10 mm cup-size reference. Its real leads/connectors differ from our enlarged grasp stem and short rigid stub. [Weaver Ten20 manufacturer instructions](https://www.weaverandcompany.com/wp-content/uploads/2021/04/L0034_Ten20-IFU_EN-English.pdf) describes an adhesive conductive paste applied with non-gelled electrodes; it does not validate the simulator's permanent joint or contact thresholds. [ACNS electrode-position nomenclature](https://www.acns.org/UserFiles/file/Guideline2-GuidelinesforStandardElectrodePositionNomenclature_v1.pdf) grounds the site names. This simulation uses authored reachable coordinates, not measured clinical 10-20 locations, and the card order is not a universal clinical placement order.

The enlarged ellipsoid head and 410 mm arm spacing are robot-training accommodations. The bare scalp has no sockets or rails. Paste is a rigid visual pad with ideal contact-triggered adhesion: no peeling, pull-off force or removal. Skin preparation, tissue response, pressure limits, full flexible leads, electrical contact/impedance and EEG signal quality are omitted. Arm/head collisions are not assigned a safety-failure score. A successful demo establishes this simulation's placement endpoint, not safe clinical use or the feasibility of a physical replica. Equipment procurement, costs and a repeatable physical paste/retention check remain unvalidated.
