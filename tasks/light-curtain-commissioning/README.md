# Light-curtain commissioning trainer

Native Isaac Sim / Isaac Lab task for bimanual press-safeguarding commissioning. The scripted reference reaches full score; the hosted model baseline is recorded separately during submission preparation.

The left arm supports and raises an optical carrier while the right withdraws and reinserts a physical height pin. The robot checks the front opening with the circular test shaft, operates the physical TEST/RESET/START buttons, demonstrates start inhibition and a moving stop, parks the rod, and completes an empty guarded cycle. Alignment and mounting locks are passive native mechanisms. The simplified press controller does not use the scorer's coverage answers.

The opening is 260 mm wide and spans 50–210 mm above the table. The carrier starts too low. The test shaft is 14 mm in diameter and approximately 249 mm end-to-end; its flat rear handle stays outside the field. Exact original geometry and fixed poses are in `geometry.py`; `task.json` describes scoring and fidelity.

## Run and record

Use a native Isaac Sim 6.1 / Isaac Lab environment with Isaac Lab-Arena, the calibrated YAM package and its converted USD assets installed. This folder contains the task, native wrapper, original labels and recorder. Keep generated assets, recordings and caches on the large storage volume.

From this folder, with ISAAC_PY pointing to the environment's Python executable:

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" tools/run_recovery_pilot.py . /absolute/path/to/fresh-output --policy commission --device cpu --seed 0 --width 1920 --height 1080
```

The EULA environment setting assumes the operator has accepted NVIDIA's terms. Native CPU PhysX runs at 240 Hz; robot control and recordings run at 20 Hz. Both `top_cam.mp4` and `close_cam.mp4` are native camera recordings encoded as H.264/yuv420p MP4, without added borders. The action budget is 3,600 controls / 180 simulated seconds. Wall time depends on hardware; `receipt.json` records the measured run time.

`receipt.json`, `trace.jsonl`, the 240 Hz `native-light-curtain-events.jsonl`, timing records, starting screenshots and frozen source copies remain beside each video. Full feasibility requires a completed `commission` run with final score 1.0, rod parked and hands clear; a preview or controls-only diagnostic is insufficient.

For calibration, `preview`, `mount_only`, `controls_only`, `low_gap` and `free_frame` policies are available. `controls_only` deliberately omits the coverage sweep and cannot earn full credit. Pure logic checks can be run with `python test_rules.py`; they do not establish native feasibility.

Operator buttons register after 100 ms continuous depression; beam trips use a separate 12.5 ms filter. Qualifying inspection requires the actual circular-shaft material-point speed, measured from consecutive native poses at 240 Hz, to stay at or below 40 mm/s. No interpolation across control frames is used. Create a `STOP` file in the output folder to interrupt a calibration cleanly. The script uses a 12 mm/s mean sweep (22.5 mm/s nominal minimum-jerk peak), leaving contact-motion margin.

This is an authored bench trainer with simplified sensing and guarding. It has no human actor or injury measurement. Specialist machine-safety review remains separate from this local simulation demonstration.

## Hosted baseline

With the Dream dependencies from `pyproject.toml` installed and credentials configured through Dream Labs' standard credentials file, run:

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" tools/run_recovery_pilot.py . /absolute/path/to/fresh-model-output --policy dream --model dreamlabs/dream-1-beta --device cpu --seed 0 --width 1920 --height 1080
```

The registered instruction, reset and 3,600-control budget are shared with the reference trial. Dream receives the station camera images and measured 14-D robot state. Native object poses and scoring state are retained for evaluation only. Default Inspect Robots action guards apply. Simulation time pauses while inference is pending. The model transcript and native receipt record requests and session IDs, so any server-driven renewals can be disclosed with the result.

## Verified reference recording

`light-curtain-commissioning-1080p-02` completed 3,075 controls with final score **1.0**. Both native camera videos are **1920 x 1080, 20 fps, H.264/yuv420p**, 3,076 frames / **153.8 s**, with faststart and no added borders. Sizes: 28.20 MB standard view; 28.05 MB close view.

The retained evidence contains 37,020 native physics samples, including 120 warmup samples. All 76 inspection cells qualified. Independent reconstruction from the logged poses verified the material-point speed calculation. Inhibit, moving stop and reset tests passed; the sampled trip-to-stop interval was 54.2 ms. One guarded cycle completed with the ram up, rod parked and both hands clear. Native sensors recorded no inter-arm contact and no robot-to-mount contact during the inspection sweep. The entire scripted ending was retained.

The frozen source and retained evidence hashes verified, and both MP4s fully decoded. Later edits are confined to documentation and task metadata; scene, controller and scorer code match the recorded source. This section describes the scripted feasibility demonstration; model results are reported separately in `task.json`.

## Real-world grounding

The represented job is preparing and checking press safeguarding before production resumes. [SOURCES.md](SOURCES.md) links a real SCHMIDT guarded-press example, SICK inspection and restart guidance, and OSHA's explanation of the human hazard. It distinguishes those references from the original trainer mechanisms and scoring choices.

For an accessible visual explanation, the [OSHA presence-sensing page](https://www.osha.gov/etools/machine-guarding/presses/presence-sensing-devices) includes animations of stroke interruption and incorrect curtain placement. The manufacturer's rod check and the authored powered-stop simulation exercise are separate procedures, as explained in the source notes.

## Related task and scope

The pre-submission board check found related task TK-ESLKJF, which tests two fixed curtains, identifies a randomized faulty beam and places a fault tag. This task's primary mechanical intervention is supporting an adjustable carrier while withdrawing and reinserting its height pin; the installation is then verified on a single powered press trainer with inhibit, finite moving-stop and reset/start checks. It has no randomized dead channel or fault-tag placement. This relationship is disclosed for review rather than treating different dimensions or colours as the contribution.

## Recorded Dream-1 baseline

The native `dreamlabs/dream-1-beta` trial completed all **3,600 controls**, seed 0, with final score **0.0**. The carrier did not reach the scored installation endpoint; inspection and press-control verification were not completed. Its primary video is 180.05 s including the initial frame, 1920 x 1080 at 20 fps, H.264/yuv420p, 38.12 MB. The matching post-settling model-reset screenshot is the submission setup image. The full-score scripted demo is 1.0.

The stock SDK obtained **75 successful action chunks across 149 action attempts**. The server reported **74 expirations**, producing 75 distinct sessions. Physical state and the simulation clock continued across those renewals; model-session continuity is limited by this infrastructure behavior. Interpret the score as a renewed-session evaluation. Every applied action came from Dream under the default Inspect Robots bounds/delta guards.

The model recording has 43,320 native physics samples including 120 warmup samples. It completed without invalid native measurements or inter-arm contact; source and evidence hashes verified. Both camera recordings fully decoded and stayed below 50 MB. The recorded render/evaluation wall time was approximately 38.2 minutes on the measured host, including inference waits. Runtime varies by hardware and service availability.

The 27 Python, pilot, dependency and label-asset files shared with the validated scripted reference matched byte-for-byte. Documentation, score text and media metadata were finalized after recording; the actual scene, controller and scorer were unchanged.

Final packaging normalized whitespace in `geometry.py` and `labels.py`. Their parsed Python ASTs, and those of every submitted Python file, match the recorded model snapshot; all remaining shared code, pilot, dependency and label assets retain their recorded bytes.
