# Used suction liner removal — review prototype

## Task

A procedure has finished. Remove the used suction liner safely, retain the reusable canister, and finish with the liner released inside the closed clinical-waste receiver. There are no loose needles or human-hand animation.

Proposed instruction:

> Remove the used suction liner and put it in the clinical-waste receiver. Leave the reusable canister at the station, close the receiver and finish with everything released.

## Grounding and scope

The [Serres instructions, removal figures 6.1–6.4 and English pages 4–5](https://www.serres.com/wp-content/uploads/2026/08/Serres-Suction-bag-IFU.pdf) describe disconnecting the patient connection, plugging the liner, switching off vacuum and lifting by the handle. Vacuum must not be switched off before closure. The [product range](https://www.serres.com/fluid-collection/product-range/serres-suction-bags/) includes pre-gelled variants. This trainer represents approximately one litre of previously solidified contents, rather than free liquid.

The orange-lidded receiver references [NHS Scotland SHTN 03-01 Table 5.1](https://www.nss.nhs.scot/media/5305/shtn-03-01-v7-oct-2023.pdf) for solidified infectious liquid and suction-set waste, assuming no medicinal or cytotoxic contamination. Actual waste routing follows facility policy. This geometry is not a certified commercial container.

The intended human-safety relevance is reducing staff exposure during removal and waste handling. No infection probability or injury is calculated. Before submission, a practitioner should review the workflow and a wider duplicate screen should be repeated. An earlier local keyword screen is recorded in `serres-suction-bag-board-screen.json`; it does not establish uniqueness. Related dust-bag replacement task TK-KFDQVM needs comparison.

## Fixed apparatus

Original trainer dimensions and friction are engineering estimates, not measured Serres specifications. `geometry.py` is authoritative. Coordinates are metres: x forward, y toward the left arm, z up, origin midway between the YAM bases. Bases are 610 mm apart; tabletop z=747.5 mm.

| Item | Native geometry and start |
| --- | --- |
| Reusable canister | Hollow ring colliders, approximately 152 mm diameter × 190 mm high, 0.38 kg, centred (0.370, 0.105, 0.7595) before settling. |
| Filled disposable liner | Rigid 1.10 kg approximation, 182 mm depth, approximately 132 mm body diameter; lid starts at z=0.9495. Red-brown residue band and dried streaks are visual only. |
| Lifting pads | Two original solid outboard grasp pads at lid-local (-0.015, ±0.100, 0.030), 44 × 12 × 10 mm, with stems and feet. These adapt the trainer to the grippers. |
| Patient connection | Removable rigid elbow and short visual tube; no patient attachment. Catch tray centred (0.290, 0.430, 0.7515). |
| Closure | Free 8 g plug with a passive seven-link tether. Physical port/stem geometry retains it; no software latch or robot attachment. |
| Vacuum control | Passive rotary lever at (0.265, 0.245, 0.7755), labelled ON/OFF. Pressure is not simulated. |
| Clinical-waste receiver | Hollow grey molded tub approximately 232 × 252 × 230 mm, centred (0.300, -0.200, 0.7555). Orange hinged lid opens 150 degrees. Original raised side pad permits physical grasping. |

The cup and receiver are compact reach-adapted training geometry. The canister's clear walls are displayed as molded edge highlights while retaining the complete physical wall colliders. The frosted filled liner and rigid orange bin lining are visual approximations. There is no deformation, fluid flow, pressure, tearing, leakage or biological contamination model. Geometric plug retention does not certify a clinical seal.

## Native scripted sequence

1. Left removes and parks the connector in the catch tray.
2. Left seats and releases the tethered plug.
3. Left rotates the control to OFF only after seating.
4. Left and right grasp separate raised liner pads. Both lift vertically clear of the reusable canister and transfer the liner over the receiver.
5. Both lower the liner inside the receiver and release together. The remaining approximately 56 mm drop occurs inside the bin. Each hand withdraws toward its own side and parks.
6. Right grasps the bin lid's side pad, closes its passive hinge, releases and parks. Final hold permits native settling/contact verification.

Only robot actions are commanded after reset. Measured robot/object poses correct gravity-induced tracking error; there is no hidden attachment, object teleportation or scripted body pose. Stock robot actuator gains and force limits are retained. Equivalent safe strategies are eligible for scoring; no compulsory simultaneous bimanual contact or hand-specific score is imposed.

## Implemented review scorer

`task.py` evaluates actual native state. Process and contact history are sampled at every 240 Hz physics step, robot control and video at 20 Hz.

- Cap seated: port-relative horizontal and vertical errors below 3.5 mm; cap/liner up-axis alignment above 0.97. Target cap origin is PORT + (0, 0, 12 mm).
- Vacuum OFF: actual lever angle above 0.90 rad.
- Latched zero: OFF with cap unseated, extraction above 12 mm with cap unseated or vacuum still ON, or liner lost beyond the table/workspace envelope. Measurement exceptions invalidate the run.
- Liner inside: conservative rotated bounds x±80 mm, y±108 mm, z[-182, +36] mm, plus every tether capsule endpoint/radius, wholly inside bin inner x±107 mm, y±117 mm, z[-2, 230] mm.
- Connector parked: centre within the catch tray's stated native location tolerance (70 mm x, 38 mm y, 40 mm z around the nominal resting origin).
- Released/clear: no native robot contact with the moving apparatus or bin walls/floor for at least one second. Settled: liner speed below 5 mm/s and angular speed below 0.05 rad/s.

| Score | Native state, provided no latched unsafe event |
| --- | --- |
| 0 | No completed safe checkpoint. |
| 0.5 | Cap seated, vacuum OFF, connector removed, liner still supported within 6 mm of its original seating height. |
| 0.75 | Complete liner/tether inside, cap seated, vacuum OFF, connector parked, settled and released; lid remains open. |
| 1.0 | Same containment and release requirements, plus lid within 0.0524 rad (3 degrees) of closed. |

A held liner midair earns no safe-stop credit. Cap, switch and extraction order are physically possible in unsafe configurations; software observes rather than preventing those actions. Inter-arm contacts are reported, and the reference scripted controller aborts if one occurs. The scorer is a local prototype, not yet a published benchmark specification.

## Recording and review

`run_scripted.py` records one continuous native episode into three 1920 × 1080 H.264/yuv420p MP4s: the unchanged calibrated top view, close oblique view and wider overview. No letterboxing or success overlay is added. A source snapshot, execution metadata, 20 Hz trace, 240 Hz contact/process events and initial/final PNGs accompany each attempt. Failed diagnostic attempts remain labelled as such; isolated lid tests do not count as full demonstrations.

Submission preparation now uses the neutral registered instruction above. See README.md and task.json for the final task fields, endpoint scorer, real-world scoring route and actual evaluation results.
