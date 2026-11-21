# Implemented scorer

This document describes the current code, including its limits. It does not add conditions that are absent from `task.py`.

## Contact, current placements and first-placement events

At 20 Hz, obtain each native cup center and orientation. Create a paste bond using the geometric contact rule in [SETUP.md](SETUP.md). No correct-site check is used to create the bond.

A sample qualifies for an electrode if it has a bond, Euclidean cup-center target error is at most **14 mm**, and its last **six samples** are each at most **10 mm** from the oldest sample. When this remains true for **0.25 s**, it becomes a current eligible placement. Its first qualifying time is latched as a placement event. The six-sample history spans 0.25 s at 20 Hz. Instantaneous native velocity is logged but is not the stability gate.

A lost qualification resets the electrode's current stability timer. Its historical first event stays latched. Incorrect-site bonds are possible and receive no current placement credit. Bond creation alone is insufficient for placement credit. The scorer does not require a particular arm, simultaneous contact or electrical measurements.

## Partial score

Let **N** be the number of current eligible placements. Sort latched events by their first qualifying times, and let **P** be the number of initial events matching the required sequence Fp1, Fp2, C3, C4, O1, O2 before the first mismatch. Compute:

`S = min(0.6 * N/6 + 0.4 * P/6, 0.999)`

Examples: nothing done gives 0; one correctly ordered current placement gives 1/6; three correctly ordered current placements give 0.5; six current placements in the wrong order cannot give 1.0. Historical ordered-prefix credit can survive a later lost placement. Wrong-site placements and dropped electrodes have no separate latched-zero hazard rule; other earned credit can remain. A geometric bond cannot subsequently peel off in the ideal-adhesion model.

## Full endpoint

N=P=6 and both TCPs must satisfy:

- Horizontal distance at least **200 mm** from world point **(430,0)** mm.
- World height at least **870 mm** (120 mm above the task construction datum).

These tests use the TCP points, not all arm collision geometry or a per-electrode fingertip clearance. The task returns endpoint score 1.0 while those conditions hold. The runtime accepts a full trial result only after **20 consecutive endpoint control steps**, or **1 s**. Until that hold completes, it reports the previous partial score. Stop on success or after 3,000 controls.

Missing/invalid native body data raises an exception; the run must remain invalid rather than be reported as a successful trial. No force, scalp-injury, arm collision, cable-tension or diagnostic-quality criterion is implemented. Those are not implied by this score.

## Physical inspection proposal

Use labeled cup centers, an unobscured top/side video and the target coordinates to measure placement; use frame order for first placement events and a ruler for final hand clearance. Record a final one-second hold. This is a proposed inspection method, not a validated real-trial counterpart: the simulator's ideal bonds and native sample history have no demonstrated physical equivalence, and placement labels refer to an enlarged authored layout. Validate obtainable electrodes, mannequin geometry, connector handling and actual paste retention before making a physical-reproducibility claim.
