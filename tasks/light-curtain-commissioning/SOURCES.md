# Real-world references and asset provenance

This task is an original, simplified assembly-press commissioning trainer. The following primary sources ground its job context and protective-device concepts. The scene is not a digital twin of a named manufacturer's machine, and its dimensions, mounting pins, forces, timings and scoring tolerances are authored simulation parameters.

## Machinery reference

[SCHMIDT Technology: guarded hydro-pneumatic press case study](https://www.schmidtpresses.com/resources/case-study/monitored-table-curtain-guarding/)

The manufacturer describes a commercial assembly station combining a press, three-sided polycarbonate guarding and a front light curtain. This is a concrete example of the machinery family represented by our trainer. Its rotary table, production tooling and industrial specifications are outside our simulation.

[SCHMIDT Technology: press workstations and safety concepts](https://www.schmidttechnology.de/en/products/press-systems/presses-automation-components-and-complete-systems/)

This overview presents light-curtain guarding as an available press-workstation configuration. Use it as a product-family reference, not as evidence that every configuration uses our RESET/START sequence.

## Inspection and restart concepts

[SICK deTec4 Core operating instructions, document 8014253/1S7O, 2025-04-08](https://www.sick.com/media/docs/1/11/011/operating_instructions_detec4_core_en_im0048011.pdf)

- Section 4.5.1, printed pages 26–27: inspection with the specified opaque rod, a slow pass through the opening and around its boundary, and observation of the receiver's red status.
- Section 4.4.1, printed page 24: an externally implemented restart interlock can require reset before a separate restart.

The manual keeps hazardous machine operation disabled during its rod inspection. Our initial scan likewise runs with the simulated press inactive. The later powered, empty moving-stop exercise is an authored simulation test; it is not presented as the manufacturer's rod-inspection procedure. Our 40 mm/s limit, 76 cells and dwell thresholds are benchmark choices.

## Human-safety relevance and animations

[OSHA: presence-sensing protection on presses](https://www.osha.gov/etools/machine-guarding/presses/presence-sensing-devices)

OSHA explains how presence sensing inhibits or interrupts a hazardous stroke and why other access routes need guarding. Its accident example describes a low-mounted curtain leaving a worker exposed to a crushing injury. This supports the task's human-safety motivation; the current simulation scores commissioning behavior rather than injury.

The same page contains useful animations. Open the page and use its **Start/Stop Animation** controls, particularly the examples of a hand interrupting the descending stroke and an incorrectly positioned curtain. These are linked explanatory references; the Isaac Sim recordings remain the actual task demonstrations.

[OSHA: protective-device safety distance](https://www.osha.gov/etools/machine-guarding/presses/safety-distance)

This explains why stopping response matters to protective-device placement. The trainer's measured stop time does not establish an approved industrial installation distance.

## Mapping to the simulation

| Simulation element | Grounding or authored choice |
| --- | --- |
| Guarded press with a protected front opening | SCHMIDT machinery examples; OSHA guarding concepts |
| Rod inspection and receiver status | SICK inspection guidance |
| External reset before a fresh start | SICK restart-interlock concept; authored controller configuration |
| Adjustable carrier and positive indexing pins | Original bimanual mechanism |
| Powered empty-stop exercise | Original simulation-only functional test |
| Coloured beam graphics | Teaching visualization of the sensing field |
| Numerical inspection grid and thresholds | Explicit benchmark scoring choices |

All simulated apparatus geometry, labels and mechanisms are procedurally authored in this task. The YAM robot and station come from the dependency identified in `pyproject.toml`. External references are linked for context; manufacturer branding and certification are not attributes of this trainer.
