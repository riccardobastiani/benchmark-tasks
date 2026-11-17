# Procedure, comparisons and asset provenance

## Procedure and human exposure

The [Serres Suction Bag System instructions](https://www.serres.com/wp-content/uploads/2026/08/Serres-Suction-bag-IFU.pdf), removal figures 6.1–6.4 and printed English pages 4–5, distinguish disposable liner and reusable canister. They describe patient-connector removal, closing the patient connection with the supplied plug, switching vacuum off, then lifting by the handle. The warning specifically requires closure before vacuum shutdown. The used liner may contain infectious waste. This is the source for the process rules; it does not establish an infection probability, a validated exposure threshold or fidelity of this trainer's dimensions.

The [manufacturer's range](https://www.serres.com/fluid-collection/product-range/serres-suction-bags/) includes pre-gelled liners. This episode begins with contents already solidified. It does not test solidifier preparation, liquid aspiration or measuring fluid volume.

[NHS Scotland SHTN 03-01, Table 5.1](https://www.nss.nhs.scot/media/5305/shtn-03-01-v7-oct-2023.pdf) informs the orange-lidded infectious-waste receiver appearance. This assumes the specified infectious stream without medicinal/cytotoxic contamination. Actual disposal follows facility policy. The original receiver is not certified as leak-resistant.

## Closest board comparisons

Authenticated board checked 17 November 2026: 643 returned entries. We compared full setup, manipulation and scoring, rather than relying on a keyword absence. No direct suction-liner task appeared in this screen; it is not a guarantee of novelty or ownership. The final pre-publication refresh returned 647 entries, again with no direct suction-liner keyword match.

| Existing task | Overlap | Physical/goal distinction and remaining concern |
| --- | --- | --- |
| [TK-8XAXYE: canister vacuum dust-bag replacement](https://tasks.robocurve.org/board?task=TK-8XAXYE), pending | Removing a loaded bag from a reusable container, disposal and lid closure. | That task extracts a cardboard collar from guide channels and installs a fresh bag. Here the patient connector must be removed, the independently tethered port plug seated and retained, and the vacuum control operated before extraction; no new liner is installed. Shared bag-removal mechanics remain a real overlap. |
| [TK-4AHH77: used procedure tray containing a needle](https://tasks.robocurve.org/board?task=TK-4AHH77), accepted | A disposal request with an unstated precaution protecting clinical/waste staff. | It separates an exposed sharp from ordinary waste. This scene has no sharps: it tests a connected suction-system closure/isolation dependency and transfers the complete retained load. The reasoning pattern alone does not establish distinct manipulation. |
| [TK-TFGQJW: safety-needle sheath](https://tasks.robocurve.org/board?task=TK-TFGQJW), accepted | Contain used material before disposal. | It locks a hinged sheath by pressing against a table. Here a separate tethered plug is inserted into a patient port while preserving the liner in its canister, then retained during lifting. Added bin handling alone is not claimed as novelty. |
| [TK-6BRCAY: cinch and remove a loaded bin liner](https://tasks.robocurve.org/board?task=TK-6BRCAY), rejected | Contain a load before extracting its liner from a reusable bin. | It pulls two drawstrings/cord locks around a flexible mouth and retains loose blocks. This trainer instead seats an independent port plug and isolates a connected suction system; flexible cinching and spill dynamics are not claimed. Closing a load before removal is a shared pattern, not evidence of novelty by itself. |

The earlier local dust-bag screen named TK-KFDQVM; the refreshed board returns the related pending task above under TK-8XAXYE. Previous same-team syringe/tray, cable, backup-power and light-curtain tasks are historical, finally rejected submissions. Their existence is considered in the internal selection assessment; this task does not recycle their apparatus or claim that a medical label creates novelty. No exhaustive external-suite equivalence review or exact-product clinical review has been completed.

## Assets and licenses

- Original trainer geometry, residue meshes and rendered text-label PNGs: this folder, MIT under `LICENSE`. All are generated from primitives or authored polygon coordinates; no manufacturer CAD, photographs, scan, product logo or certification mark is included. Source procedure documents are linked, not redistributed.
- Label lettering is rasterized with DejaVu Sans Bold (Bitstream Vera/DejaVu license). Font source: [DejaVu license](https://dejavu-fonts.github.io/License.html). No font file is bundled; install the system DejaVu fonts to regenerate the included labels.
- Bimanual YAM URDF, visual/collision meshes and converted USD: installed through the pinned `yam-arena` dependency in `pyproject.toml`, originating from [i2rt](https://github.com/i2rt-robotics/i2rt), MIT. The dependency includes `yam_arena/assets/yam/LICENSE.i2rt` and `SOURCE_COMMIT`; this folder does not duplicate the robot assets. Dream-coloured split meshes are the same I2RT model, also MIT.
- [YAM Isaac Lab-Arena adapter](https://github.com/kairosiann/yam-isaac-lab-arena) code is an external dependency, not relicensed by this task. Its README states that the adapter itself has no declared license; this is separate from the MIT robot model. It is not vendored here.
- [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena): Apache-2.0 framework. [Isaac Lab](https://github.com/isaac-sim/IsaacLab): BSD-3-Clause framework, with separately licensed components. Isaac Sim requires NVIDIA's own terms. These runtimes are installed separately; no NVIDIA runtime or asset library is redistributed.
- Inspect Robots and Inspect Robots Dream: MIT; Dream Labs client: Apache-2.0. The hosted policy is a service, not a bundled model or task asset.
- Included `resist_isaac` modules are the native RESIST runtime/helpers from the development source, licensed with the original task code. They generate the original apparatus in the installed simulator.

Stock YAM actuator limits/gains are retained. The liner and bin-lid grasp pads are original trainer additions and are explicitly disclosed; successful use of those pads does not establish performance on an unmodified commercial Serres liner.
