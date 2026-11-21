# Fixed EEG training layout

All coordinates are world millimetres. x points forward, y left and z up. Moving the arm bases to x=200 does not redefine the coordinate origin. The tabletop is at z=747.5; task geometry uses z=750 as its construction datum. Fixed task shapes begin 2.5 mm above the table plane; no support/contact validation of that gap is claimed. See `geometry.py` for full procedural shape definitions.

## Station and head

- Table: wooden appearance, 1100 x 1300 x 60 mm, center (360,0,717.5).
- Left/right YAM base positions: (200,205,750) and (200,-205,750), 410 mm apart. This differs from the calibrated 610 mm station and must also be used for Dream-1.
- Initial joints: each arm [0,1,1,-0.6,0,0] radians; normalized gripper action 1 (open).
- Fixed ellipsoid scalp: center (440,0,850), radii (120,127.5,94) mm; enlarged for robot clearance, with skin-tone polymer appearance. Both the visual scalp and the convex collision mesh follow this geometry. Mesh resolution: 24 rings, 48 sectors.
- Ear proxies: ellipsoids with radii (19,16.5,27), centers (446,+/-129,858). Nose: radii (14.4,10,15), center (330.8,0,843). Eye landmarks: radius 4, centers (344,+/-39,830).
- Fixed support coaxial with z at x=440,y=0: cylinders (diameter,height,center-z) = (78,8,754), (60,68,792), (110,6,829), (108,8,836). Metal-style stand and skin-tone collar. They are part of the static fixture, not separate movable objects.
- No gray pads, rails, target sockets or colored spots exist on the scalp.

## Electrode positions

Cup centers, not connector tips, determine target error. Start orientation is identity, with the cup axis upward and the lead pointing along -x. The work-card order is the table order. Colors and this order are task conventions, not clinical standards.

| Site/order | Color | Initial cup center (x,y,z) | Target cup center (x,y,z) |
| --- | --- | --- | --- |
| Fp1 | red | (245.0, 190.0, 757.5) | (377.6, 32.0, 930.247) |
| Fp2 | blue | (245.0, -190.0, 757.5) | (377.6, -32.0, 930.247) |
| C3 | yellow | (310.0, 190.0, 757.5) | (440.0, 90.0, 920.083) |
| C4 | green | (310.0, -190.0, 757.5) | (440.0, -90.0, 920.083) |
| O1 | white | (375.0, 190.0, 757.5) | (502.4, 47.8, 925.645) |
| O2 | black | (375.0, -190.0, 757.5) | (502.4, -47.8, 925.645) |

Target height is `850 + 94*sqrt(1-(dx/120)^2-(y/127.5)^2) + 3.5`, with `dx=x-440`. These fixed authored targets borrow EEG names; they are not computed from measured 10-20 landmarks. There is no hidden randomization.

## Cup assemblies and staging

Each 4 g assembly is one rigid body. The metal-style circular cup is 10 mm diameter and 3 mm thick, centered at its body origin. Paste is a cream cylinder, 12 mm diameter and 2 mm thick, centered 2.5 mm below the cup center. The colored top identifier is 7.6 mm diameter x 1 mm thick at local z=1.8. The insulated grasp stem is 6.4 mm diameter x 12 mm high at local z=8; its top cap is 8 mm diameter x 2 mm high at local z=14. A non-colliding colored lead stub is a 1.8 mm diameter capsule from (-4,0,2) to (-32,0,2). The connector is an enlarged robot handling proxy; it is not a Natus connector replica. Linear/angular damping: 1.5/4.0; solver position/velocity iterations: 64/16. `recovery.body` computes assembly inertia from colliding primitives.

The fixed staging mats are 224 x 108 x 4, centers (310,+/-190,752), with 224 x 4 x 4 outer lips centered (310,+/-242,754). The fixed order card is 172 x 156 x 4 centered (600,205,754), with raised printed-style site codes, numbers and six colored chips. Mats, card, head and stand belong to the same static fixture. Colors represent visual materials; tissue, skin deformation and paste spreading are not simulated.

## Paste contact proxy

For each unbonded electrode, sample the gel's bottom center and twelve points around its 6 mm radius rim, transformed by the current native body orientation. Approximate the closest surface gap as `(min(norm((sample-head_center)/head_radii))-1)*94 mm`. On the upper scalp, a gap from -4 to +1.5 mm and cup-center movement below 50 mm/s between control steps enable an ideal fixed joint at the measured pose. The 4 mm lower allowance accounts for the coarse convex collider and rigid gel proxy. This geometric contact approximation does not verify contact force, cup alignment or true gel deformation.

The joint may form at an incorrect scalp location. It has no attraction to a target and no force/peeling limit. It can engage while the gripper still holds the connector. This is an explicit ideal-adhesion approximation, not a measured material law. Full trial scoring additionally requires the arms to withdraw. Geometry does not model full cables, a recording amplifier or an electrically functioning montage.
