"""Original mechanism-preview geometry; dimensions are trainer estimates, not Serres CAD."""
import math
from pathlib import Path

import numpy as np

from resist_isaac.recovery import body, box, description

FOLDER = Path(__file__).parent
TABLE_Z = .7475
CANISTER = np.array([.370, .105, TABLE_Z + .012])
CANISTER_HEIGHT = .190
LINER_DEPTH = .182
LID = CANISTER + [0, 0, CANISTER_HEIGHT]
PORT = np.array([-.028, -.022, .011])
PORT_RADIUS = .008
PORT_INSERT_THICKNESS = .002
CAP_SEAT = PORT + [0, 0, .012]
RECEIVER = np.array([.300, -.200, TABLE_Z + .008])
CONTROL = np.array([.265, .245, TABLE_Z + .028])
BIN_OPEN_ANGLE = math.radians(150)
BIN_HEIGHT = .230
BIN_HINGE = np.array([.110,0,BIN_HEIGHT+.009])
BIN_HANDLE = np.array([-.060,-.190,.040])
CATCH = np.array([.290,.430,TABLE_Z+.004])
HANDLES = {'left':np.array([-.015,.100,.030]),'right':np.array([-.015,-.100,.030])}


def cylinder(radius, half_height, position=(0, 0, 0), color=(.7, .7, .7, 1), **kwargs):
    return dict(type='cylinder', size=[radius, half_height], position=list(position),
                color=list(color), **kwargs)


def ring(radius, thickness, half_height, z=0, color=(.7, .8, .8, .22), n=32):
    """Separate wall colliders preserve the hollow interior."""
    shapes = []
    for i in range(n):
        a = 2 * math.pi * i / n
        shapes.append(box((thickness / 2, radius * math.tan(math.pi / n), half_height),
                          (radius * math.cos(a), radius * math.sin(a), z), color,
                          orientation=[math.cos(a / 2), 0, 0, math.sin(a / 2)]))
    return shapes


def cap_lip(color):
    """Rounded-insertion profile for the compliant replica contact approximation."""
    profile = [(-.0095, .0072), (-.0069, .00825), (-.0058, .00825), (-.0048, .0076)]
    n = 32
    points = [[r*math.cos(2*math.pi*i/n), r*math.sin(2*math.pi*i/n), z]
              for z, r in profile for i in range(n)]
    faces = [list(reversed(range(n))), list(range((len(profile)-1)*n, len(profile)*n))]
    for level in range(len(profile)-1):
        for i in range(n):
            a, b = level*n+i, level*n+(i+1)%n
            faces.append([a, b, b+n, a+n])
    return dict(type='mesh', mesh=dict(points=points, faces=faces), color=list(color),
                compliant_fit=True, friction=[.75, .6])


def rounded_tub(hx, hy, z0, z1, radius, wall, color):
    """Visual molded shell; separate primitive colliders keep the opening hollow."""
    outline = []
    for cx, cy, start in ((hx-radius, hy-radius, 0), (-hx+radius, hy-radius, 90),
                           (-hx+radius, -hy+radius, 180), (hx-radius, -hy+radius, 270)):
        for a in np.linspace(math.radians(start), math.radians(start+90), 9):
            outline.append((cx+radius*math.cos(a), cy+radius*math.sin(a)))
    n = len(outline)
    inner = [(x*(hx-wall)/hx, y*(hy-wall)/hy) for x, y in outline]
    points = [[x,y,z] for contour,z in ((outline,z0),(outline,z1),(inner,z0),(inner,z1))
              for x,y in contour]
    faces = []
    for i in range(n):
        j = (i+1)%n
        faces += [[i,j,n+j,n+i], [2*n+j,2*n+i,3*n+i,3*n+j],
                  [n+i,n+j,3*n+j,3*n+i], [j,i,2*n+i,2*n+j]]
    return dict(type='mesh', collision=False, color=list(color), roughness=.65,
                mesh=dict(points=points,faces=faces))


def used_liner_marks():
    """Deterministic residue on the disposable liner only, with no collision or mass."""
    n = 128
    angles = np.arange(n)*2*math.pi/n
    boundary = lambda a: -LINER_DEPTH+.090+.012*math.sin(3*a)+.007*math.sin(7*a+.4)
    points = [[.06615*math.cos(a),.06615*math.sin(a),z]
              for high in (False,True) for a in angles for z in [boundary(a) if high else -LINER_DEPTH+.001]]
    faces = [[i,(i+1)%n,(i+1)%n+n,i+n] for i in range(n)]
    shapes = [dict(type='mesh',collision=False,color=[.22,.047,.025,1],roughness=.82,
                   mesh=dict(points=points,faces=faces))]
    for i in range(11):
        angle = i*2*math.pi/11+.12
        bottom = boundary(angle)-.012
        top = -.040+.014*math.sin(i*2.1)
        points = []
        for t in np.linspace(0,1,18):
            centre = angle+.014*math.sin(t*4+i)
            half = .009+.017*(1-t)+.004*math.sin(t*8+i)
            points += [[.06625*math.cos(a),.06625*math.sin(a),bottom+t*(top-bottom)]
                       for a in (centre-half,centre+half)]
        faces = [[2*j,2*j+1,2*j+3,2*j+2] for j in range(17)]
        shapes.append(dict(type='mesh',collision=False,color=[.30,.065,.035,1],roughness=.9,
                           mesh=dict(points=points,faces=faces)))
    return shapes


def clear_skin(radius, low, high):
    # A continuous thin visual wall avoids refractive seams between primitive colliders.
    n = 96
    points = [[radius*math.cos(i*2*math.pi/n),radius*math.sin(i*2*math.pi/n),z]
              for z in (low,high) for i in range(n)]
    return dict(type='mesh',collision=False,clear_plastic=True,visible=False,color=[.95,.98,.97,.1],
                mesh=dict(points=points,faces=[[i,(i+1)%n,(i+1)%n+n,i+n] for i in range(n)]))


def lid_sectors():
    """Convex sectors leave a real off-centre hole for the patient port."""
    shapes = []
    for i in range(32):
        a, b = 2 * math.pi * np.array([i, i + 1]) / 32
        polygon = [[.080 * math.cos(a), .080 * math.sin(a)],
                   [.080 * math.cos(b), .080 * math.sin(b)],
                   [PORT[0] + .009 * math.cos(b), PORT[1] + .009 * math.sin(b)],
                   [PORT[0] + .009 * math.cos(a), PORT[1] + .009 * math.sin(a)]]
        points = [[x, y, z] for z in (-.003, .003) for x, y in polygon]
        shapes.append(dict(type='mesh', color=[.87, .90, .87, 1],
                           mesh=dict(points=points, faces=[[0, 3, 2, 1], [4, 5, 6, 7],
                               [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])))
    return shapes


def label(text, position, size, color=(.91, .95, .95)):
    from PIL import Image, ImageDraw, ImageFont
    path = FOLDER / 'assets' / (''.join(c if c.isalnum() else '_' for c in text) + '.png')
    path.parent.mkdir(exist_ok=True)
    width, height = (round(v * 10000) for v in size)
    image = Image.new('RGB', (width, height), tuple(round(c * 255) for c in color))
    draw = ImageDraw.Draw(image)
    font_file = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    font = ImageFont.truetype(font_file, height)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    font = ImageFont.truetype(font_file, max(1, int(height * min(.92 * width / (r-l), .72 * height / (b-t)))))
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text(((width-r-l)/2, (height-b-t)/2), text, font=font, fill=(12, 25, 30))
    if not path.exists():
        image.save(path)
    w, h = size
    return dict(type='mesh', position=list(position), collision=False, texture='assets/' + path.name,
                roughness=1., mesh=dict(points=[[-h/2, -w/2, 0], [-h/2, w/2, 0],
                    [h/2, w/2, 0], [h/2, -w/2, 0]], faces=[[0, 3, 2, 1]],
                    uv=[[1, 0], [1, 1], [0, 1], [0, 0]]))


def geometry():
    steel = (.46, .52, .55, 1)
    fixture = [box((.116, .112, .005), CANISTER - [0, 0, .006], (.16, .24, .27, 1))]
    # A tabletop bracket retains the canister at its base, without filling its interior.
    fixture += ring(.083, .005, .012, CANISTER[2]+.012, steel)
    for shape in fixture[1:]:
        shape['position'][0] += CANISTER[0]
        shape['position'][1] += CANISTER[1]
    fixture += [label('USED SUCTION LINER', (.220, .105, TABLE_Z+.001), (.205, .030)),
                label('VACUUM', (.212, .285, TABLE_Z+.001), (.112, .027)),
                label('ON', (.315, .245, TABLE_Z+.001), (.045, .025), (.67, .94, .76)),
                label('OFF', (.308, .330, TABLE_Z+.001), (.045, .025), (.96, .77, .65)),
                label('CLINICAL WASTE', (.135, -.200, TABLE_Z+.001), (.210, .032), (.97, .82, .56))]
    # Catch tray for the detached connector and its short tube.
    catch = CATCH
    fixture.append(box((.080, .047, .003), catch, (.22, .41, .48, 1)))
    for sign in (-1, 1):
        fixture += [box((.003, .047, .009), catch+[sign*.083, 0, .006], steel),
                    box((.080, .003, .009), catch+[0, sign*.050, .006], steel)]
    assets = {'fixture': description('Fixture', [body('frame', fixture, static=True)])}

    # Compact clinical-waste receiver: molded body, orange rim/lid and real hinge.
    # Original trainer geometry, not a certified commercial product replica.
    tub = [box((.114,.124,.004), (0,0,0), (.80,.83,.84,1))]
    for sign in (-1,1):
        tub += [box((.004,.120,(BIN_HEIGHT-.006)/2), (sign*.110,0,(BIN_HEIGHT+.002)/2), visible=False),
                box((.106,.004,(BIN_HEIGHT-.006)/2), (0,sign*.120,(BIN_HEIGHT+.002)/2), visible=False)]
    tub += [rounded_tub(.116,.126,.003,BIN_HEIGHT-.008,.021,.008,(.82,.85,.86,1)),
            rounded_tub(.119,.129,BIN_HEIGHT-.014,BIN_HEIGHT,.022,.011,(.97,.39,.055,1)),
            rounded_tub(.107,.117,.006,BIN_HEIGHT-.016,.017,.001,(.87,.39,.10,1))]
    # Printed markings belong to the bin; no team identity or certification marks.
    for text,z,size,bg in [('INFECTIOUS WASTE',.205,(.190,.026),(.97,.97,.94)),
                           ('\u2623',.143,(.074,.074),(.97,.97,.94)),
                           ('SEALED LINERS ONLY',.077,(.190,.025),(.97,.97,.94))]:
        decal = label(text,(-.1166,0,z),size,bg)
        decal['orientation'] = [2**-.5,0,-2**-.5,0]
        tub.append(decal)
    lid_shapes = [box((.117,.127,.006),(-.110,0,0),(.97,.39,.055,1)),
                  box((.105,.115,.0015),(-.110,0,-.0075),(.13,.16,.17,1)),
                  box((.007,.035,.008),(-.215,0,.012),(.25,.28,.29,1))]
    lid_shapes += [dict(type='capsule',size=[.006],
        fromto=[*(BIN_HANDLE-[.020,0,0]),*(BIN_HANDLE+[.020,0,0])],color=[.25,.28,.29,1])]
    lid_shapes.append(box((.008,.005,.019),(-.060,-.190,.021),(.25,.28,.29,1)))
    lid_shapes.append(box((.012,.034,.003),(-.060,-.158,.003),(.25,.28,.29,1)))
    hinge = BIN_HINGE
    lid = body('lid',lid_shapes,.20,RECEIVER+hinge)
    lid['orientation'] = [math.cos(BIN_OPEN_ANGLE/2),0,math.sin(BIN_OPEN_ANGLE/2),0]
    assets['waste_bin'] = description('ClinicalWasteBin',[
        body('tub',tub,static=True,position=RECEIVER),lid],joints=[dict(name='lid_hinge',
            type='revolute',parent='tub',child='lid',axis='Y',position_parent=hinge.tolist(),
            lower=0.,upper=BIN_OPEN_ANGLE,stiffness=0.,damping=.03,collision=True)],
            solver_position_iterations=32,solver_velocity_iterations=8)

    canister = [cylinder(.076, .004, (0, 0, .004), (.65, .80, .84, .28))]
    walls = ring(.076,.003,(CANISTER_HEIGHT-.016)/2,CANISTER_HEIGHT/2)
    for shape in walls:
        shape['visible'] = False
    canister += walls
    canister += [clear_skin(.0776,.008,CANISTER_HEIGHT-.008)]
    # Clear walls are represented by their molded edges, without refractive distortion.
    for y in (-.0765,.0765):
        canister.append(box((.001,.001,(CANISTER_HEIGHT-.016)/2),(0,y,CANISTER_HEIGHT/2),(.69,.80,.81,1),collision=False))
    canister += ring(.077,.004,.003,CANISTER_HEIGHT-.007,(.47,.61,.64,.8))
    # External stabilization feature, drawn as a normal rim/side handle.
    canister += [box((.025,.005,.006),(0,.097,CANISTER_HEIGHT-.060),steel)]
    for x in (-.024, .024):
        canister.append(box((.004,.013,.006),(x,.086,CANISTER_HEIGHT-.060),steel))
    # Grey vacuum connector and rear line remain with the reusable apparatus.
    canister += [cylinder(.010,.017,(.093,.015,CANISTER_HEIGHT-.015),(.32,.36,.38,1)),
                 dict(type='capsule',size=[.006],fromto=[.093,.015,CANISTER_HEIGHT+.002,.170,.015,CANISTER_HEIGHT+.002],
                      color=[.35,.43,.44,1], collision=False)]
    assets['canister'] = description('Canister', [body('shell', canister, .38, CANISTER)])

    # ponytail: rigid filled-body preview only; flexible extraction must be validated separately.
    liner = [cylinder(.066,(LINER_DEPTH-.012)/2,(0,0,-(LINER_DEPTH+.012)/2),(.73,.79,.72,.12),visible=False),
             clear_skin(.0663,-LINER_DEPTH,-.012),
             cylinder(.06605,(LINER_DEPTH-.012)/2,(0,0,-(LINER_DEPTH+.012)/2),(.77,.75,.68,1),collision=False,roughness=.85),
             cylinder(.064,.038,(0,0,-LINER_DEPTH+.043),(.34,.09,.055,1),collision=False)]
    liner += used_liner_marks()
    for offset in HANDLES.values():
        liner.append(box((.022,.006,.005),offset,(.85,.88,.84,1)))
        sign = float(np.sign(offset[1]))
        liner.append(box((.008,.004,.012),(-.015,sign*.100,.014),(.85,.88,.84,1)))
        liner.append(box((.010,.020,.003),(-.015,sign*.083,.005),(.85,.88,.84,1)))
    # Normal closed unused serial port, with no extra manipulation requirement.
    liner += [cylinder(.017, .010, (.032, .028, .013), (.82, .86, .83, 1))]
    neck = ring(.0105, .003, .007, PORT[2]-.001, (.88, .91, .87, 1), n=32)
    # The 2 mm insert has shoulder support and a clear underside for the lip.
    insert = ring(PORT_RADIUS+.0015, .003, PORT_INSERT_THICKNESS/2,
                  CAP_SEAT[2]-.003-PORT_INSERT_THICKNESS/2, (.27, .32, .34, 1))
    for shape in insert:
        shape['compliant_fit'] = True
    neck += insert
    for shape in neck:
        shape['position'][0] += PORT[0]
        shape['position'][1] += PORT[1]
    liner += neck
    part = body('lid_and_load', liner, 1.10, LID)
    # Gel dominates the preview inertia; the thin lid sectors add collision detail only.
    part['shapes'] += lid_sectors()
    cap_position = LID + [-.029, .037, .018]
    rubber = (.21, .27, .26, 1)
    cap = body('closure', [cylinder(.0115, .003, color=rubber),
                          cylinder(.0076, .001, (0,0,-.004), rubber),
                          cylinder(.008, .00235, (0,0,-.00715), rubber)], .008, cap_position)
    cap['shapes'][-1] = cap_lip(rubber)
    parts, joints = [part, cap], []
    # Small passive jointed tether; it does not attach the cap to the gripper or seat it.
    start, end = LID + [-.060,-.010,.008], cap_position + [.010,0,0]
    bend = LID+[-.054,.044,.009]
    nodes = np.vstack((np.linspace(start,bend,5),np.linspace(bend,end,4)[1:]))
    for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        centre = (a+b)/2
        parts.append(body(f'tether{i}', [dict(type='capsule', size=[.0012],
                     fromto=[*(a-centre), *(b-centre)], color=[.82,.89,.82,1])], .00008, centre))
        parent = 'lid_and_load' if i == 0 else f'tether{i-1}'
        parent_position = LID if i == 0 else (nodes[i-1]+a)/2
        joints.append(dict(name=f'tether_joint{i}', type='spherical', parent=parent,
                      child=f'tether{i}', position_parent=(a-parent_position).tolist(),
                      position_child=(a-centre).tolist(), stiffness=0., damping=.00002,
                      angular_limit=1.5, collision=False))
    joints.append(dict(name='closure_tether', type='spherical', parent=f'tether{len(nodes)-2}', child='closure',
                      position_parent=((nodes[-1]-nodes[-2])/2).tolist(),
                      position_child=[.010,0,0], stiffness=0., damping=.00002, collision=False))
    assets['liner'] = description('Liner', parts, joints=joints,
                                solver_position_iterations=32, solver_velocity_iterations=8)

    connection = LID + PORT + [0,0,.012]
    elbow = [cylinder(.0079, .012, (0,0,-.008), (.92,.94,.90,1)),
             box((.012,.011,.009), (0,0,.009), (.92,.94,.90,1)),
             dict(type='capsule', size=[.008], fromto=[0,0,.009,.045,0,.009],
                  color=[.92,.94,.90,1]),
             dict(type='capsule', size=[.005], fromto=[.045,0,.009,.105,.045,-.005],
                  color=[.70,.77,.73,.7], collision=False)]
    assets['connector'] = description('Connector', [body('elbow', elbow, .025, connection)])

    housing = body('housing', [box((.028,.030,.012), color=(.20,.28,.31,1))],
                   static=True, position=CONTROL)
    lever = body('lever', [cylinder(.016,.005, color=(.52,.58,.59,1)),
                 box((.018,.004,.006), (.010,0,.012), (.16,.65,.33,1))], .025,
                 CONTROL+[0,0,.018])
    assets['vacuum'] = description('Vacuum', [housing, lever], joints=[dict(name='switch',
                type='revolute', child='lever', axis='Z',
                position_parent=(CONTROL+[0,0,.018]).tolist(), lower=0., upper=1.2,
                stiffness=0., damping=.005, collision=False)])
    return assets
