"""Routed oval and dynamic slot cars with retained, separately rotating wheels."""
import json
import math
from pathlib import Path
import numpy as np
from resist_isaac.recovery import body,box,description

FOLDER=Path(__file__).parent
CX=.60
RADIUS=.20
HALF_STRAIGHT=.22
ROAD_Z=.760
SLOT_HALF=.002
WHEEL_RADIUS=26.28*.00028
CAR_Z=ROAD_Z+WHEEL_RADIUS+.0002
# Off-track cars rest on the rear tires and the guide blade's rear lower edge.
# Starting all four tires flat would bury the protruding blade in the tabletop.
OFF_TRACK_PITCH=math.atan2(.0115,.06768)-math.asin(WHEEL_RADIUS/math.hypot(.06768,.0115))
OFF_TRACK_Z=.7475+WHEEL_RADIUS+.02968*math.sin(OFF_TRACK_PITCH)+.0001


def track_coordinates(point):
    """Nearest oval location: signed lane offset, clockwise arc coordinate."""
    x,y=map(float,point[:2]);dx=x-CX; length=4*HALF_STRAIGHT+2*math.pi*RADIUS
    if abs(y)<=HALF_STRAIGHT:
        if dx<0:return -dx-RADIUS,y+HALF_STRAIGHT
        return dx-RADIUS,2*HALF_STRAIGHT+math.pi*RADIUS+HALF_STRAIGHT-y
    cy=math.copysign(HALF_STRAIGHT,y)
    angle=math.atan2(y-cy,dx)
    if y>0:s=2*HALF_STRAIGHT+(math.pi-angle)*RADIUS
    else:s=4*HALF_STRAIGHT+math.pi*RADIUS-angle*RADIUS
    return math.hypot(dx,y-cy)-RADIUS,s%length


def oval_strip(inner,outer,z0,z1):
    points=[];faces=[]
    # Shared vertices join curves and straights without internal end faces.
    for cy,start,end in ((HALF_STRAIGHT,0,math.pi),(-HALF_STRAIGHT,math.pi,2*math.pi)):
        for a in np.linspace(start,end,181):
            points.extend([[CX+r*math.cos(a),cy+r*math.sin(a),z]
                for z in (z0,z1) for r in (inner,outer)])
    for i in range(len(points)//4):
        a=4*i;b=(a+4)%len(points)
        faces.extend([[a,a+1,b+1,b],[a+2,b+2,b+3,a+3],
                      [a,b,b+2,a+2],[a+1,a+3,b+3,b+1]])
    # Static triangle meshes are one-sided: the driving surface must face up.
    faces=[face[::-1] for face in faces]
    return dict(type='mesh',mesh=dict(points=points,faces=faces),collision_approximation='none',
                color=[.055,.06,.065,1],friction=[.8,.65],contact_offset=.0001)


def guide_shape():
    """One convex, rounded guide; compound collider seams can snag mesh edges."""
    points=[[0,0,.0025],[0,0,-.0025]]; faces=[]
    for latitude in range(1,10):
        t=math.pi*latitude/10
        for longitude in range(24):
            a=2*math.pi*longitude/24
            points.append([.006*math.sin(t)*math.cos(a),
                .0008*math.sin(t)*math.sin(a),.0025*math.cos(t)])
    for j in range(24):
        k=(j+1)%24
        faces.extend([[0,2+j,2+k],[1,194+k,194+j]])
        for i in range(8):
            a=2+24*i
            faces.append([a+j,a+24+j,a+24+k,a+k])
    return dict(type='mesh',mesh=dict(points=points,faces=faces),
        position=[0,0,-.009],collision_approximation='convexHull',
        color=[.1,.1,.1,1],friction=[.12,.08],contact_offset=.0001)


def geometry():
    source=json.loads((FOLDER/'assets/SOURCE.json').read_text())
    road=[oval_strip(lo,hi,ROAD_Z-.01,ROAD_Z)
        for lo,hi in ((RADIUS-.043,RADIUS-SLOT_HALF),(RADIUS+SLOT_HALF,RADIUS+.043))]
    road.append(oval_strip(RADIUS-SLOT_HALF,RADIUS+SLOT_HALF,ROAD_Z-.008,ROAD_Z-.006))
    # Copper pickup conductors are visual surfaces, not collision ridges.
    for offset in (-.004,.004):
        rail=oval_strip(RADIUS+offset-.001,RADIUS+offset+.001,ROAD_Z,ROAD_Z+.00003)
        rail.update(collision=False,color=[.66,.49,.25,1]);road.append(rail)
    fixture=dict(name='Track',bodies=[dict(name='track',static=True,shapes=road)])
    assets={'fixture':fixture};poses={'fixture':[0,0,0]}
    colors={'red':(.85,.04,.03,1),'yellow':(.98,.75,.02,1),'blue':(.025,.23,.90,1)}
    for name,color in colors.items():
        visual=dict(type='mesh',mesh_file='assets/body.obj',collision=False,color=[1,1,1,1])
        shapes=[box((.0535,.0262,.003),(0,0,.002),visible=False),
                box((.022,.017,.007),(-.001,0,.015),visible=False),visual,
                dict(type='mesh',mesh_file='assets/paint.obj',collision=False,color=list(color))]
        parts=[body('chassis',shapes,.060)];joints=[]
        for i,center in enumerate(source['wheel_centers_m']):
            shape=dict(type='cylinder',size=[WHEEL_RADIUS,.0041],orientation=[math.sqrt(.5),math.sqrt(.5),0,0],
                       visible=False,friction=[.10,.08] if i<2 else [1.0,.85],contact_offset=.0001)
            visual=dict(type='mesh',mesh_file=f'assets/wheel{i}.obj',collision=False,color=[1,1,1,1])
            parts.append(body(f'wheel{i}',[shape,visual],.003,center))
            joints.append(dict(name=f'axle{i}',type='revolute',axis='Y',parent='chassis',child=f'wheel{i}',
                position_parent=center,collision=False))
        # Nominal uniform ellipsoid inertia, matching the rounded collider.
        radii=np.array([.006,.0008,.0025])
        moments=.002/5*(np.sum(radii**2)-radii**2)
        parts.append(dict(name='guide',shapes=[guide_shape()],mass=.002,
            position=[.044,0,0],center_of_mass=[0,0,-.009],
            inertia=dict(diagonal=moments.tolist())))
        joints.append(dict(name='guide_swivel',type='revolute',axis='Z',parent='chassis',child='guide',
            position_parent=[.044,0,0],lower=-1.4,upper=1.4,stiffness=0,damping=.000001,collision=False))
        assets[name]=description('Car',parts,joints=joints,solver_position_iterations=64,solver_velocity_iterations=16)
    for name,s in INITIAL_S.items():
        guide,yaw=lane_pose(s)
        chassis=guide-.044*np.array([math.cos(yaw),math.sin(yaw)])
        poses[name]=[*chassis,CAR_Z]
    for name,(x,y) in PADS.items():
        color=(.82,.61,.02,1) if name=='yellow' else (.02,.13,.65,1)
        fixture['bodies'][0]['shapes'].append(box((.065,.075,.002),(x,y,.7495),color=color))
    return assets,poses


LENGTH = 4*HALF_STRAIGHT + 2*math.pi*RADIUS
INITIAL_S = {name: .22+i*LENGTH/3 for i,name in enumerate(('red','yellow','blue'))}
PADS = {'yellow': (.25,.22), 'blue': (.25,-.22)}


def lane_pose(s):
    """Guide XY and chassis yaw on the clockwise lane; reset placement only."""
    s %= LENGTH
    if s < 2*HALF_STRAIGHT:
        return np.array([CX-RADIUS,s-HALF_STRAIGHT]), math.pi/2
    s -= 2*HALF_STRAIGHT
    if s < math.pi*RADIUS:
        a=math.pi-s/RADIUS
        return np.array([CX+RADIUS*math.cos(a),HALF_STRAIGHT+RADIUS*math.sin(a)]), a-math.pi/2
    s -= math.pi*RADIUS
    if s < 2*HALF_STRAIGHT:
        return np.array([CX+RADIUS,HALF_STRAIGHT-s]), -math.pi/2
    a=-(s-2*HALF_STRAIGHT)/RADIUS
    return np.array([CX+RADIUS*math.cos(a),-HALF_STRAIGHT+RADIUS*math.sin(a)]), a-math.pi/2
