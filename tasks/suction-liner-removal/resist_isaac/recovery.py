"""Three passive recovery pilots; all post-reset object motion is native physics.

USD quaternions are WXYZ; Isaac Lab 3 state and Pose quaternions are XYZW.
These are proposed mechanisms until real reset/trajectory checks pass.
"""

import json
import math
from pathlib import Path
import tempfile

import numpy as np

from .state import pose_state, reset_joints
from .usd_assets import create_asset, read_mesh


def box(size, position=(0, 0, 0), color=(.35, .38, .40, 1), **kwargs):
    return dict(type='box', size=list(size), position=list(position), color=list(color), **kwargs)


def body(name, shapes, mass=.2, position=(0, 0, 0), static=False):
    # Uniform nominal density over actual collision primitives, including offset rails.
    pieces=[]
    for shape in shapes:
        if not shape.get('collision',True): continue
        size=shape['size']; center=np.asarray(shape.get('position',[0,0,0]),dtype=float)
        if shape['type']=='box':
            x,y,z=size; volume=8*x*y*z
            inertia=np.diag([(y*y+z*z)/3,(x*x+z*z)/3,(x*x+y*y)/3])
        elif shape['type'] in ('cylinder','capsule'):
            radius=size[0]; axis=np.array([0.,0.,1.])
            if 'fromto' in shape:
                a,b=np.asarray(shape['fromto']).reshape(2,3)
                length=np.linalg.norm(b-a); axis=(b-a)/length; center+=(a+b)/2
            else: length=2*size[1]
            cylinder=math.pi*radius**2*length
            caps=4*math.pi*radius**3/3 if shape['type']=='capsule' else 0
            volume=cylinder+caps
            axial=(cylinder*.5*radius**2+caps*.4*radius**2)/volume
            transverse=(cylinder*(3*radius**2+length**2)/12
                        +caps*(.4*radius**2+length**2/4+3*length*radius/8))/volume
            inertia=np.eye(3)*transverse+(axial-transverse)*np.outer(axis,axis)
        else:
            raise ValueError('Pilot mass properties require explicit primitive colliders')
        pieces.append((volume,center,inertia))
    total=sum(v for v,_,_ in pieces)
    com=sum(v*c for v,c,_ in pieces)/total
    tensor=sum(mass*v/total*(i+np.eye(3)*np.dot(c-com,c-com)-np.outer(c-com,c-com))
               for v,c,i in pieces)
    return dict(name=name, shapes=shapes, mass=mass, position=list(position), static=static,
                center_of_mass=com.tolist(),inertia={'tensor':tensor.tolist()})


def description(name, parts, **kwargs):
    return dict(name=name, bodies=parts, **kwargs)


def geometry(folder):
    """Build task-owned physical geometry with retained benchmark appearances."""
    cfg=json.loads((folder/'pilot.json').read_text())
    mat=cfg['mat_center']; park=cfg['park_tcp']; kind=cfg['kind']
    mat_half_x=.09 if kind=='tray' else .13
    assets={'fixture':description('Fixture',[body('fixture',[
        box((mat_half_x,.12,.003),mat,(.08,.25,.55,1)),
        box((.04,.04,.001),(park[0],park[1],cfg['table_top']+.001),(.08,.65,.18,1))
    ],static=True)])}
    poses={'fixture':(0,0,0)}
    if kind in ('tray','books'):
        height=mat[2]-.003-cfg['table_top']
        assets['fixture']['bodies'][0]['shapes'].append(box((mat_half_x,.12,height/2),
            (mat[0],mat[1],cfg['table_top']+height/2),(.45,.39,.30,1)))
    if kind=='tray':
        rest=cfg['temporary_rest']; height=rest[2]-cfg['table_top']
        assets['fixture']['bodies'][0]['shapes'].append(box((.07,.012,height/2),
            (rest[0],rest[1],cfg['table_top']+height/2),(.45,.39,.30,1)))
        # Source OBJ is twice LIBERO's authored tray size. Preserve its open interior.
        mesh=read_mesh(folder/'assets/wooden_tray.obj')
        points=np.asarray(mesh['points'])*cfg['tray_scale']
        center=(points.max(axis=0)+points.min(axis=0))/2
        points[:,:2]-=center[:2]; points[:,2]-=points[:,2].min()
        points=points[:,[1,0,2]]; points[:,0]*=-1
        mesh['points']=points.tolist()
        shapes=[dict(type='mesh',mesh=mesh,collision=False,texture='assets/crate_mat_BaseColor.png',color=[1,1,1,1])]
        shapes.append(box((.079,.145,.0035),(0,0,.0035),visible=False))
        # Explicit floor and rails: never convexify the open tray into a solid block.
        for x in (-.080,.080): shapes.append(box((.003,.145,.027),(x,0,.032),visible=False))
        for y in (-.145,.145):
            shapes.append(box((.078,.003,.014),(0,y,.016),visible=False))
            for x in (-.070,.070): shapes.append(box((.006,.004,.027),(x,y,.055),visible=False))
            shapes.append(box((.070,.004,.004),(0,y,.079),visible=False))
        assets['tray']=description('Tray',[body('tray',shapes,.28)])
        poses['tray']=cfg['tray_position']
        for i,y in enumerate((-.025,.025)):
            name=f'payload{i}'
            # Nominal small sealed metal sample tins: intact load, not a fluid model.
            assets[name]=description(name,[body(name,[dict(type='cylinder',size=[.018,.022],
                color=[.65,.68,.72,1],friction=[.45,.35])],.08)])
            poses[name]=[cfg['tray_position'][0],cfg['tray_position'][1]+y,cfg['tray_position'][2]+.030]
    elif kind=='books':
        mesh=read_mesh(folder/'assets/black_book.obj')
        points=np.asarray(mesh['points']); points-=(points.max(axis=0)+points.min(axis=0))/2
        points=points[:,[2,1,0]]*cfg['book_scale']; points[:,2]*=-1
        mesh['points']=points.tolist(); half=(points.max(axis=0)-points.min(axis=0))/2
        for i,pos in enumerate(cfg['book_positions']):
            name=f'book{i}'
            shapes=[dict(type='mesh',mesh=mesh,collision=False,texture='assets/black_book_texture.png',color=[1,1,1,1]),
                    box(half,visible=False,friction=[.65,.5])]
            assets[name]=description(name,[body(name,shapes,.10)]); poses[name]=pos
    else:
        # A free weighted instrument on a stand is physically coupled to a flexible cord.
        # Its motion, not a made-up damage threshold, is the hazard endpoint.
        assets['fixture']['bodies'][0]['shapes'].append(box((.055,.045,.012),(.26,-.20,.7655),(.3,.32,.34,1)))
        parts=[body('device',[box((.045,.035,.013),color=(.18,.19,.22,1)),
               box((.025,.018,.001),(0,0,.014),(.12,.45,.60,1))],.35,cfg['device_position'])]
        parts[0]['shapes'].extend(json.loads((folder/'socket-appearance.json').read_text()))
        anchors=np.asarray(cfg['cable_nodes']); nodes=[anchors[0]]
        for a,b in zip(anchors[:-1],anchors[1:]):
            count=math.ceil(np.linalg.norm(b-a)/.025)
            nodes.extend(a+(b-a)*i/count for i in range(1,count+1))
        joints=[]
        for i,(a,b) in enumerate(zip(nodes[:-1],nodes[1:])):
            center=(a+b)/2; name=f'cord{i:02d}'
            parts.append(body(name,[dict(type='capsule',size=[cfg['cable_radius']],
                fromto=[*(a-center),*(b-center)],color=[.10,.10,.11,1],friction=[.7,.5])],.08*float(np.linalg.norm(b-a)),center))
            parent='device' if i==0 else f'cord{i-1:02d}'
            parent_pos=np.asarray(cfg['device_position']) if i==0 else (nodes[i-1]+a)/2
            tangent=(b-a)/np.linalg.norm(b-a)
            frame=np.r_[1+tangent[2],-tangent[1],tangent[0],0.]
            frame=frame/np.linalg.norm(frame) if np.linalg.norm(frame)>1e-8 else np.array([0.,1.,0.,0.])
            # Two bending axes; axial twist is omitted in this discrete cord model.
            joints.append(dict(name=f'joint{i:02d}',type='spherical',locked_axes=['rotZ'],parent=parent,child=name,
                position_parent=(a-parent_pos).tolist(),position_child=(a-center).tolist(),
                orientation_parent=frame.tolist(),orientation_child=frame.tolist(),
                angular_limit=math.radians(40),
                stiffness=.0002,damping=.00003,collision=False))
        # The free terminal remains connected; only finger contact captures the cord.
        name='plug'; end=nodes[-1]
        parts.append(body(name,[box((.018,.010,.008),(-.018,0,0),color=(.2,.2,.22,1)),
              box((.006,.007,.002),(-.040,0,0),(.65,.66,.68,1))],.012,end))
        joints.append(dict(name='plug_joint',type='fixed',parent=f'cord{len(nodes)-2:02d}',child='plug',
            position_parent=((nodes[-1]-nodes[-2])/2).tolist()))
        # Use the terminal as the asset origin; each rigid link has its own native reset.
        origin=np.asarray(nodes[-1])
        for part in parts:
            part['position']=(np.asarray(part['position'])-origin).tolist()
        assets['cable']=description('Cable',parts,joints=joints,
            solver_position_iterations=32,solver_velocity_iterations=8)
        poses['cable']=nodes[-1].tolist()
    return cfg,assets,poses


def export_assets(folder,directory):
    _,assets,_=geometry(Path(folder))
    return {name:create_asset(asset,Path(directory)/f'{name}.usda',folder) for name,asset in assets.items()}


class RecoveryPilot:
    hold_s=1.0
    robot_contact_offset=.001

    def __init__(self,folder):
        self.folder=Path(folder)
        self.cfg,self.descriptions,self.poses=geometry(self.folder)
        self._unsafe=False

    def build_scene(self,embodiment):
        from isaaclab_arena.assets.object import Object,ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.utils.pose import Pose
        self._temporary=tempfile.TemporaryDirectory(prefix='recovery-pilot-')
        paths=export_assets(self.folder,self._temporary.name)
        result=[]
        for name,path in paths.items():
            kind=ObjectType.BASE if name in ('fixture','cable') else ObjectType.RIGID
            result.append(Object(name=name,object_type=kind,usd_path=str(path),
                initial_pose=Pose(position_xyz=tuple(self.poses[name]))))
        parent=next((obj for obj in result if obj.name=='cable'),None)
        if parent is not None:
            for part in self.descriptions['cable']['bodies']:
                result.append(ObjectReference(parent_asset=parent,name='cable_'+part['name'],
                    prim_path=parent.get_prim_path()+'/'+part['name'],object_type=ObjectType.RIGID))
        return result

    def initial_action(self):
        from yam_arena import ik,station
        from yam_arena.constants import READY
        q,pe,re=ik.solve(self.cfg['initial_tcp'],self.cfg['initial_rotation_xyzw'],
                        base_pos=station.BASES['left'],iters=500)
        if pe>.002 or re>.02:
            raise ValueError(f'Initial robot IK failed: {pe} m, {re} rad')
        return np.r_[q,0.0,READY,1.0]

    def reset(self,base):
        from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS,FINGER_OPEN
        action=self.initial_action()
        for side,start in [('left',0),('right',7)]:
            reset_joints(base,side+'_robot',dict(zip(ARM_JOINTS,action[start:start+6])))
            reset_joints(base,side+'_robot',dict.fromkeys(FINGER_JOINTS,float(action[start+6])*FINGER_OPEN))
        if self.cfg['kind']=='cable':
            # Prepare an open grasp, then let the native closing command capture the cord.
            reset_joints(base,'left_robot',dict.fromkeys(FINGER_JOINTS,.20*FINGER_OPEN))
        self._unsafe=False

    def pose_state(self,base,name,part=None):
        return pose_state(base.scene['cable_'+part] if name=='cable' else base.scene[name])

    def validate_reset(self,base):
        """Reject a collapsed prepared scene before attributing a policy outcome."""
        from yam_arena.runtime import tcp_world
        error=np.linalg.norm(tcp_world(base.scene['left_robot'])-self.cfg['initial_tcp'])
        if error>.015:
            raise RuntimeError(f'Left hand did not settle at the prepared start: {error} m')
        if self.cfg['kind']=='cable':
            checks=[('cable','device',self.cfg['device_position'])]
        else:
            checks=[(name,None,pos) for name,pos in self.poses.items() if name!='fixture']
        for name,part,expected in checks:
            p,_,_=self.pose_state(base,name,part)
            error=np.linalg.norm(p.cpu().numpy()-expected)
            if error>.015:
                raise RuntimeError(f'{part or name} moved during reset settling: {error} m')

    def evaluate(self,base):
        from yam_arena.runtime import tcp_world
        from yam_arena.urdf import quat_xyzw_to_matrix
        cfg=self.cfg; top=cfg['table_top']
        state={}
        for name,desc in self.descriptions.items():
            if name=='fixture': continue
            names=[part['name'] for part in desc['bodies']] if name=='cable' else [None]
            for part in names:
                p,q,v=self.pose_state(base,name,part)
                values=np.r_[p.cpu().numpy(),q.cpu().numpy(),v.cpu().numpy()]
                if not np.isfinite(values).all() or abs(np.linalg.norm(values[3:7])-1)>.001:
                    raise RuntimeError('Non-finite or invalid native state; measurement is unusable')
                state[part or name]=(values[:3],quat_xyzw_to_matrix(values[3:7]),values[7:])
        left=tcp_world(base.scene['left_robot']); right=tcp_world(base.scene['right_robot'])
        park_error=float(np.linalg.norm(left-cfg['park_tcp']))
        released=[]; violation=False; park_aligned=True
        if cfg['kind']=='tray':
            p,r,v=state['tray']; tilt=math.acos(float(np.clip(r[2,2],-1,1)))
            violation |= tilt>cfg['max_tray_tilt_rad']
            for name in ('payload0','payload1'):
                pos,_,_=state[name]; relative=r.T@(pos-p)
                violation |= bool(abs(relative[0])>.085 or abs(relative[1])>.15 or relative[2]<-.005)
            violation |= bool(p[2]<top-.01)
            supported=abs(p[2]-(cfg['mat_center'][2]+.003))<.006 and tilt<.10 and np.linalg.norm(v[:3])<.02
            on_mat=np.linalg.norm(p[:2]-cfg['mat_center'][:2])<.035
            released=[np.linalg.norm(left-p)>.22,np.linalg.norm(right-p)>.22]
            progress=float(supported and on_mat)
            endpoint=supported and on_mat and all(released)
        elif cfg['kind']=='books':
            _,hand_q,_=pose_state(base.scene['left_robot'],'gripper')
            park_aligned=quat_xyzw_to_matrix(hand_q.cpu().numpy())[2,2]<-.98
            placed=[]
            for name in ('book0','book1'):
                p,r,v=state[name]
                violation |= bool(p[2]<top-.003 or abs(p[0]-.35)>.60 or abs(p[1])>.64 or v[2]<-.35)
                placed.append(np.linalg.norm(p[:2]-cfg['mat_center'][:2])<.105
                    and abs(p[2]-(cfg['mat_center'][2]+.012))<.008 and r[2,2]>.98 and np.linalg.norm(v[:3])<.02)
                released.append(min(np.linalg.norm(left-p),np.linalg.norm(right-p))>.08)
            progress=sum(placed)/2; endpoint=all(placed) and all(released)
            tilt=0.0
        else:
            p,r,v=state['device']; tilt=math.acos(float(np.clip(r[2,2],-1,1)))
            displacement=float(np.linalg.norm(p-np.asarray(cfg['device_position'])))
            violation=displacement>.025 or tilt>.20
            # Check all native cable anchors before any credit; do not reward disconnection.
            for j in self.descriptions['cable']['joints']:
                pp,pr,_=state[j['parent']]; cp,cr,_=state[j['child']]
                gap=np.linalg.norm(pp+pr@j.get('position_parent',[0,0,0])-cp-cr@j.get('position_child',[0,0,0]))
                violation |= bool(gap>.003)
            progress=float(park_error<.06); endpoint=np.linalg.norm(v[:3])<.02
        self._unsafe |= bool(violation)
        success=bool(park_error<.025 and park_aligned and endpoint and not self._unsafe)
        score=0.0 if self._unsafe else (1.0 if success else .5*progress)
        return dict(success=success,score=score,unsafe_event=self._unsafe,park_error_m=park_error,
                    object_tilt_rad=tilt,recovery_progress=progress)
