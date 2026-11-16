"""Nominal primitive mass properties and USD scene descriptors."""
import math
import numpy as np

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
