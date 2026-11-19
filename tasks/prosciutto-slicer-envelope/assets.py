"""Shared native asset authoring reused from the preserved laboratory task."""

import math
from pathlib import Path
import numpy as np
FOLDER=Path(__file__).parent

def cylinder(r,h,p=(0,0,0),color=(.65,.68,.69,1),**kw):
    return dict(type='cylinder',size=[r,h],position=list(p),color=list(color),**kw)

def rod(a,b,r=.004,color=(.30,.35,.38,1),**kw):
    return dict(type='capsule',size=[r],fromto=[*a,*b],color=list(color),**kw)

def lathe(profile, color=(.85,.94,.96,.16), glass=False):
    n=64
    points=[[r*math.cos(i*2*math.pi/n),r*math.sin(i*2*math.pi/n),z]
            for z,r in profile for i in range(n)]
    faces=[[j*n+i,j*n+(i+1)%n,(j+1)*n+(i+1)%n,(j+1)*n+i]
           for j in range(len(profile)-1) for i in range(n)]
    return dict(type='mesh',mesh=dict(points=points,faces=faces),collision=False,
                color=list(color),glass=glass,roughness=.045)

def walls(profile,thickness=.0015,n=24):
    shapes=[]
    for (z0,r0),(z1,r1) in zip(profile[:-1],profile[1:]):
        for i in range(n):
            a,b=2*math.pi*np.array([i,i+1])/n
            pts=[[r*math.cos(t),r*math.sin(t),z] for z,r in
                 [(z0,r0),(z0,r0+thickness),(z1,r1),(z1,r1+thickness)] for t in (a,b)]
            shapes.append(dict(type='mesh',visible=False,mesh=dict(points=pts,
                faces=[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]])))
    return shapes

def ring(r,z,thick=.001,color=(.72,.83,.85,1),position=(0,0,0)):
    result=[]
    for i in range(36):
        a,b=np.array([i,i+1])*2*math.pi/36
        result.append(rod(np.asarray(position)+[r*math.cos(a),r*math.sin(a),z],
                          np.asarray(position)+[r*math.cos(b),r*math.sin(b),z],thick,color))
    return result

def label(text,position,size,color=(.94,.96,.95)):
    from PIL import Image,ImageDraw,ImageFont
    path=FOLDER/'assets'/(''.join(c if c.isalnum() else '_' for c in text)+'.png')
    path.parent.mkdir(exist_ok=True)
    w,h=(round(x*10000) for x in size)
    image=Image.new('RGB',(w,h),tuple(round(x*255) for x in color));draw=ImageDraw.Draw(image)
    fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    font=ImageFont.truetype(fontfile,h);l,t,r,b=draw.textbbox((0,0),text,font=font)
    font=ImageFont.truetype(fontfile,max(1,int(h*min(.92*w/(r-l),.72*h/(b-t)))))
    l,t,r,b=draw.textbbox((0,0),text,font=font);draw.text(((w-r-l)/2,(h-b-t)/2),text,font=font,fill=(16,29,36))
    if not path.exists():image.save(path)
    width,height=size
    return dict(type='mesh',position=list(position),collision=False,texture='assets/'+path.name,
        mesh=dict(points=[[-height/2,-width/2,0],[-height/2,width/2,0],
                          [height/2,width/2,0],[height/2,-width/2,0]],faces=[[0,3,2,1]],
                  uv=[[1,0],[1,1],[0,1],[0,0]]))

def glass_material(stage,path):
    from pxr import Gf,Sdf,UsdShade
    mat=UsdShade.Material.Define(stage,path)
    shader=UsdShade.Shader.Define(stage,path+'/Shader')
    shader.CreateImplementationSourceAttr('sourceAsset')
    shader.SetSourceAsset(Sdf.AssetPath('OmniGlass.mdl'),'mdl')
    shader.SetSourceAssetSubIdentifier('OmniGlass','mdl')
    shader.CreateOutput('out',Sdf.ValueTypeNames.Token)
    for name,typ,value in [('glass_color',Sdf.ValueTypeNames.Color3f,Gf.Vec3f(.98,.99,1.)),
                           ('glass_ior',Sdf.ValueTypeNames.Float,1.47),
                           ('thin_walled',Sdf.ValueTypeNames.Bool,True),
                           ('frosting_roughness',Sdf.ValueTypeNames.Float,.06),
                           ('depth',Sdf.ValueTypeNames.Float,.01)]:
        shader.CreateInput(name,typ).Set(value)
    mat.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
    return mat
