"""Extract the retained CC0 car; separate original wheels for native revolute joints.

This is asset conversion, not physics or a rendered simulation. Source coordinates
are converted uniformly to a 107 mm car. The display cloth and cameras are omitted.
"""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

FOLDER=Path(__file__).parent
SCALE=.00028
WHEEL_RADIUS=26.28*SCALE


def main():
    folder=FOLDER/'assets'
    data=(folder/'ToyCar.glb').read_bytes()
    length=struct.unpack_from('<I',data,12)[0]
    gltf=json.loads(data[20:20+length]); binary=data[28+length:]
    def accessor(index):
        a=gltf['accessors'][index]; v=gltf['bufferViews'][a['bufferView']]
        width={'SCALAR':1,'VEC2':2,'VEC3':3}[a['type']]
        dtype={5126:'<f4',5125:'<u4'}[a['componentType']]
        if v.get('byteStride', width*np.dtype(dtype).itemsize)!=width*np.dtype(dtype).itemsize:
            raise ValueError('Unexpected interleaved source')
        return np.frombuffer(binary,dtype,count=a['count']*width,
            offset=v.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1,width)
    primitive=gltf['meshes'][0]['primitives'][0]
    vertices=accessor(primitive['attributes']['POSITION'])
    faces=accessor(primitive['indices']).reshape(-1,3)
    uv=accessor(primitive['attributes']['TEXCOORD_0'])
    unique,inverse=np.unique(np.round(vertices,3),axis=0,return_inverse=True)
    parents=list(range(len(unique)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]; i=parents[i]
        return i
    for face in inverse[faces]:
        a=root(int(face[0]))
        for b in face[1:]: parents[root(int(b))]=a
    roots=np.array([root(int(i)) for i in inverse])
    # Detached tire/hub components at the four existing wheel centers. Do not
    # cut the chassis or deform its wheel arches to suit approximate colliders.
    groups=np.full(len(vertices),-1)
    centers=[(133.57,-73.56), (133.57,73.86), (-106.,-72.66), (-106.,72.34)]
    for component in np.unique(roots):
        mask=roots==component; points=vertices[mask]
        lo,hi=points.min(0),points.max(0)
        for i,(x,y) in enumerate(centers):
            if lo[0]>y-21 and hi[0]<y+21 and lo[1]>x-28 and hi[1]<x+28 and lo[2]>-65 and hi[2]<-8:
                groups[mask]=i
    points=np.column_stack((vertices[:,1],vertices[:,0],-vertices[:,2]-36.21))*SCALE
    def write_obj(name,mask,positions,textured=True):
        selected=faces[mask]
        indices=np.unique(selected); mapping=np.full(len(vertices),-1); mapping[indices]=np.arange(len(indices))+1
        with (folder/name).open('w') as out:
            if textured:out.write('mtllib car.mtl\nusemtl car\n')
            for p in positions[indices]: out.write('v '+' '.join(f'{v:.9g}' for v in p)+'\n')
            for v in uv[indices]: out.write(f'vt {v[0]:.9g} {1-v[1]:.9g}\n')
            for face in mapping[selected]: out.write('f '+' '.join(f'{i}/{i}' for i in face)+'\n')
    face_groups=groups[faces]
    assert np.all(face_groups==face_groups[:,:1]),'Source component crosses body split'
    components,counts=np.unique(roots[groups==-1],return_counts=True)
    painted=components[np.argmax(counts)]
    paint_faces=roots[faces[:,0]]==painted
    write_obj('body.obj',(face_groups[:,0]==-1)&~paint_faces,points)
    write_obj('paint.obj',paint_faces,points,textured=False)
    wheel_centers=[]
    for i,(x,y) in enumerate(centers):
        center=np.array([x*SCALE,y*SCALE,0.])
        assert np.count_nonzero(face_groups[:,0]==i)>500,'Missing retained wheel'
        write_obj(f'wheel{i}.obj',face_groups[:,0]==i,points-center)
        wheel_centers.append(center.tolist())
    texture=gltf['materials'][0]['pbrMetallicRoughness']['baseColorTexture']['index']
    image=gltf['images'][gltf['textures'][texture]['source']]
    view=gltf['bufferViews'][image['bufferView']];start=view['byteOffset']
    (folder/'car.png').write_bytes(binary[start:start+view['byteLength']])
    (folder/'car.mtl').write_text('newmtl car\nKd 1 1 1\nmap_Kd car.png\n')
    meta=dict(source='https://github.com/KhronosGroup/glTF-Sample-Assets/tree/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/ToyCar',
        license='CC0-1.0',authors=['Guido Odendahl','Eric Chadwick'],
        source_sha256=hashlib.sha256(data).hexdigest(),
        scope='Retained appearance mesh, not a scan or calibrated commercial slot car. Authored physical chassis/guide/wheel collision approximations.',
        conversion='Uniform scale; source y,x,-z to local x,y,z; wheel-axis height origin; original disconnected wheel components separated. Main painted shell separated for red/yellow/blue USD materials without editing the texture. Cloth, glass overlay and source cameras omitted. Other base colors retained; advanced material extensions omitted.',
        source_unit_scale_m=SCALE,wheel_centers_m=wheel_centers,wheel_radius_m=WHEEL_RADIUS,
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.suffix in ('.obj','.mtl','.png')})
    (folder/'SOURCE.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__': main()
