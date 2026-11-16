"""Original post-procedure surgical tray props; no human actor."""
from io import BytesIO
from pathlib import Path
import numpy as np
from resist_isaac.recovery import body,box,description

FOLDER=Path(__file__).parent
TABLE_Z=.7475
TRAY_START=np.array([.290,0.,TABLE_Z+.003])
DOCK=np.array([.470,0.,TABLE_Z+.003])
BIN=np.array([.230,.355,TABLE_Z])
BIN_INNER=np.array([.096,.071,.144])
HANDLE={'left':np.array([0.,.184,.045]),'right':np.array([0.,-.184,.045])}
SHARPS=('sharp_a','sharp_b')
TOOLS=('spatula','holder')
TIP=np.array([.0652,0.,0.])
TIP_RADIUS=.00045


def label(text,position,size=(.13,.026),color=(.96,.97,.98),subtitle=None):
    from PIL import Image,ImageDraw,ImageFont
    w,h=size;width,height=round(w*12000),round(h*12000)
    image=Image.new('RGB',(width,height),tuple(round(c*255) for c in color));draw=ImageDraw.Draw(image)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    font=ImageFont.truetype(font_path,height)
    l,t,r,b=draw.textbbox((0,0),text,font=font)
    font=ImageFont.truetype(font_path,max(1,int(height*min(.90*width/(r-l),(.45 if subtitle else .74)*height/(b-t)))))
    l,t,r,b=draw.textbbox((0,0),text,font=font)
    centre=.34*height if subtitle else .5*height
    draw.text(((width-r-l)/2,centre-(b+t)/2),text,font=font,fill=(10,16,22))
    if subtitle:
        small=ImageFont.truetype(font_path,round(.21*height))
        l,t,r,b=draw.textbbox((0,0),subtitle,font=small)
        draw.text(((width-r-l)/2,.79*height-(b+t)/2),subtitle,font=small,fill=(10,16,22))
    draw.rectangle((1,1,width-2,height-2),outline=(55,65,75),width=2)
    folder=FOLDER/'assets';folder.mkdir(exist_ok=True)
    name=''.join(c if c.isalnum() else '_' for c in text)+'.png';path=folder/name
    data=BytesIO();image.save(data,format='PNG')
    if not path.exists() or path.read_bytes()!=data.getvalue():path.write_bytes(data.getvalue())
    return dict(type='mesh',position=list(position),collision=False,roughness=1.,texture='assets/'+name,
        mesh=dict(points=[[-h/2,-w/2,0],[-h/2,w/2,0],[h/2,w/2,0],[h/2,-w/2,0]],
                  faces=[[0,3,2,1]],uv=[[1,0],[1,1],[0,1],[0,0]]))


def capsule(a,b,r,color,**kw):
    return dict(type='capsule',size=[r,0.],fromto=[*a,*b],color=list(color),**kw)


def geometry():
    metal=(.54,.59,.63,1);rubber=(.13,.17,.19,1)
    tray=[box((.130,.150,.003),color=metal,friction=[.65,.5])]
    for sign in (-1,1):
        tray.append(box((.003,.150,.010),(sign*.130,0.,.013),metal))
        tray.append(box((.130,.003,.010),(0.,sign*.150,.013),metal))
        for x in (-.031,.031):tray.append(box((.004,.019,.022),(x,sign*.167,.024),metal))
        tray.append(box((.032,.006,.008),(0.,sign*.184,.045),rubber,friction=[.9,.75]))
    for shape in tray:
        if shape['color']==list(metal):shape.update(metallic=.65,roughness=.35)
    tray.append(label('USED SURGICAL INSTRUMENTS',[.108,0,.0032],(.235,.022)))
    result={'tray':description('Tray',[body('pan',tray,.32,TRAY_START)],
        solver_position_iterations=64,solver_velocity_iterations=16)}
    # A stable, open point-of-use container: no hidden disposal inventory.
    red=(.72,.055,.045,1);bin_shapes=[box((.100,.075,.004),(BIN[0],BIN[1],TABLE_Z+.004),red)]
    for sign in (-1,1):
        bin_shapes.append(box((.004,.075,.070),(BIN[0]+sign*.100,BIN[1],TABLE_Z+.074),red))
        bin_shapes.append(box((.100,.004,.070),(BIN[0],BIN[1]+sign*.075,TABLE_Z+.074),red))
    # Labels sit on flat, physical rim shelves beside the opening.
    bin_shapes.append(box((.022,.075,.004),(BIN[0]-.125,BIN[1],TABLE_Z+.135),red))
    bin_shapes.append(label('SHARPS ONLY',[BIN[0]-.125,BIN[1],TABLE_Z+.1392],(.148,.035),(1.,.90,.74)))
    # Visual draping preserves the calibrated table's native support surface.
    fixture=[*bin_shapes,
        box((.550,.650,.00001),[.360,0,TABLE_Z+.00001],(.12,.32,.43,1),collision=False,roughness=1.),
        box((.550,.0005,.040),[.360,.650,TABLE_Z-.040],(.12,.32,.43,1),collision=False,roughness=1.),
        box((.550,.0005,.040),[.360,-.650,TABLE_Z-.040],(.12,.32,.43,1),collision=False,roughness=1.),
        box((.0005,.650,.040),[.910,0,TABLE_Z-.040],(.12,.32,.43,1),collision=False,roughness=1.),
        box((.140,.208,.0001),[DOCK[0],0,TABLE_Z+.0001],(.58,.67,.70,1),collision=False),
        label('SOILED INSTRUMENTS',[.645,0,TABLE_Z+.002],(.34,.058),(.83,.91,.93),subtitle='COLLECTION AREA'),
        box((.015,1.30,.800),[1.150,0,.850],(.58,.72,.74,1),collision=False,roughness=1.)]
    result['fixture']=description('Station',[body('bench',fixture,static=True)])
    for name,offset in [('sharp_a',[-.0492,.015,.016]),('sharp_b',[.005,-.088,.012])]:
        syringe=[
            dict(type='cylinder',size=[.008,0.],fromto=[-.035,0,0,.035,0,0],color=[.86,.88,.86,1]),
            box((.002,.016,.009),(-.037,0,0),(.83,.86,.84,1)),
            capsule([-.062,0,0],[-.037,0,0],.003,(.17,.22,.25,1)),
            box((.003,.012,.007),(-.063,0,0),(.15,.21,.24,1)),
            capsule([.035,0,0],[.041,0,0],.004,(.92,.43,.06,1)),
            capsule([.041,0,0],[.065,0,0],TIP_RADIUS,(.63,.68,.73,1),contact_offset=.0001),
            capsule([.0650,0,0],TIP,TIP_RADIUS,(.65,.70,.75,1),contact_offset=.0001)]
        for x in np.linspace(-.023,.025,9):
            syringe.append(box((.00045,.005,.0002),(x,0,.0081),(.08,.10,.12,1),collision=False))
        # Mark the barrel, while keeping the needle at its nominal 0.9 mm diameter.
        syringe.append(box((.008,.003,.0002),(0,0,.0082),(.96,.61,.05,1),collision=False))
        result[name]=description('Syringe',[body('device',syringe,.018,TRAY_START+offset)],
            solver_position_iterations=64,solver_velocity_iterations=16)
    spatula=[capsule([0,-.025,0],[0,.025,0],.006,(.16,.36,.64,1)),
        box((.014,.018,.002),(0,-.045,-.004),metal,metallic=.65,roughness=.35)]
    result['spatula']=description('Spatula',[body('tool',spatula,.035,TRAY_START+[0.,.030,.009])])
    holder=[box((.012,.038,.005),(0,0,0),(.58,.63,.66,1)),
        box((.022,.005,.005),(0,.043,0),(.58,.63,.66,1)),
        box((.022,.005,.005),(0,-.043,0),(.58,.63,.66,1))]
    for shape in holder:shape.update(metallic=.65,roughness=.35)
    result['holder']=description('Holder',[body('tool',holder,.030,TRAY_START+[-.100,.085,.008])])
    return result
