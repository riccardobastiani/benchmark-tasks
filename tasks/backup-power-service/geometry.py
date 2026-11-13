"""Full-size 24 V UPS service trainer; passive hardware and procedural labels."""
from pathlib import Path
import numpy as np
from resist_isaac.recovery import body,box,description

FOLDER=Path(__file__).parent
BOARD_Z=.7555
SEAT=np.array([.365,0.,BOARD_Z+.027])
TRAYS={'faulty':np.array([.400,-.225,BOARD_Z+.005+.015]),
       'spare':np.array([.400,.225,BOARD_Z+.005+.015])}
ISOLATORS={'main':np.array([.210,.225,BOARD_Z+.034]),
           'backup':np.array([.210,-.225,BOARD_Z+.034])}
BUTTON=np.array([.185,0.,BOARD_Z+.029])
GUARD=np.array([.370,0.,BOARD_Z+.092])
GRIP_OFFSET=np.array([0.,0.,.035])
ISO_OFFSET=np.array([-.025,0.,.025])
GUARD_OFFSET=np.array([-.062,0.,.020])

def label(text,position,size=(.11,.025),color=(.9,.93,.94)):
    from io import BytesIO
    from PIL import Image,ImageDraw,ImageFont
    folder=FOLDER/'assets';folder.mkdir(exist_ok=True)
    name=''.join(c if c.isalnum() else '_' for c in text)+'.png';path=folder/name
    w,h=size
    # Match raster and physical aspect ratios; fit the actual glyph bounds.
    width,height=round(w*12000),round(h*12000)
    image=Image.new('RGB',(width,height),tuple(round(c*255) for c in color))
    draw=ImageDraw.Draw(image)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    font=ImageFont.truetype(font_path,height)
    left,top,right,bottom=draw.textbbox((0,0),text,font=font)
    font=ImageFont.truetype(font_path,max(1,int(height*min(.90*width/(right-left),.74*height/(bottom-top)))))
    left,top,right,bottom=draw.textbbox((0,0),text,font=font)
    draw.text(((width-right-left)/2,(height-bottom-top)/2),text,font=font,fill=(10,15,20))
    draw.rectangle((1,1,width-2,height-2),outline=(45,55,65),width=2)
    data=BytesIO();image.save(data,format='PNG')
    if not path.exists() or path.read_bytes()!=data.getvalue():path.write_bytes(data.getvalue())
    # Face-varying UV order follows faces, not points. Text left is +Y,
    # text top is +X, so the station's front camera sees upright lettering.
    return dict(type='mesh',position=list(position),collision=False,texture='assets/'+name,roughness=1.,
        mesh=dict(points=[[-h/2,-w/2,0],[-h/2,w/2,0],[h/2,w/2,0],[h/2,-w/2,0]],
                  faces=[[0,3,2,1]],uv=[[1,0],[1,1],[0,1],[0,0]]))

def geometry():
    shapes=[box((.225,.310,.004),(.320,0.,BOARD_Z-.004),(.15,.19,.23,1))]
    visuals={}
    def visual(name,shape):visuals[name]=len(shapes);shapes.append(shape)
    shapes.append(label('24 V - DUAL SUPPLY',(.114,0,BOARD_Z+.001),(.30,.027)))
    for y,text,color in [(.225,'MAIN',(.18,.32,.48,1)),(-.225,'BACKUP',(.25,.25,.27,1))]:
        shapes.append(box((.030,.037,.025),(.300,y,BOARD_Z+.025),color))
        shapes.append(label(text,(.300,y,BOARD_Z+.0502),(.072,.052),(.97,.98,1.)))
        shapes.append(box((.007,.005,.003),(.263,y,BOARD_Z+.004),(.25,.30,.35,1)))
    shapes.append(label('SPARE',(.496,.225,BOARD_Z+.002),(.136,.040),(.66,.95,.74)))
    shapes.append(label('REMOVED',(.496,-.225,BOARD_Z+.002),(.136,.040),(1.,.73,.66)))
    shapes.append(label('HOLD TEST',(.150,0,BOARD_Z+.002),(.145,.030),(.97,.98,1.)))
    shapes.append(label('BUS VOLTS',(.282,0,BOARD_Z+.002),(.145,.023),(.97,.98,1.)))
    shapes.append(label('SERVICE',(.310,-.136,BOARD_Z+.002),(.08,.032),(.97,.98,1.)))
    for name,p in TRAYS.items():
        color=(.55,.13,.09,1) if name=='faulty' else (.10,.38,.18,1)
        shapes.append(box((.062,.050,.0025),(p[0],p[1],BOARD_Z+.0025),color))
        for sign in (-1,1):
            shapes.append(box((.062,.003,.006),(p[0],p[1]+sign*.053,BOARD_Z+.006),color))
            shapes.append(box((.003,.050,.006),(p[0]+sign*.065,p[1],BOARD_Z+.006),color))
    # Insulated service channel, guided only by real collision geometry.
    shapes.append(box((.075,.044,.006),(.370,0,BOARD_Z+.006),(.24,.27,.29,1)))
    for sign in (-1,1):
        shapes.append(box((.075,.003,.016),(.370,sign*.038,BOARD_Z+.022),(.4,.43,.45,1)))
        shapes.append(box((.165,.004,.004),(.440,sign*.091,GUARD[2]),(.5,.52,.54,1)))
    shapes.append(box((.005,.036,.021),(.422,0,BOARD_Z+.027),(.12,.14,.15,1)))
    terminals=[]
    for y in (-.020,.020):
        terminals.append(len(shapes));shapes.append(box((.006,.004,.005),(.414,y,BOARD_Z+.028),(.78,.44,.12,1)))
    visual('main_led',dict(type='sphere',size=[.007],position=[.278,.225,BOARD_Z+.009],color=[.04,.1,.04,1],collision=False))
    visual('backup_led',dict(type='sphere',size=[.007],position=[.278,-.225,BOARD_Z+.009],color=[.1,.06,.01,1],collision=False))
    visual('run_led',dict(type='sphere',size=[.009],position=[.405,.125,BOARD_Z+.012],color=[.08,.02,.02,1],collision=False))
    shapes.append(label('RUN',(.447,.142,BOARD_Z+.002),(.045,.040),(.97,.98,1.)))
    # Three seven-segment digits: 24.0, or a measured decaying bus voltage.
    shapes.append(box((.021,.042,.003),(.245,0,BOARD_Z+.003),(.015,.020,.024,1)))
    for digit,y in enumerate((.025,0,-.025)):
        segments=[(0,.007,.0015,.0055),(0,-.007,.0015,.0055),(.014,0,.0015,.0055)]
        # Local x is vertical on the display; y is horizontal.
        for segment,(dx,dy,hx,hy) in enumerate(((.014,0,.0014,.007),(.007,-.008,.006,.0014),
                (-.007,-.008,.006,.0014),(-.014,0,.0014,.007),(-.007,.008,.006,.0014),
                (.007,.008,.006,.0014),(0,0,.0014,.007))):
            visual(f'd{digit}_{segment}',box((hx,hy,.0005),(.245+dx,y+dy,BOARD_Z+.0065),(.025,.05,.03,1),collision=False))
    visual('decimal',dict(type='sphere',size=[.0015],position=[.231,-.012,BOARD_Z+.007],color=[.1,.8,.3,1],collision=False))
    result={'fixture':description('Panel',[body('plate',shapes,static=True)])}
    for name,p in ISOLATORS.items():
        housing=[box((.080,.025,.004),(-.025,0,-.012),(.16,.18,.2,1))]
        for sign in (-1,1):housing.append(box((.050,.004,.014),(0,sign*.026,0),(.3,.32,.34,1)))
        slider=[box((.025,.018,.007),(0,0,0),(.08,.32,.65,1) if name=='main' else (.85,.43,.07,1)),
            box((.007,.018,.010),ISO_OFFSET,(.12,.14,.16,1)),
            box((.015,.005,.011),(-.010,0,.012),(.12,.14,.16,1))]
        result[name]=description(name.title(),[body('housing',housing,static=True,position=p),
            body('slider',slider,mass=.10,position=p)],joints=[dict(name='disconnect',type='prismatic',child='slider',axis='X',
                position_parent=p.tolist(),lower=-.070,upper=0.,stiffness=0.,damping=2.,collision=False)])
    result['tester']=description('Tester',[body('housing',[box((.020,.026,.004),BUTTON-[0,0,.017],(.12,.13,.14,1))],static=True),
        body('button',[dict(type='cylinder',size=[.014,.005],color=[.85,.87,.9,1])],mass=.025,position=BUTTON)],
        joints=[dict(name='press',type='prismatic',child='button',axis='Z',position_parent=BUTTON.tolist(),
            lower=-.008,upper=0.,stiffness=300.,damping=2.,target=0.,collision=False)])
    result['guard']=description('Guard',[body('cover',[box((.090,.080,.002),color=(.45,.65,.73,.35)),
            box((.006,.025,.009),GUARD_OFFSET,(.26,.28,.3,1)),
            box((.062,.004,.009),(-.03,.026,.012),(.26,.28,.3,1)),
            box((.062,.004,.009),(-.03,-.026,.012),(.26,.28,.3,1))],mass=.20,position=GUARD)],
        joints=[dict(name='slide',type='prismatic',child='cover',axis='X',position_parent=GUARD.tolist(),
            lower=0.,upper=.145,stiffness=0.,damping=2.)])
    for name,p,color in [('faulty',SEAT,(.55,.16,.12,1)),('spare',TRAYS['spare'],(.12,.5,.25,1))]:
        unit=[box((.040,.032,.015),color=color),box((.006,.006,.012),(0,0,.020),(.18,.18,.18,1)),
            box((.012,.018,.010),GRIP_OFFSET,(.18,.18,.18,1))]
        if name=='faulty':
            for sign in (-1,1):unit.append(box((.002,.023,.0006),(0,0,.0158),(.98,.86,.6,1),orientation=[.9238795,0,0,sign*.3826834],collision=False))
        result[name]=description(name.title(),[body('unit',unit,mass=.16,position=p)])
    return result,visuals,terminals
