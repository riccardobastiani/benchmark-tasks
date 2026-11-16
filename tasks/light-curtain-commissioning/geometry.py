"""Original guarded press trainer, physical indexing locks and geometric optics."""
from pathlib import Path
import numpy as np
from resist_isaac.recovery import body,box,description
from .labels import label,capsule
FOLDER=Path(__file__).parent
Z=.7475
FRAME=np.array([.350,0.,Z]);HEIGHT_START=.000;HEIGHT_GOOD=.050
HANDLE=np.array([-.100,.280,.190])
RX=np.array([0.,-.150,.082]);RX_HANDLE=np.array([-.100,0.,-.005])
BOLT=np.array([.280,-.195,Z+.130]);BOLT_GRIP=np.array([-.020,0.,0.])
PIN=np.array([-.060,-.150,.078])
ROD=np.array([.280,.320,Z+.040]);ROD_GRIP=np.array([-.100,0.,.007])
ROD_TIP=.110
RAM=np.array([.620,0.,Z+.215])
BUTTONS={n:np.array([.190,y,Z+.065]) for n,y in [('mode',-.250),('reset',-.350),('start',-.450)]}
BEAMS=.007+np.arange(16)*.010
OPENING=(.050,.210,.130)

def geometry():
    metal=(.36,.42,.47,1);yellow=(.95,.62,.035,1);dark=(.095,.13,.16,1);blue=(.14,.37,.54,1)
    fixtures=[box((.360,.570,.0001),[.410,0,Z+.0001],(.30,.36,.39,1),collision=False)]
    # Raised bed, side walls and top cover close the other access routes.
    fixtures += [box((.200,.180,.030),[.600,0,Z+.030],dark),
        box((.220,.008,.110),[.560,.178,Z+.110],(.3,.58,.67,.26)),
        box((.220,.008,.110),[.560,-.178,Z+.110],(.3,.58,.67,.26)),
        box((.095,.180,.020),[.465,0,Z+.230],(.35,.63,.70,.20)),
        box((.0675,.180,.020),[.7325,0,Z+.230],(.35,.63,.70,.20)),
        box((.070,.055,.020),[.630,.125,Z+.230],(.35,.63,.70,.20)),
        box((.070,.055,.020),[.630,-.125,Z+.230],(.35,.63,.70,.20)),
        box((.012,.180,.120),[.802,0,Z+.130],dark),
        box((.025,.025,.140),[.740,.145,Z+.130],metal),
        box((.025,.025,.140),[.740,-.145,Z+.130],metal),
        box((.050,.060,.007),[.620,0,Z+.067],(.57,.62,.65,1))]
    fixtures += [box((.020,.130,.025),[.350,0,Z+.025],dark),
        box((.020,.130,.020),[.350,0,Z+.230],(.35,.63,.70,.20))]
    for y in (-.154,.154):fixtures.append(box((.010,.024,.080),[.380,y,Z+.130],metal))
    # Frame guide supports sit outside the optical opening.
    for y in (-.230,.230):fixtures.append(box((.022,.008,.150),[.350,y,Z+.150],metal))
    fixtures += [box((.025,.025,.025),[.290,-.195,Z+.130],metal,collision=False),
        label('LIGHT CURTAIN TRAINER',[.080,.0,Z+.002],(.30,.035)),
        label('HEIGHT LOCK',[.175,-.195,Z+.002],(.135,.022)),
        label('TEST ROD 14 mm',[.110,.320,Z+.002],(.170,.025)),
        label('BEAMS VISUALIZED',[.420,.395,Z+.002],(.20,.022))]
    for n,p in BUTTONS.items():

        for side in (-1,1):
            fixtures.append(box((.008,.029,.025),p+[side*.022,0,-.025],dark))
            fixtures.append(box((.014,.0075,.025),p+[0,side*.0215,-.025],dark))
        fixtures.append(label({'mode':'TEST MODE','reset':'RESET','start':'CYCLE START'}[n],[.120,p[1],Z+.002],(.085,.034)))
    for x in (.245,.385):
        fixtures.append(box((.012,.016,.004),[x,.320,Z+.027],blue))
        for s in (-1,1):fixtures.append(box((.012,.003,.010),[x,.320+s*.013,Z+.037],blue))
    result={'fixture':description('Fixture',[body('bench',fixtures,static=True)])}
    frame=[box((.015,.015,.081),[0,.150,.082],yellow),
        box((.012,.295,.008),[-.060,0,.190],metal),
        box((.030,.012,.008),[-.030,.150,.174],metal),
        box((.030,.012,.008),[-.030,-.150,.174],metal),
        box((.032,.006,.009),HANDLE,(.15,.20,.24,1),friction=[1.1,.9])]
    # Two actual through-slots in a keeper plate, not software clamp flags.
    y=-.195
    for s in (-1,1):frame.append(box((.006,.0045,.090),[-.040,y+s*.0115,.100],metal))
    bounds=[(.010,.0755),(.0845,.1255),(.1345,.190)]
    for lo,hi in bounds:frame.append(box((.006,.007,(hi-lo)/2),[-.040,y,(lo+hi)/2],metal))
    frame.append(box((.018,.018,.008),[-.040,-.180,.174],metal))
    tx=label('TX',[-.0153,.150,.145],(.024,.018));tx['orientation']=[2**-.5,0,-2**-.5,0];frame.append(tx)
    visuals={}
    for i,h in enumerate(BEAMS):
        visuals['beam'+str(i)]=('mount','frame',len(frame))
        frame.append(capsule([0,-.132,float(h)],[0,.132,float(h)],.00045,(.035,.65,.80,.45),collision=False))
    receiver=[box((.015,.015,.081),color=yellow)]
    for s in (-1,1):receiver.append(box((.030,.009,.0025),[-.055,s*.011,-.030],metal))
    for lo,hi in [(-.085,-.062),(-.058,-.025)]:receiver.append(box(((hi-lo)/2,.002,.0025),[(lo+hi)/2,0,-.030],metal))
    receiver += [box((.004,.010,.015),[-.090,0,-.015],metal),box((.018,.006,.009),RX_HANDLE,blue,friction=[1.1,.9])]
    rx=label('RX',[-.0153,0,.060],(.024,.018));rx['orientation']=[2**-.5,0,-2**-.5,0];receiver.append(rx)
    visuals['alignment']=('mount','receiver',len(receiver));receiver.append(box((.004,.010,.006),[-.0154,0,.039],(.3,.02,.02,1),collision=False))
    pin=[capsule([0,0,-.035],[0,0,0],.0015,(.7,.7,.72,1)),box((.009,.009,.005),[0,0,.007],(.94,.32,.03,1))]
    bolt=[capsule([0,0,0],[.065,0,0],.003,(.65,.69,.71,1)),box((.012,.007,.012),BOLT_GRIP,(.12,.25,.36,1),friction=[1.2,1.])]
    parts=[body('frame',frame,.32,FRAME+[0,0,HEIGHT_START]),body('receiver',receiver,.08,FRAME+RX+[0,0,HEIGHT_START]),
        body('height_bolt',bolt,.035,BOLT),body('index_pin',pin,.012,FRAME+PIN+[0,0,HEIGHT_START+.014])]
    a=np.deg2rad(12.);parts[1]['orientation']=[float(np.cos(a/2)),0,0,float(np.sin(a/2))]
    joints=[dict(name='height',type='prismatic',child='frame',axis='Z',position_parent=FRAME.tolist(),lower=.0,upper=.120,stiffness=0.,damping=1.),
        dict(name='yaw',type='revolute',parent='frame',child='receiver',axis='Z',position_parent=RX.tolist(),lower=0.,upper=.30,stiffness=.005,damping=.003,target=a,collision=True),
        dict(name='height_lock',type='prismatic',child='height_bolt',axis='X',position_parent=BOLT.tolist(),lower=-.052,upper=0.,stiffness=25.,damping=1.,target=0.),
        dict(name='alignment_lock',type='prismatic',parent='frame',child='index_pin',axis='Z',position_parent=PIN.tolist(),lower=0.,upper=.014,stiffness=80.,damping=.3,target=0.)]
    result['mount']=description('Mount',parts,joints=joints,solver_position_iterations=128,solver_velocity_iterations=32)
    rod=[capsule([-.125,0,0],[ROD_TIP,0,0],.007,(.89,.67,.08,1),friction=[1.1,.9]),
        box((.025,.009,.010),[-.100,0,0],(.12,.17,.19,1),friction=[1.2,1.])]
    result['rod']=description('Probe',[body('tool',rod,.050,ROD)],solver_position_iterations=64,solver_velocity_iterations=16)
    ram=[box((.040,.050,.015),color=(.72,.16,.10,1)),box((.012,.014,.065),[0,0,.075],metal)]
    result['press']=description('Press',[body('ram',ram,.50,RAM)],joints=[dict(name='stroke',type='prismatic',child='ram',axis='Z',position_parent=RAM.tolist(),lower=-.120,upper=0.,stiffness=12000.,damping=110.,max_force=40.,target=0.)],solver_position_iterations=64,solver_velocity_iterations=16)
    for n,p in BUTTONS.items():
        colors={'mode':(.14,.35,.75,1),'reset':(.95,.69,.04,1),'start':(.1,.65,.22,1)}
        result[n]=description(n.title(),[body('button',[box((.012,.012,.008),color=colors[n])],.025,p)],joints=[dict(name='push',type='prismatic',child='button',axis='Z',position_parent=p.tolist(),lower=-.008,upper=0.,stiffness=160.,damping=1.,target=0.)])
    # Large front-board indicators are hardware outputs, not scorer answers.
    plate=[box((.028,.085,.008),[.470,-.320,Z+.008],dark)]
    for i,n in enumerate(('permit','blocked','reset_required')):
        yy=-.260-i*.06;visuals[n]=('display','panel',len(plate));plate.append(box((.012,.014,.002),[.470,yy,Z+.018],(.05,.05,.05,1),collision=False))
        plate.append(label({'permit':'RUN PERMIT','blocked':'FIELD BLOCKED','reset_required':'RESET REQUIRED'}[n],[.430,yy,Z+.020],(.057,.022)))
    result['display']=description('Display',[body('panel',plate,static=True)])
    return result,visuals
