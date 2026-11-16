"""Declared optical coverage and finite-response machine-control model."""
import numpy as np

def shaft_point_speed(previous,position,axis,axial_distance,dt):
    """Displacement of the same shaft material point over one native physics interval."""
    if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid native sample interval')
    old_position,old_axis=previous
    displacement=(position-old_position)+(axis-old_axis)*axial_distance
    return float(np.linalg.norm(displacement)/dt)

class Proof:
    max_speed=.040
    def __init__(self):
        self.lanes=np.zeros((3,12));self.edges=np.zeros((2,20));self.reference=None
    def reset(self):self.lanes[:]=0;self.edges[:]=0;self.reference=None
    def observe(self,dt,y,h,speed,eligible):
        if not eligible or speed>self.max_speed:return
        for i,centre in enumerate((-.120,0.,.120)):
            if abs(y-centre)<.012 and .061<=h<=.199:
                j=min(11,max(0,int((h-.062)/.136*12)));self.lanes[i,j]+=dt
        for i,centre in enumerate((.062,.198)):
            if abs(h-centre)<.010 and -.121<=y<=.121:
                j=min(19,max(0,int((y+.120)/.240*20)));self.edges[i,j]+=dt
    @property
    def fraction(self):return float((np.count_nonzero(self.lanes>=.060)+np.count_nonzero(self.edges>=.060))/76)
    @property
    def complete(self):return bool(np.all(self.lanes>=.060) and np.all(self.edges>=.060))

class Controller:
    """Sensing does not know the scorer's correct height or rod-test coverage."""
    def __init__(self):
        self.mode=False;self.state='SETUP';self.target=0.;self.velocity=0.;self.old={n:False for n in ('mode','reset','start')}
        self.button_time={n:0. for n in self.old}
        self.block_time=0.;self.inhibit=False;self.stop=False;self.reset_test=False;self.cycles=0
        self._inhibit_pending=None;self._reset_pending=None;self.trip_time=None;self.time=0.;self.stop_times=[];self.failure=''
    def step(self,dt,aligned,blocked,buttons,ram_q,ram_v,warmup=False):
        if not np.isfinite([dt,ram_q,ram_v]).all():raise ValueError('Invalid native machine sample')
        self.time+=dt
        self.button_time={n:self.button_time[n]+dt if buttons[n] else 0. for n in buttons}
        stable={n:self.button_time[n]>=.100-1e-8 for n in buttons}
        edges={n:stable[n] and not self.old[n] for n in buttons};self.old=stable
        if warmup:return
        self.block_time=self.block_time+dt if blocked or not aligned else 0.
        tripped=self.block_time>=.0125-1e-8
        if edges['mode']:
            self.mode=not self.mode;self.state='RESET_REQUIRED' if self.mode else 'SETUP'
        if not self.mode:self.target=0.;self.velocity=0.;return
        if tripped:
            if self.state in ('DOWN','UP'):
                self.state='STOPPING';self.trip_time=self.time
            elif self.state=='ARMED':self.state='RESET_REQUIRED'
        if edges['start'] and blocked and aligned:
            self._inhibit_pending=(self.time,ram_q)
        if self._inhibit_pending:
            t,q=self._inhibit_pending
            if abs(ram_q-q)>.003:self._inhibit_pending=None
            elif self.time-t>=.2:self.inhibit=True;self._inhibit_pending=None
        if self._reset_pending:
            t,q=self._reset_pending
            if abs(ram_q-q)>.003:self._reset_pending=None
            elif self.time-t>=.2:self.reset_test=True;self._reset_pending=None
        if edges['reset'] and aligned and not blocked and self.state=='RESET_REQUIRED':
            self.state='ARMED';self._reset_pending=(self.time,ram_q)
        if edges['start'] and aligned and not blocked and self.state=='ARMED':self.state='DOWN'
        if self.state=='DOWN':
            self.velocity=-.080;self.target=max(-.120,self.target+self.velocity*dt)
            if ram_q<-.116:self.state='UP'
        elif self.state=='UP':
            self.velocity=.080;self.target=min(0.,self.target+self.velocity*dt)
            if ram_q>-.003 and self.target==0.:self.cycles+=1;self.state='ARMED';self.velocity=0.
        elif self.state=='STOPPING':
            self.velocity=float(np.sign(self.velocity)*max(0.,abs(self.velocity)-1.6*dt))
            self.target=float(np.clip(self.target+self.velocity*dt,-.12,0.))
            if abs(ram_v)<.004 and self.velocity==0.:
                elapsed=self.time-self.trip_time;self.stop_times.append(elapsed);self.stop=True;self.state='RESET_REQUIRED'
            elif self.time-self.trip_time>.20:self.failure='stop_response_exceeded_200_ms'
        else:self.velocity=0.


def opening_covered(heights):
    # Independent dense sampling for a nominal 14 mm circular object that fits the opening.
    centres=np.arange(.057,.20301,.0005)
    return bool(np.all(np.min(np.abs(centres[:,None]-np.asarray(heights)[None,:]),axis=1)<=.007))
