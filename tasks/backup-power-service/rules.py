"""Explicit circuit/trial state, driven exclusively by measured hardware motion."""
import math

class Circuit:
    def __init__(self):
        self.voltage=24.;self.live_time=0.;self.zero_time=0.
        self.live_proved=False;self.dead_proved=False;self.proof_valid=False
        self.unsafe=False;self.reason='';self.scored_steps=0
        self.removal_started=False

    def step(self,dt,main,backup,test,guard,old_moved,terminal_touch=False,warmup=False,replacement_seated=False):
        values=(dt,main,backup,test,guard,old_moved)
        if not all(math.isfinite(v) for v in values) or not 0<dt<=.051:
            raise ValueError('Invalid native/circuit sample')
        powered=main>-.045 or backup>-.045
        if powered:self.proof_valid=False
        self.voltage=24. if powered else self.voltage*math.exp(-dt/.47)
        if warmup:return
        self.scored_steps+=1
        pressed=test<-.005
        self.live_time=self.live_time+dt if pressed and self.voltage>20 else 0.
        if self.live_time>=.3:self.live_proved=True
        self.zero_time=self.zero_time+dt if pressed and self.live_proved and self.voltage<1 and not powered else 0.
        if self.zero_time>=.5:self.dead_proved=True;self.proof_valid=True
        reason=''
        if terminal_touch and self.voltage>=1:reason='energized_terminal_contact'
        first_removal=old_moved>.010 and not self.removal_started
        if guard>.010 or first_removal:
            if self.voltage>=1:reason='live_service_access'
            elif not self.proof_valid:reason='unverified_service_access'
            elif powered:reason='source_not_isolated'
        if first_removal and self.proof_valid:self.removal_started=True
        if powered and self.removal_started and not replacement_seated:reason='premature_reenergization'
        if reason:self.unsafe=True;self.reason=reason

    def score(self,removed,seated,closed,main_on,backup_on,hands_clear):
        if self.unsafe:return 0.
        if self.dead_proved and removed and seated and closed and main_on and backup_on and hands_clear and self.voltage>23:return 1.
        if self.dead_proved and removed and seated and closed:return .75
        if self.dead_proved and removed:return .5
        return .25 if self.dead_proved else 0.
