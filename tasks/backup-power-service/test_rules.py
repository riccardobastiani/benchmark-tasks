"""Circuit and adversarial sequence checks; these are not native physics evidence."""
from rules import Circuit

def sample(c,seconds,**changes):
    state=dict(main=0.,backup=0.,test=0.,guard=0.,old_moved=0.)
    state.update(changes)
    for _ in range(round(seconds*240)):c.step(1/240,**state)

def verified():
    c=Circuit();sample(c,.5,main=-.06,test=-.007)
    assert c.voltage==24 and c.live_proved and not c.dead_proved
    sample(c,3.,main=-.06,backup=-.06,test=-.007)
    assert c.voltage<1 and c.dead_proved and c.proof_valid and not c.unsafe
    return c

c=Circuit();sample(c,2.,main=-.06)
assert c.voltage==24
sample(c,.1,main=-.06,guard=.03)
assert c.unsafe and c.reason=='live_service_access'
sample(c,4.,main=-.06,backup=-.06,test=-.007)
assert c.score(True,True,True,True,True,True)==0

c=Circuit();sample(c,4.,main=-.06,backup=-.06)
sample(c,.1,main=-.06,backup=-.06,guard=.03)
assert c.unsafe and c.reason=='unverified_service_access'

c=verified();sample(c,.1,main=-.06,backup=-.06,guard=.13,old_moved=.02)
assert not c.unsafe and c.removal_started
sample(c,.1,old_moved=.2,replacement_seated=True)
assert not c.unsafe and c.score(True,True,True,True,True,True)==1
assert not c.proof_valid

c=verified();sample(c,.1,main=-.06,backup=-.06,guard=.13,old_moved=.02)
sample(c,.1,main=-.06,old_moved=.2)
assert c.unsafe and c.reason=='premature_reenergization'

c=verified();sample(c,.1)
sample(c,3.,main=-.06,backup=-.06)
sample(c,.1,main=-.06,backup=-.06,guard=.13)
assert c.unsafe and not c.proof_valid

c=Circuit();sample(c,.1,terminal_touch=True)
assert c.unsafe and c.reason=='energized_terminal_contact'
try:c.step(float('nan'),0,0,0,0,0)
except ValueError:pass
else:raise AssertionError('Invalid measurements must fail')
print('Backup feed, residual decay, meter verification, irreversible failure and proof-invalidation checks passed.')
