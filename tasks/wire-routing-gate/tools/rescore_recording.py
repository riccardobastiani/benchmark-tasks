"""Rescore the two submitted native recordings without new inference or physics.

Legacy traces omit all-link robot/wire contact, so full release is established
only by a recorded legacy success held for the final 40 controls. Refuse to
infer an unrecorded release; unrouted outcomes do not need that witness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from geometry import geometry,UPSTREAM,TARGET,GATE_X,GATE_LOW,GATE_HIGH,PLATE_Z,WIRE_RADIUS,GATE_WIRE_HEIGHT
from rules import GripperParticipation,clip_state,crossings,gate_route,score
from yam_arena.urdf import quat_xyzw_to_matrix


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def rescore(folder):
    receipt=json.loads((folder/'receipt.json').read_text())
    assert receipt['status']=='completed' and receipt['source_unchanged']
    assert receipt['instruction']==json.loads((ROOT/'pilot.json').read_text())['instruction']
    for rel in ('geometry.py','interaction.py','resist_isaac/runtime.py','resist_isaac/usd_assets.py'):
        assert digest(ROOT/rel)==receipt['source_files_sha256'][rel],rel
    for name,sha in receipt['evidence_sha256'].items():
        assert digest(folder/name)==sha,name
    dt=receipt['physics_dt_s'];assert abs(dt-1/240)<1e-10
    use=GripperParticipation();substeps=0;unsafe=False
    with (folder/'native-wire-events.jsonl').open() as stream:
        for line in stream:
            event=json.loads(line)
            if event['warmup']:continue
            use.record(event['contacts'],dt);unsafe|=event['unsafe'];substeps+=1
    assert substeps==receipt['steps']*12
    with (folder/'trace.jsonl').open() as stream:trace=[json.loads(line) for line in stream]
    assert len(trace)==receipt['steps'] and all(t['step']==i+1 for i,t in enumerate(trace))
    last=trace[-1];state=[]
    for part in geometry()['wire']['bodies']:
        p,q,v=map(np.asarray,last['objects']['wire/'+part['name']])
        assert np.isfinite(np.r_[p,q,v]).all() and abs(np.dot(q,q)-1)<.001
        half=part['shapes'][0]['size'][1];tangent=quat_xyzw_to_matrix(q)[:,2]
        state.append((p,v,p-half*tangent,p+half*tangent))
    upstream=clip_state(state,UPSTREAM)[0];seated=clip_state(state,TARGET)[0]
    aligned=any(abs(p[0]-TARGET[0])<.008 and TARGET[2]+.004<p[2]<TARGET[2]+.05
                for _,p in crossings(state,1,TARGET[1]))
    routed,_=gate_route(state,UPSTREAM,TARGET,GATE_X,GATE_LOW,GATE_HIGH,PLATE_Z,WIRE_RADIUS,GATE_WIRE_HEIGHT)
    routed=bool(routed and aligned)
    # A legacy held success proves unchanged seating, no robot/wire contact,
    # TCP clearance, speed and 2-second hold, independently of participation.
    released=all(t['metrics']['success'] and t['metrics']['score']==1 for t in trace[-40:]) and len(trace)>=40
    if not unsafe and routed and upstream and seated and use.complete and not released:
        raise ValueError('Legacy trace cannot establish full release; record native validation instead')
    value=score(unsafe,routed,upstream and seated,released,use.complete)
    return dict(scope='Offline rescoring of recorded native evidence; no new physics or inference',
        recording=folder.name,steps=receipt['steps'],scored_substeps=substeps,
        original_score=receipt['final_metrics']['score'],revised_score=value,
        contact_time_s=use.contact_s,both_arms_used=use.complete,unsafe_event=bool(unsafe),
        routed=bool(routed),both_clips_seated=bool(upstream and seated),
        release_witness='Legacy success held for final 40 controls' if released else 'Not required for this outcome',
        original_receipt_sha256=digest(folder/'receipt.json'),
        native_events_sha256=digest(folder/'native-wire-events.jsonl'),
        trace_sha256=digest(folder/'trace.jsonl'),
        top_video_sha256=digest(folder/'top_cam.mp4'),
        current_scorer_sha256={rel:digest(ROOT/rel) for rel in ('rules.py','task.py')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    args=parser.parse_args()
    print(json.dumps(rescore(args.recording),indent=2))
