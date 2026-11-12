"""Geometry-only measurements and scalar outcomes; native state supplies inputs."""
import numpy as np


def crossings(state, axis, value):
    result = []
    segments = [(index,a,b) for index,(_,_,a,b) in enumerate(state)]
    # Capsule end hemispheres cover the small native anchor gaps between links.
    # Include those seams so crossing a joint cannot look like wire removal.
    segments.extend((index,a[3],b[2]) for index,(a,b) in enumerate(zip(state,state[1:])))
    for index, a, b in segments:
        if (a[axis]-value)*(b[axis]-value)>0 or abs(b[axis]-a[axis])<1e-9: continue
        t = (value-a[axis])/(b[axis]-a[axis])
        result.append((index, a+t*(b-a)))
    return result


def clip_state(state, centre):
    hits = [p for _, p in crossings(state, 1, centre[1])]
    seated = any(abs(p[0]-centre[0])<.002 and centre[2]+.004<p[2]<centre[2]+.012 for p in hits)
    retained = any(abs(p[0]-centre[0])<.012 and centre[2]-.001<p[2]<centre[2]+.0225 for p in hits)
    return seated, retained


def ordered_path(state):
    """Preserve material order and join capsule axes across their covered seams."""
    points = [state[0][2]]
    for i, (_, _, a, b) in enumerate(state):
        points.append((a+b)/2)
        points.append((b+state[i+1][2])/2 if i+1<len(state) else b)
    return points


def path_hits(points, axis, value):
    hits = []
    for i, (a,b) in enumerate(zip(points,points[1:])):
        delta=b[axis]-a[axis]
        if abs(delta)<1e-10:
            if abs(a[axis]-value)<1e-10:
                hits.extend([(float(i),a),(float(i+1),b)])
            continue
        t=(value-a[axis])/delta
        if 0<=t<=1: hits.append((i+float(t),a+t*(b-a)))
    unique=[]
    for s,p in sorted(hits,key=lambda item:item[0]):
        if not unique or abs(s-unique[-1][0])>1e-8: unique.append((s,p))
    return unique


def gate_route(state, upstream, target, gate_x, low, high, board_z, radius, height):
    """Only the ordered span connecting the two clips can earn routing credit."""
    points=ordered_path(state)
    source=[(s,p) for s,p in path_hits(points,1,upstream[1])
            if abs(p[0]-upstream[0])<.002 and upstream[2]+.004<p[2]<upstream[2]+.012]
    if not source: return False,[]
    source_s,source_p=source[0]
    seated=clip_state(state,target)[0]
    x_limit,z_limit=(.002,.012) if seated else (.008,.050)
    destination=[(s,p) for s,p in path_hits(points,1,target[1])
                 if s>source_s and abs(p[0]-target[0])<x_limit
                 and target[2]+.004<p[2]<target[2]+z_limit]
    if not destination: return False,[]
    target_s,target_p=destination[0]
    span=[source_p]+[p for i,p in enumerate(points) if source_s<i<target_s]+[target_p]
    hits=[p for _,p in path_hits(span,0,gate_x)]
    routed=bool(hits) and all(low+radius<=p[1]<=high-radius
                 and board_z+radius-.001<=p[2]<=board_z+height for p in hits)
    return bool(routed),hits


def score(unsafe, routed, seated, released, coordinated):
    if unsafe: return 0.
    if routed and seated and released and coordinated: return 1.
    return .5 if routed else 0.
