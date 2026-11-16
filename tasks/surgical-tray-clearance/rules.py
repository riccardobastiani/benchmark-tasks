"""Geometric accounting and endpoint scores for the surgical tray clearance."""
import numpy as np


def contained(points,bin_position,half,margin=.0005):
    p=np.asarray(points)-bin_position
    # Account for native floor contact and conservative capsule bounding corners.
    return bool(np.all(np.abs(p[:,:2])<=half[:2]+margin) and np.all(p[:,2]>=.008-margin)
                and np.all(p[:,2]<=half[2]+margin))


def tray_contains(points,centre,rotation):
    local=(np.asarray(points)-centre)@rotation
    return bool(np.all(np.abs(local[:,0])<.127) and np.all(np.abs(local[:,1])<.147)
                and np.all(local[:,2]>.001) and np.all(local[:,2]<.060))


def score(tools_available,disposed,docked,released):
    if not tools_available:return 0.
    if all(disposed) and docked and released:return 1.
    return .25*sum(bool(value) for value in disposed)
