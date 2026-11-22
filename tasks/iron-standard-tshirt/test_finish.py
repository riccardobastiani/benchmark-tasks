"""Small numerical checks of the finish rubric; no simulated-run claims."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('shirt_check', root/'task.py',
                                                 submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    task = module.make_task('recoverable')
    from shirt_check.front_finish import FrontFinish
    from shirt_check.soleplate import measure_rest

    points = task.rest + [.09, 0., .7585]
    iron = np.array(task.iron_rest)
    quaternion = np.array([0., 0., 0., 1.])
    geometry = task.geometry(points)
    coverage = dict(front_loaded_crease_coverage=.8)
    rest_metrics = measure_rest(iron, quaternion, task.iron_rest, module.REST_HALF_XY)
    tcp = dict(left=np.array([0., .6, 1.1]), right=np.array([0., -.6, 1.1]))

    def trial(p=points, *, covered=coverage, park=rest_metrics, hands=tcp, frames=20):
        finish = FrontFinish(task.rest, task.faces[:task.n_face])
        for step in range(frames):
            out = finish.observe(step, p, geometry=task.geometry(p), coverage=covered,
                rest_metrics=park, tcp_m=hands, iron_position=iron, iron_orientation=quaternion)
        return out

    assert trial()['score'] == 1.
    assert trial(frames=19)['score'] == 0.
    assert trial(covered={})['score'] == 0.  # Idle or hover cannot earn credit.
    assert trial(park={**rest_metrics, 'iron_rest_xy_clearance_mm': -.01})['score'] == 0.
    assert trial(hands={**tcp, 'left': points[0]})['score'] == 0.
    try:
        trial(hands={**tcp, 'left': np.full(3, np.nan)})
    except ValueError:
        pass
    else:
        raise AssertionError('An invalid native endpoint must invalidate the trial')
    finish = FrontFinish(task.rest, task.faces[:task.n_face])
    wrinkled = points.copy()
    wrinkled[finish.rows[2][8], 2] += .003
    assert trial(wrinkled)['score'] == .8  # One unfinished station loses only its credit.
    # Tilt and translation do not change a geometric straightedge gap.
    rotated = points @ np.array([[1., 0., 0.], [0., .8, -.6], [0., .6, .8]]).T
    assert np.allclose(finish.gaps(points)[0], finish.gaps(rotated)[0], atol=1e-9)
    print('Finish checks passed: release window, idle/hover, parking, held cloth, partial finish and rigid invariance.')


if __name__ == '__main__':
    main()
