"""Native contact-only reference: discard complete devices, carry tray, withdraw."""
import numpy as np
from .geometry import HANDLE,SHARPS,DOCK,BIN


def actions(task,base,policy,max_steps):
    from tools.run_recovery_pilot import planned_actions
    from yam_arena import ik,station
    from yam_arena.runtime import tcp_world,arm_q
    action=task.initial_action();steps=0
    z=np.array([np.sin(np.deg2rad(20.)),0,-np.cos(np.deg2rad(20.))])
    rotations={}
    for side in ('left','right'):
        x=np.array([0,-1,0]);y=np.cross(z,x)
        rotations[side]=ik.matrix_to_quat_xyzw(np.column_stack([x,y,z])).tolist()

    def command(side):
        i=0 if side=='left' else 7
        return ik.tcp_pose(action[i:i+6],station.BASES[side])[0]

    def move(phase,seconds,side=None,position=None,grip=1.,**targets):
        nonlocal action,steps
        if side:targets[side]=(position,grip)
        stage={'phase':phase,'seconds':seconds}
        for arm,(p,g) in targets.items():stage[arm]=dict(position=np.asarray(p).tolist(),rotation_xyzw=rotations[arm],grip=g)
        for target,label,errors in planned_actions(action,[stage],max_steps-steps):
            action=target;steps+=1;yield action.copy(),label,errors

    def hold(phase,seconds):
        nonlocal steps
        for _ in range(round(seconds*20)):
            steps+=1
            if steps>max_steps:raise RuntimeError('Controller exceeds declared budget')
            yield action.copy(),phase,{}

    def align(side,goal,grip,phase):
        nonlocal action,steps
        i=0 if side=='left' else 7;target=command(side).copy()
        for _ in range(24):
            point=goal() if callable(goal) else goal
            error=np.asarray(point)-tcp_world(base.scene[side+'_robot'])
            target+=np.clip(.25*error,-.001,.001)
            q,pe,re=ik.solve(target,rotations[side],arm_q(base.scene[side+'_robot']),base_pos=station.BASES[side],iters=120)
            if pe>.003 or re>.04:raise RuntimeError('Alignment unreachable: '+phase)
            action[i:i+6]=q;action[i+6]=grip;steps+=1
            yield action.copy(),phase,dict(tcp_error_m=float(np.linalg.norm(error)))
        point=goal() if callable(goal) else goal
        if np.linalg.norm(np.asarray(point)-tcp_world(base.scene[side+'_robot']))>.005:
            raise RuntimeError('Native alignment blocked: '+phase)

    def park(side):
        yield from move('clear_'+side,1.4,side,[.22,.26 if side=='left' else -.26,.96],1.)

    def grasp_handle(side):
        tray=task.native('tray')[0];anchor=tray+HANDLE[side]+[0,0,.010]
        yield from move('approach_'+side+'_tray_handle',1.5,side,anchor+[0,0,.11],1.)
        yield from move('reach_'+side+'_tray_handle',1.,side,anchor,1.)
        yield from align(side,lambda:task.native('tray')[0]+HANDLE[side]+[0,0,.010],1.,'align_'+side+'_handle')
        yield from move('grasp_'+side+'_tray_handle',.7,side,command(side),0.)

    def discard(name,index):
        anchor=task.native(name)[0]+[0,0,.007]
        yield from move('approach_'+name,1.6,'left',anchor+[0,0,.13],1.)
        yield from move('reach_'+name,1.2,'left',anchor,1.)
        yield from align('left',lambda:task.native(name)[0]+[0,0,.007],1.,'align_'+name+'_barrel')
        yield from move('pinch_'+name+'_barrel',.6,'left',command('left'),0.)
        start=task.native(name)[0].copy()
        yield from move('lift_'+name,1.4,'left',command('left')+[0,0,.070],0.)
        if task.native(name)[0][2]-start[2]<.040:raise RuntimeError('Device did not follow native grasp: '+name)
        offset=task.native(name)[0]-tcp_world(base.scene['left_robot'])
        target=BIN+[0.,-.027 if index==0 else .027,.180]
        yield from move('carry_'+name+'_above_bin',2.,'left',target-offset,0.)
        yield from move('release_'+name+'_in_bin',.6,'left',command('left'),1.)
        yield from hold('observe_'+name+'_drop',.8)
        yield from move('clear_bin_'+name,1.,'left',command('left')+[0,0,.04],1.)
        if not task.evaluate(base)[name+'_contained']:raise RuntimeError('Whole device not contained: '+name)

    if policy=='preview':yield from hold('native_scene_preview',3.);return
    if policy not in ('clear_and_stage','skip_disposal','leave_second'):raise ValueError(policy)
    yield from grasp_handle('right')
    if policy!='skip_disposal':yield from discard('sharp_a',0)
    if policy=='clear_and_stage':
        yield from move('release_support_for_shared_workspace',.5,'right',command('right'),1.)
        yield from move('clear_right_handle',1.,'right',command('right')+[0,0,.06],1.)
        yield from park('right')
        yield from discard('sharp_b',1)
        yield from grasp_handle('right')
    yield from park('left')
    yield from grasp_handle('left')
    anchor={side:task.native('tray')[0]-tcp_world(base.scene[side+'_robot']) for side in ('left','right')}
    tray=task.native('tray')[0].copy()
    yield from move('lift_tray_with_both_arms',2.,**{s:(tray+[0,0,.05]-anchor[s],0.) for s in anchor})
    if task.native('tray')[0][2]-tray[2]<.035:raise RuntimeError('Both-arm tray grasp did not lift tray')
    yield from move('carry_tray_to_reprocessing',3.,**{s:(DOCK+[0,0,.05]-anchor[s],0.) for s in anchor})
    yield from move('lower_tray_into_reprocessing_area',2.,**{s:(DOCK-anchor[s],0.) for s in anchor})
    yield from hold('settle_tray_for_reprocessing',.6)
    yield from move('release_both_tray_handles',.7,**{s:(command(s),1.) for s in anchor})
    yield from move('lift_both_grippers_clear',1.3,**{s:(command(s)+[0,0,.045],1.) for s in anchor})
    yield from move('withdraw_both_arms',1.5,left=([.22,.26,.96],1.),right=([.22,-.26,.96],1.))
    yield from hold('observe_released_tray_for_reprocessing',2.5)
