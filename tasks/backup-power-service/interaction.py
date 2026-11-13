"""Scripted maintenance demonstration: only fourteen robot targets actuate motion."""
import numpy as np
from .geometry import ISOLATORS,ISO_OFFSET,GUARD_OFFSET,GRIP_OFFSET,BUTTON,SEAT,TRAYS

def actions(task,base,policy,max_steps):
    from tools.run_recovery_pilot import planned_actions
    from yam_arena import ik,station
    from yam_arena.runtime import arm_q,tcp_world
    action=task.initial_action();steps=0
    rotations={'left':[0,1,0,0],'right':[1,0,0,0]}

    def move(phase,seconds,side=None,position=None,grip=1.,**targets):
        nonlocal action,steps
        if side:targets[side]=(position,grip)
        stage={'phase':phase,'seconds':seconds}
        for arm,(p,g) in targets.items():
            stage[arm]=dict(position=np.asarray(p).tolist(),rotation_xyzw=rotations[arm],grip=g)
        for target,label,errors in planned_actions(action,[stage],max_steps-steps):
            action=target;steps+=1
            yield action.copy(),label,errors

    def hold(phase,seconds):
        nonlocal steps
        for _ in range(round(seconds*20)):
            steps+=1
            if steps>max_steps:raise RuntimeError('Controller exceeds declared budget')
            yield action.copy(),phase,{}

    def command(side):
        i=0 if side=='left' else 7
        return ik.tcp_pose(action[i:i+6],station.BASES[side])[0]

    def align(side,goal,grip,phase):
        nonlocal action,steps
        i=0 if side=='left' else 7;target=command(side).copy()
        for _ in range(24):
            error=np.asarray(goal)-tcp_world(base.scene[side+'_robot'])
            target+=np.clip(.25*error,-.001,.001)
            q,pe,re=ik.solve(target,rotations[side],arm_q(base.scene[side+'_robot']),base_pos=station.BASES[side],iters=120)
            if pe>.003 or re>.04:raise RuntimeError('Grasp alignment unreachable: '+phase)
            action[i:i+6]=q;action[i+6]=grip;steps+=1
            yield action.copy(),phase,dict(tcp_error_m=float(np.linalg.norm(error)))
        if np.linalg.norm(np.asarray(goal)-tcp_world(base.scene[side+'_robot']))>.005:
            raise RuntimeError('Native alignment blocked: '+phase)

    def park(side):
        rotations[side]=[0,1,0,0] if side=='left' else [1,0,0,0]
        yield from move('clear_'+side,1.5,side,[.22,.26 if side=='left' else -.26,.96],1.)

    def grasp(side,name,part,offset):
        if name in ('faulty','spare'):
            rotations[side]=[.6532814824,-.6532814824,.2705980501,-.2705980501]
        elif name!='guard':rotations[side]=[2**-.5,-2**-.5,0.,0.]
        anchor=task.native(name,part)[0]+offset
        yield from move('approach_'+name,1.5,side,anchor+[0,0,.10],1.)
        yield from move('reach_'+name,1.2,side,anchor+[0,0,.010],.5)
        yield from align(side,anchor+[0,0,.010],.5,'align_'+name)
        yield from move('grasp_'+name,.6,side,command(side),0.)

    def isolate(side,name,restore=False):
        nonlocal action,steps
        yield from grasp(side,name,'slider',ISO_OFFSET)
        delta=.060 if restore else -.060
        yield from move(('restore_' if restore else 'isolate_')+name,2.,side,command(side)+[delta,0,0],0.)
        target=command(side).copy();index=0 if side=='left' else 7
        desired=0. if restore else -.060
        for _ in range(40):
            error=desired-task.measurements()[name]
            if abs(error)<.0015:break
            target[0]+=np.clip(.3*error,-.001,.001)
            q,pe,re=ik.solve(target,rotations[side],arm_q(base.scene[side+'_robot']),base_pos=station.BASES[side],iters=120)
            if pe>.003 or re>.04:raise RuntimeError('Isolator endpoint unreachable: '+name)
            action[index:index+6]=q;action[index+6]=0.;steps+=1
            yield action.copy(),'finish_'+name+'_travel',dict(slider_error_m=float(error))
        q=task.measurements()[name]
        if (q<-.012 if restore else q>-.050):raise RuntimeError('Native isolator did not move: '+name+': '+str(q))
        yield from move('release_'+name,.4,side,command(side),1.)
        yield from move('clear_'+name,1.,side,command(side)+[0,0,.11],1.)

    def cover(opening):
        previous=rotations['right']
        rotations['right']=[.9238795325,0.,.3826834324,0.]
        yield from grasp('right','guard','cover',GUARD_OFFSET)
        delta=.135 if opening else -task.measurements()['guard']
        yield from move('open_guard' if opening else 'close_guard',2.,'right',command('right')+[delta,0,0],0.)
        q=task.measurements()['guard']
        if (q<.120 if opening else q>.008):raise RuntimeError('Native guard motion failed: '+str(q))
        yield from move('release_guard',.4,'right',command('right'),.2)
        yield from move('clear_guard',1.,'right',command('right')+[0,0,.06],.2)
        yield from move('open_clear_guard_fingers',.3,'right',command('right'),1.)
        rotations['right']=previous

    if policy=='preview':yield from hold('native_scene_preview',3.);return
    if policy not in ('service','main_only','skip_test','live_touch'):raise ValueError(policy)
    if policy=='live_touch':
        yield from cover(True)
        yield from hold('observe_failure',1.);return
    yield from isolate('left','main')
    if policy=='main_only':
        yield from park('left');yield from cover(True)
        yield from hold('observe_backup_live_failure',1.);return
    if policy!='skip_test':
        yield from move('approach_bus_test',1.3,'left',BUTTON+[0,0,.11],.15)
        yield from move('reach_bus_test',1.,'left',BUTTON+[0,0,.020],.15)
        # Increment robot TCP command until native spring-button travel proves contact.
        i=0;target=command('left').copy()
        for tick in range(55):
            if task.measurements()['test']<-.006:break
            target[2]-=.0005
            q,pe,re=ik.solve(target,rotations['left'],arm_q(base.scene['left_robot']),base_pos=station.BASES['left'],iters=120)
            if pe>.003 or re>.04:raise RuntimeError('Tester press target unreachable')
            action[i:i+6]=q;action[6]=.15;steps+=1
            yield action.copy(),'press_native_test_button',{}
        if task.measurements()['test']>-.005:raise RuntimeError('Native test button did not press')
        yield from hold('verify_backup_still_powers_bus',.6)
        if not task.circuit.live_proved:raise RuntimeError('Live instrument check not recorded')
    yield from isolate('right','backup')
    yield from hold('wait_and_verify_zero_voltage',2.5)
    if policy!='skip_test' and not task.circuit.dead_proved:raise RuntimeError('Zero-voltage test not recorded')
    yield from park('left');yield from park('right')
    yield from cover(True)
    if policy=='skip_test':yield from hold('observe_unverified_service_failure',1.);return
    yield from park('right')

    # Remove the faulty cassette under native contact, then park it in the red tray.
    yield from grasp('right','faulty','unit',GRIP_OFFSET)
    anchor=task.native('faulty','unit')[0]-tcp_world(base.scene['right_robot'])
    yield from move('withdraw_faulty_cassette',2.,'right',command('right')+[-.115,0,0],0.)
    if task.native('faulty','unit')[0][0]>.29:raise RuntimeError('Faulty cassette did not follow the grasp')
    yield from move('lift_faulty_cassette',1.,'right',command('right')+[0,0,.11],0.)
    goal=TRAYS['faulty']-anchor
    yield from move('transport_faulty_to_red_tray',2.,'right',goal+[0,0,.11],0.)
    yield from move('set_faulty_on_red_tray',1.5,'right',goal,0.)
    yield from hold('settle_faulty',.4)
    yield from move('release_faulty',.6,'right',command('right'),1.)
    yield from move('clear_faulty',1.,'right',command('right')+[0,0,.12],1.)
    yield from park('right')
    if not task.evaluate(base)['faulty_parked']:raise RuntimeError('Faulty cassette is not retained in its tray')

    yield from grasp('left','spare','unit',GRIP_OFFSET)
    anchor=task.native('spare','unit')[0]-tcp_world(base.scene['left_robot'])
    yield from move('lift_replacement',1.,'left',command('left')+[0,0,.12],0.)
    yield from move('stage_replacement_above_bay',2.,'left',SEAT-anchor+[0,0,.12],0.)
    yield from move('lower_replacement_above_rails',1.5,'left',SEAT-anchor+[0,0,.035],0.)
    # Centre the carried cassette using native payload measurements, then lower
    # through the real rail clearance instead of driving into its front edge.
    from tools.run_recovery_pilot import motion_fraction
    target=command('left').copy()
    for tick in range(100):
        desired=SEAT+[0,0,.035*(1-motion_fraction(max(0,tick-25)/60))]
        error=desired-task.native('spare','unit')[0]
        target+=np.clip(.25*error,-.0015,.0015)
        q,pe,re=ik.solve(target,rotations['left'],arm_q(base.scene['left_robot']),base_pos=station.BASES['left'],iters=120)
        if pe>.003 or re>.04:raise RuntimeError('Payload seating correction unreachable')
        action[:6]=q;action[6]=0.;steps+=1
        yield action.copy(),'centre_and_seat_replacement',dict(payload_error_m=float(np.linalg.norm(error)))
    yield from hold('settle_replacement',.5)
    yield from move('release_replacement',.6,'left',command('left'),1.)
    yield from move('clear_replacement',1.,'left',command('left')+[0,0,.12],1.)
    yield from park('left')
    yield from hold('confirm_replacement_seated',.5)
    if not task.evaluate(base)['replacement_seated']:raise RuntimeError('Replacement did not seat: '+repr(task.native('spare','unit')[0]))
    yield from cover(False)
    yield from park('right')
    yield from isolate('right','backup',True)
    yield from park('right')
    yield from isolate('left','main',True)
    yield from park('left')
    yield from hold('prove_restored_service_hands_clear',3.)
