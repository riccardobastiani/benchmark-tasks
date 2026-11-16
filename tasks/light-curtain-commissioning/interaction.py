"""Robot joint commands only: support, unlock, raise, lock, align and inspect."""
import numpy as np
from .geometry import Z,FRAME,HANDLE,RX,RX_HANDLE,BOLT,BOLT_GRIP,ROD,ROD_GRIP,BUTTONS

def actions(task,base,policy,max_steps):
    from tools.run_recovery_pilot import planned_actions
    from yam_arena import ik,station
    from yam_arena.runtime import tcp_world,arm_q
    from yam_arena.urdf import quat_xyzw_to_matrix
    action=task.initial_action();steps=0
    def rotation(deg,lateral=False):
        s,c=np.sin(np.deg2rad(deg)),np.cos(np.deg2rad(deg))
        z=np.array([0,-s,-c]) if lateral else np.array([s,0,-c])
        x=np.array([0,-c,s]) if lateral else np.array([0,-1,0])
        return ik.matrix_to_quat_xyzw(np.column_stack([x,np.cross(z,x),z])).tolist()
    rotations={s:rotation(20) for s in ('left','right')}
    def command(side):return ik.tcp_pose(action[(i:=0 if side=='left' else 7):i+6],station.BASES[side])[0]
    def move(phase,seconds,side=None,position=None,grip=1.,**targets):
        nonlocal action,steps
        if side:targets[side]=(position,grip)
        stage={'phase':phase,'seconds':seconds}
        for arm,(p,g) in targets.items():stage[arm]={'position':np.asarray(p).tolist(),'rotation_xyzw':rotations[arm],'grip':g}
        for target,label,err in planned_actions(action,[stage],max_steps-steps):
            action=target;steps+=1;yield action.copy(),label,err
    def hold(phase,seconds):
        nonlocal steps
        for _ in range(round(seconds*20)):
            steps+=1
            if steps>max_steps:raise RuntimeError('Commissioning exceeds declared budget')
            yield action.copy(),phase,{}
    def align(side,goal,grip,phase,count=24):
        nonlocal action,steps
        i=0 if side=='left' else 7;target=command(side).copy()
        for _ in range(count):
            p=goal() if callable(goal) else goal;error=np.asarray(p)-tcp_world(base.scene[side+'_robot'])
            target+=np.clip(.25*error,-.001,.001)
            q,pe,re=ik.solve(target,rotations[side],arm_q(base.scene[side+'_robot']),base_pos=station.BASES[side],iters=120)
            if pe>.003 or re>.04:raise RuntimeError('Unreachable alignment: '+phase)
            action[i:i+6]=q;action[i+6]=grip;steps+=1
            yield action.copy(),phase,{'tcp_error_m':float(np.linalg.norm(error))}
        point=goal() if callable(goal) else goal
        if np.linalg.norm(np.asarray(point)-tcp_world(base.scene[side+'_robot']))>.006:
            raise RuntimeError('Native TCP did not reach grasp: '+phase)
    def park(side):yield from move('withdraw_'+side,1.3,side,[.18 if side=='left' else .22,.36 if side=='left' else -.33,.96],1.)
    def grasp(side,goal,name,grip=0.):
        p=goal() if callable(goal) else goal
        yield from move('approach_'+name,1.4,side,p+[0,0,.09],1.)
        yield from move('reach_'+name,1.,side,p,1.)
        yield from align(side,goal,1.,'align_'+name)
        yield from move('grasp_'+name,.6,side,command(side),grip)
    def frame_handle():return task.native('mount','frame')[0]+HANDLE+[0,0,.010]
    def bolt_handle():return task.native('mount','height_bolt')[0]+BOLT_GRIP+[0,0,.010]
    def receiver_handle():
        p,q,_=task.native('mount','receiver');return p+quat_xyzw_to_matrix(q)@RX_HANDLE+[0,0,.010]
    def press_down(name):
        nonlocal action,steps
        p=BUTTONS[name]
        yield from move('approach_'+name,1.,'right',p+[0,0,.08],.15)
        yield from move('reach_'+name,.8,'right',p+[0,0,.020],.15)
        target=command('right').copy()
        for _ in range(44):
            if task.native(name)[0][2]-p[2]<-.005:break
            target[2]-=.0005
            q,pe,re=ik.solve(target,rotations['right'],arm_q(base.scene['right_robot']),base_pos=station.BASES['right'],iters=120)
            if pe>.003 or re>.04:raise RuntimeError('Button unreachable: '+name)
            action[7:13]=q;action[13]=.15;steps+=1
            yield action.copy(),'press_native_'+name,{}
        if task.native(name)[0][2]-p[2]>-.004:raise RuntimeError('Native button not pressed: '+name)
    def release_button(name):yield from move('release_'+name,.8,'right',command('right')+[0,0,.09],.15)
    def pulse(name):
        yield from press_down(name);yield from hold('observe_'+name+'_response',.35);yield from release_button(name)

    if policy=='preview':yield from hold('native_scene_preview',3.);return
    if policy not in ('commission','mount_only','low_gap','free_frame','controls_only'):raise ValueError(policy)
    if policy!='low_gap':
        yield from grasp('left',frame_handle,'frame_support')
        yield from grasp('right',bolt_handle,'height_lock')
        yield from move('withdraw_height_lock',1.,'right',command('right')+[-.050,0,0],0.)
        if task.measurements()['bolt']>-.035:raise RuntimeError('Native height lock did not withdraw')
        old=task.measurements()['height']
        yield from move('raise_optical_frame',2.,'left',command('left')+[0,0,.050-old],0.)
        for _ in range(12):
            error=.050-task.measurements()['height']
            if abs(error)<.0005:break
            yield from move('finish_frame_height',.15,'left',command('left')+[0,0,np.clip(error,-.001,.001)],0.)
        if task.measurements()['height']<.045:raise RuntimeError('Frame did not follow gripper')
        if policy=='free_frame':
            yield from move('release_unlocked_frame',.5,'left',command('left'),1.)
            yield from move('clear_unlocked_frame',.8,'left',command('left')+[0,0,.06],1.)
            yield from hold('observe_actual_unlocked_fall',2.);return
        yield from move('insert_height_lock',1.,'right',BOLT+BOLT_GRIP+[0,0,.010],0.)
        yield from move('release_height_lock',.4,'right',command('right'),1.)
        yield from move('clear_height_lock',.7,'right',command('right')+[0,0,.06],1.)
        yield from move('retract_from_height_lock',.7,'right',command('right')+[-.080,0,0],1.)
        if task.measurements()['bolt']<-.008:raise RuntimeError('Height indexing pin did not re-engage')
        yield from move('test_locked_frame_against_small_lift',.7,'left',command('left')+[0,0,.004],0.)
        if task.measurements()['height']>.053:raise RuntimeError('Locked frame failed the native holding check')
        yield from move('unload_frame_grip',.5,'left',command('left')+[0,0,-.004],0.)
        yield from move('release_frame_support',.5,'left',command('left'),1.)
        yield from move('lift_clear_of_frame',1.,'left',[.170,.180,1.020],1.)
        yield from park('left')
    if not task.measurements()['aligned'] or task.measurements()['pin']>.004:
        yield from grasp('right',receiver_handle,'receiver_adjustment')
        goal=FRAME+[0,0,task.measurements()['height']]+RX+RX_HANDLE+[0,0,.010]
        yield from move('align_receiver_to_sender',1.2,'right',goal,0.)
        yield from hold('observe_alignment_index_engagement',.5)
        yield from move('release_receiver',.5,'right',command('right'),1.)
        yield from move('clear_receiver',.8,'right',command('right')+[-.080,0,.04],1.)
    yield from park('right');yield from hold('verify_released_mount',1.5)
    if not task.measurements()['aligned']:raise RuntimeError('Receiver did not retain optical alignment')
    if policy in ('mount_only','low_gap'):yield from hold('observe_mount_result',2.);return
    if not task.measurements()['locked'] or not task.measurements()['covered']:raise RuntimeError('Installation failed native endpoint')
    rotations['left']=rotation(20,lateral=True)
    def rod_grip():
        p,q,_=task.native('rod');return p+quat_xyzw_to_matrix(q)@ROD_GRIP
    yield from grasp('left',rod_grip,'test_rod',.12)
    initial=task.native('rod')[0].copy()
    yield from move('lift_test_rod',1.1,'left',command('left')+[0,0,.10],.12)
    if task.native('rod')[0][2]-initial[2]<.06:raise RuntimeError('Native probe grasp failed')
    offset=task.native('rod')[0]-tcp_world(base.scene['left_robot'])
    def probe(y,h,x=.280):return np.array([x,y,Z+h])-offset
    yield from move('retract_probe_before_posts',1.5,'left',probe(.32,.12,.180),.12)
    yield from move('move_probe_to_opening',4.5,'left',probe(-.120,.062,.180),.12)
    yield from move('insert_probe_in_opening',1.8,'left',probe(-.120,.062),.12)
    if policy=='controls_only':
        yield from move('position_probe_for_control_diagnostic',1.5,'left',probe(0.,.130),.12)
    else:
        # Mean path speed 12 mm/s => minimum-jerk peak 22.5 mm/s; retain margin for native contact motion.
        path=[(-.120,.198),(.120,.198),(.120,.062),(-.120,.062)]
        previous=np.array([-.120,.062])
        for i,(y,h) in enumerate(path):
            duration=float(np.linalg.norm(np.array([y,h])-previous)/.012)
            yield from move('slow_opening_edge_check_'+str(i),duration,'left',probe(y,h),.12)
            yield from hold('observe_field_response',.15);previous=np.array([y,h])
        yield from move('position_probe_for_centre_check',1.3,'left',probe(0.,.062),.12)
        yield from move('slow_centre_field_check',.136/.012,'left',probe(0.,.198),.12)
        yield from hold('observe_completed_field_check',.5)
        if not task.proof.complete:raise RuntimeError('Qualified rod coverage incomplete: '+str(task.proof.fraction))
    yield from pulse('mode')
    if not task.control.mode:raise RuntimeError('Native TEST mode button did not latch')
    yield from pulse('start');yield from hold('verify_blocked_start_inhibited',.35)
    if not task.control.inhibit:raise RuntimeError('Blocked start inhibit not proved')
    yield from move('remove_probe_for_reset',1.5,'left',probe(0.,.130,.180),.12)
    yield from pulse('reset');yield from hold('verify_reset_alone_does_not_start',.35)
    if not task.control.reset_test:raise RuntimeError('Reset-without-motion not proved')
    yield from press_down('start')
    yield from move('interrupt_moving_test_stroke',2.,'left',probe(0.,.130),.12)
    yield from hold('observe_finite_press_stop',.4);yield from release_button('start')
    if not task.control.stop:raise RuntimeError('Native moving stop not proved')
    yield from move('withdraw_probe_after_stop',1.5,'left',probe(0.,.130,.180),.12)
    yield from move('return_probe_outside_posts',4.,'left',probe(.32,.12,.180),.12)
    if np.linalg.norm(task.native('rod')[0]-(tcp_world(base.scene['left_robot'])+offset))>.025:
        raise RuntimeError('Native rod grasp lost during return')
    yield from move('approach_probe_cradle',1.2,'left',ROD-offset+[0,0,.08],.12)
    yield from move('seat_probe_in_cradle',1.2,'left',ROD-offset,.12)
    yield from move('release_probe',.5,'left',command('left'),1.)
    yield from move('clear_probe_cradle',.8,'left',command('left')+[0,0,.08],1.)
    rotations['left']=rotation(20)
    yield from park('left')
    yield from pulse('reset');yield from pulse('start')
    yield from park('right');yield from hold('complete_guarded_empty_cycle',4.)
    if task.control.cycles<1:raise RuntimeError('Guarded cycle did not complete')
    yield from hold('observe_completed_commissioning',2.)
