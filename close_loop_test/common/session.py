"""Dataset-independent episode, planning and memory orchestration."""
import time
from pathlib import Path
import cv2
import numpy as np
from .policy.controller import BaselinePolicy, jpeg, _slug
from .policy.localization_tf import StaticLocalizationTF
from .policy.routes import habitat_heading_model_xy

class PlannerBridge:
    def __init__(self,client):self.client=client
    def localize_view(self,scene,memories,image):
        return self.client.request('localize',memories=list(map(str,memories)),image=str(image))
    def select_task_memory(self,scene,memories,image):
        newest=sorted(range(len(memories)),key=lambda i:(Path(memories[i]).stat().st_mtime_ns,i),reverse=True)
        return self.client.request('localize',memories=list(map(str,memories)),image=str(image),select=True,newest_first=newest)

class Session(BaselinePolicy):
    dataset = None

    def __init__(self,env,client,output,budget,event,save_observation,*,map1=False,init_explore_steps=200,independent_initial_exploration=False):
        self.env,self.client,self.output=env,client,output
        self.sim=env.sim;self.agent=self.sim.get_agent(0)
        self.observations={k:np.array(v,copy=True) for k,v in env.sim.get_sensor_observations().items() if k in ('rgb','depth')}
        self.event,self.save_observation=event,save_observation
        self.config=dict(dataset=self.dataset,lingbot_map1=map1,global_plan_work=output/'planning',spatial_memory=True,
            init_explore_max_steps=init_explore_steps,init_explore_navigation_reserve=200,
            independent_initial_exploration=independent_initial_exploration,
            habitat_goal_radius_m=.20,habitat_max_projection_m=2.,habitat_max_height_delta_m=.5,
            allow_direct_goal=True)
        self.camera={'rgb':dict(width=640,height=480,hfov=90,position=[0,1.5,0])}
        self.forward_m,self.turn_deg,self.budget=.25,30.,budget
        ep=env.current_episode
        self.episode=dict(episode_id=ep.episode_id,tasks=ep.tasks)
        self.cases=[dict(scene_name=Path(ep.scene_id).name.replace('.basis.glb', ''))];self.case_pos=0
        self.global_planner=PlannerBridge(client)
        self._plan_ui={'state':'waiting'};self._spatial_ui={'state':'idle'};self._plan_token=0
        self._localization_tf=StaticLocalizationTF()
        self._tf_history_samples=[];self._tf_history_quality={}
        self._task_memory_selection=None
        self._episode_active=True;self._init_explore_active=False;self._init_explored_scenes=set()
        self._explore_map=None;self._auto_follow=None;self._follow=self._blank_follow()
        self.step=self.subtask_idx=self.subtask_step=0;self.done=False;self.path_length=0.
        self.memories=[];self._skip_push=False;self._last_motion_collided=False;self.plans=0
        self._begin_subtask(plan=False)
        self._arm_init_explore()

    def current_task(self):return None if self.done else self.episode['tasks'][self.subtask_idx]
    def _current_scene_memories(self):return list(self.memories)
    def _record_behavior(self,kind,message,**details):
        if kind in ('action', 'subtask_end'):
            state = self.agent.get_state()
            q = state.rotation
            details.update(position=np.asarray(state.position).tolist(),
                           rotation_xyzw=[q.x, q.y, q.z, q.w], pose_frame='habitat_y_up')
        self.event(kind,step=self.step,subtask=self.subtask_idx,message=message,**details)
    def _has_independent_initial_exploration_budget(self):return self.config.get('independent_initial_exploration',False)
    def _agent_model_pose(self):
        state=self.agent.get_state();q=state.rotation
        return np.array([float(state.position[0]),-float(state.position[2])],dtype=np.float64),np.asarray(habitat_heading_model_xy([q.x,q.y,q.z,q.w]))
    def _sensor_pose(self):
        state=self.agent.get_state().sensor_states['rgb'];q=state.rotation
        return np.asarray(state.position),np.array([q.x,q.y,q.z,q.w])
    def _current_rgb_depth(self):
        return np.asarray(self.observations['rgb'])[...,:3],np.asarray(self.observations['depth']).reshape(480,640)
    def observation_jpegs(self):return jpeg(cv2.cvtColor(self._current_rgb_depth()[0],cv2.COLOR_RGB2BGR)),None
    def _observe(self):
        self.save_observation(self.env,self.observations,self.output, **({'map1':True} if self.config['lingbot_map1'] else {}))
        self.client.request('observe',bundle=str(self.output/'observation.npz'),frame_id=self.step)
    def _restart_sim_geom(self):
        self._observe();self._skip_push=True;self._begin_frame_id=self.step
    def _push_sim_geom(self):
        if self._skip_push:
            self._skip_push=False
            # Learned geometry must retain the first new motion frame.
            if not self.config['lingbot_map1'] or self.step==self._begin_frame_id:return
        self._observe()
    def _finish_memory_segment(self,phase='subtask'):
        segment='ep'+_slug(str(self.episode['episode_id']))+('_sub%d'%(self.subtask_idx+1) if phase=='subtask' else '_'+phase)
        self._spatial_ui=self.client.request('finish',segment=segment,phase=phase)
        self.memories=self._spatial_ui['paths']
        if self.config['lingbot_map1']:
            self._episode_wait_since=time.monotonic()
            if self._auto_follow is not None:self._auto_follow['wait_since']=self._episode_wait_since
        self._record_behavior('memory_finish','',state=self._spatial_ui['state'],memory_count=len(self.memories))
    def _task_target(self):
        raise NotImplementedError("Dataset session must provide task targets")
    def _begin_subtask(self,plan=True):
        self._task_explorations=0;self._task_exploration_closed=False
        self._text_task_search=self.current_task()[1] in ('object','description')
        self._pre_task_explore_pending=False;self._task_has_retrieval=False;self._previous_goal_xy=None
        self._blocked_a_goals=[];self._visited_a_goals=[];self._episode_wait_since=time.monotonic()
        self.subtask_step=0;self.path_length=0.
        self.target=self._task_target()
        self._record_behavior('subtask_start','',target=self.target)
        if not self._repeat_subtask_pending():
            self._restart_sim_geom()
            if plan:self._request_global_plan()
    def _submit_auto_task(self,reason):
        self._end_follow(reason,ground=False)
        self.last_action='subtask_stop';self.step+=1;self.subtask_step+=1
        self._finish_subtask('stop',submit_reason=reason)
    def _finish_subtask(self,reason,submit_reason=None):
        self._auto_follow=None;self._follow=self._blank_follow();self._init_explore_active=False
        self._finish_memory_segment()
        self._record_behavior('action',submit_reason or reason,action='subtask_stop')
        self.env.step({'action':'subtask_stop'})
        self._record_behavior('subtask_end',submit_reason or reason,metrics=self.env.get_metrics(),subtask_steps=self.subtask_step)
        self.subtask_idx+=1;self.done=self.subtask_idx>=len(self.episode['tasks'])
        if not self.done:self._begin_subtask()
    def _apply_motion(self,name,*,ground=False,phase='subtask'):
        independent_explore = self._has_independent_initial_exploration_budget() and phase.startswith('init_explore')
        if not independent_explore and self._submit_at_budget_boundary():return 'timeout'
        before=np.asarray(self.agent.get_state().position).copy()
        self._record_behavior('action',phase,action=name)
        self.env.step({'action':name})
        # Preserve the action's raw frame before TF probes render another view.
        # Habitat-Lab's observation wrapper clips depth; LingBot-Navi consumes raw metres.
        self.observations={k:np.array(v,copy=True) for k,v in self.env.sim._prev_sim_obs.items() if k in ('rgb','depth')}
        after=np.asarray(self.agent.get_state().position)
        moved=float(np.linalg.norm(after-before))
        self._last_motion_collided=name=='move_forward' and (self.env.sim.previous_step_collided or moved<self.forward_m*.8)
        self.path_length+=moved;self.step+=1
        if not independent_explore:self.subtask_step+=1
        self._capture_tf_history_sample()
        self._push_sim_geom()
        return 'blocked' if self._last_motion_collided else 'ok'
    def _begin_plan_alignment(self,heading=None):self._refresh_follow_action()
    def _refresh_follow_action(self):
        if self._follow['active']:self._refresh_endpoint_follow_action()
    def _initialize_tf_from_exploration(self,scene,anchor,memories=None):
        return self._initialize_task_tf(scene,memories or [anchor],stage='initial_exploration')
    def _end_init_explore(self,reason):
        self._init_explore_active=False
        self._record_behavior('init_end',reason)
        if self.done:return
        self._finish_memory_segment(phase='init_explore');self._restart_sim_geom()
        for _ in range(self._init_explore_look_steps):
            if self._apply_motion('look_up',phase='init_explore_restore')=='timeout':return
        self._init_explore_look_steps=0
        self._request_global_plan()
    def _wait_exploration_memory(self,auto):
        # Synchronous IPC has already completed this exact segment; no polling/callback races.
        if self._spatial_ui['state'] not in ('done','skipped'):raise RuntimeError(str(self._spatial_ui))
        auto.pop('memory_phase')
        retry=auto.pop('memory_retry_reason',None)
        if retry is None:self._blocked_a_goals=[]
        if 'coverage_before' in auto:self._apply_exploration_gain(auto)
        self._replan_follow('失败重试记忆已更新：'+retry if retry else '探索记忆已更新')
    def _retry_exploration_memory(self,reason):
        if self.subtask_step>=self.budget:self._stop_follow_for_timeout();return
        if self._task_exploration_closed:self._submit_auto_task(reason);return
        self._auto_follow.pop('exploration_memory_retry',None)
        self._record_behavior('explore_end',reason)
        self._finish_exploration_segment()
    def _request_global_plan(self,force=False):
        if self.done or self._repeat_subtask_pending() or not self.memories:return
        scene=self.cases[0]['scene_name'];task=self.subtask_idx
        try:
            memories=self._task_planning_memories(scene,self.memories)
            recovery=(self._auto_follow or {}).get('exploration_memory_retry')
            if recovery:memories=recovery['memories']
            selection=self._select_observation_view(scene,memories)
            if selection is None or self.done or task!=self.subtask_idx:return
            view,current=selection
            if current is None:raise ValueError('No confirmed localization TF')
            self._plan_token+=1
            image=self.output/'planning'/('rgb_%d.jpg'%self._plan_token)
            image.write_bytes(self.observation_jpegs()[0])
            result=self.client.request('plan',memories=memories,image=str(image),target=self.target,
                current_localization=current,previous_goal_xy=self._previous_goal_xy if force else None,
                force_explore=bool(recovery or self._pre_task_explore_pending),
                exploration_goal_world=(self._auto_follow or {}).get('explore_target'))
            self.plans+=1;self._plan_ui=result
            self._record_behavior('plan','',plan=self.plans,memory_count=len(memories),modes=[m['mode'] for m in result['modes']])
        except (OSError,RuntimeError,ValueError,KeyError) as error:
            self._plan_ui=dict(state='error',text=str(error))
            self._record_behavior('planning_error',str(error))
