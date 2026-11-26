"""Static Lingbot OCC environment; Habitat supplies pose/action nodes, not geometry."""
import math
from types import SimpleNamespace
import numpy as np
from .occupancy import OccupancyNavigation,install_occupancy_filter
from .start_pose import set_initial_sensor_pitch

class Environment:
    def __init__(self,release,scene_id,episode_index,renderer_python,*,gpu=0,seed=42,renderer_options=None):
        import habitat_sim as hs
        import quaternion
        from data_toolkit.hm3d_render.simulator import CudaSimulator
        self.release=release;self.scene=release.scene(scene_id,verify_assets=True)
        content=release.content(release.root/'tasks/static'/f'{scene_id}.json')
        if not 0 <= episode_index < len(content['episodes']):raise ValueError('episode-index outside scene')
        episode=content['episodes'][episode_index]
        self.current_episode=SimpleNamespace(**episode)
        self.atlas=release.make_atlas();self.navigation=OccupancyNavigation.load(self.scene)
        self.boxes=[release.regions[scene_id+'/'+str(t[2])]['target_annotation']['success_region']['bbox_union'] for t in episode['tasks']]
        self.fields=[self.navigation.field(self.atlas.goal_cells(scene_id,str(t[2]),b,self.navigation)) for t,b in zip(episode['tasks'],self.boxes)]
        cfg=hs.SimulatorConfiguration();cfg.scene_id='NONE';cfg.enable_physics=False;cfg.gpu_device_id=gpu;cfg.allow_sliding=False
        agent=hs.agent.AgentConfiguration();agent.radius=self.scene['agent_radius'];agent.height=self.scene['agent_height']
        agent.action_space={name:hs.agent.ActionSpec(name,hs.agent.ActuationSpec(amount=amount)) for name,amount in [('move_forward',.25),('turn_left',30.),('turn_right',30.),('look_up',30.),('look_down',30.)]}
        specs=[]
        for name,kind in [('rgb',hs.SensorType.COLOR),('depth',hs.SensorType.DEPTH)]:
            spec=hs.CameraSensorSpec();spec.uuid=name;spec.sensor_type=kind;spec.resolution=[480,640];spec.hfov=90.;spec.position=[0.,1.5,0.];spec.near=.01;spec.far=100.
            specs.append(spec)
        agent.sensor_specifications=specs
        options=dict(renderer_options or {});options.setdefault('frame_timeout_s',900);options.update(python=str(renderer_python),worker_module='close_loop_test.lingbot.render_worker',worker_options=self.scene)
        self.sim=CudaSimulator(hs.Configuration(cfg,[agent]),options)
        try:
            self.sim.seed(seed)
            install_occupancy_filter(self.sim,self.navigation)
            self.sim.previous_step_collided=False
            state=self.sim.get_agent(0).get_state();state.position=np.asarray(episode['start_position'],np.float32)
            q=episode['start_rotation'];state.rotation=quaternion.quaternion(q[3],*q[:3]);state.sensor_states={}
            self.sim.get_agent(0).set_state(state,reset_sensors=True)
            set_initial_sensor_pitch(self.sim.get_agent(0),episode['lingbo']['start_sensor_pitch_rad'])
            self.index=0;self.records=[];self.path=0.;self.start_distance=self.distance()
            self.sim._prev_sim_obs=self.sim.get_sensor_observations()
        except BaseException:
            self.close();raise

    @property
    def episode_over(self):return self.index==len(self.current_episode.tasks)
    def position(self):return np.asarray(self.sim.get_agent(0).get_state().position).copy()
    def distance(self):return self.navigation.distance(self.position()[[0,2]]*[1,-1],self.fields[self.index])
    def reset(self):return self.sim._prev_sim_obs
    def step(self,action):
        if self.episode_over:raise RuntimeError('Episode is complete')
        name=action['action']
        if name=='subtask_stop':
            task=self.current_episode.tasks[self.index];position=self.position()
            success=bool(self.navigation.component(position[[0,2]]*[1,-1]) and self.atlas.contains([position],self.scene['scene_id'],str(task[2]),self.boxes[self.index])[0])
            start=self.start_distance;denom=max(start or 0.,self.path)
            efficiency=0. if start is None or not math.isfinite(start) else (1. if denom<=1e-8 else start/denom)
            self.records.append(dict(success=success,spl=float(success)*efficiency,start_distance=start,end_distance=self.distance(),path_length=self.path))
            self.index+=1;self.path=0.
            if not self.episode_over:self.start_distance=self.distance()
        else:
            if name not in self.sim.get_agent(0).agent_config.action_space:raise ValueError('Unknown action: '+name)
            before=self.position();self.sim.occupancy_last_block=None
            self.sim._prev_sim_obs=self.sim.step(name)
            self.sim.previous_step_collided=self.sim.occupancy_last_block is not None
            self.path+=float(np.linalg.norm(self.position()-before))
        return self.sim._prev_sim_obs
    def get_metrics(self):
        return dict(success=dict(subtask_success=[float(r['success']) for r in self.records]),spl=dict(spl_by_subtask=[r['spl'] for r in self.records]),subtasks=self.records)
    def close(self):self.sim.close()
