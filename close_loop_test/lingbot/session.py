"""Reuse the navigation/memory policy; replace only dataset and executor boundaries."""
from close_loop_test.common.session import Session as CommonSession
from .occ_follower import OccupancyWaypointFollower,ExplorationMotionGuard

class Session(CommonSession):
    dataset = "lingbot"
    def _task_target(self):
        ep=self.env.current_episode
        return dict(kind=ep.tasks[self.subtask_idx][1],text=ep.task_instructions[self.subtask_idx].strip().rstrip('.').rstrip()+'.')

    def _new_endpoint_follower(self):
        return OccupancyWaypointFollower(self.sim,self._agent_model_pose,forward=self.forward_m,
                turn_deg=self.turn_deg,goal_radius=self.config['habitat_goal_radius_m'],
                max_projection_m=self.config['habitat_max_projection_m'])


    def _arm_init_explore(self):
        super()._arm_init_explore()
        if self._init_explore_active:
            self._init_occ_guard=ExplorationMotionGuard(self.sim.occupancy_navigation,self.forward_m,self.turn_deg,self._agent_model_pose()[0])

    def _choose_init_endpoint_action(self,xy,heading):
        override,action,committed=self._init_occ_guard.action(*self._agent_model_pose())
        if committed:
            self._explore_map.navigation_failed('原地扫描一圈无位移')
            self._init_endpoint_goal=None;self._init_endpoint_follower=None
        if override:
            if action is None:self._explore_map.stop_reason='原地扫描后仍无安全的 OCC 前进动作'
            return action
        return super()._choose_init_endpoint_action(xy,heading)

    def _apply_motion(self,name,*,ground=False,phase='subtask'):
        result=super()._apply_motion(name,ground=ground,phase=phase)
        if phase=='init_explore':self._init_occ_guard.observe(self._agent_model_pose()[0])
        return result
