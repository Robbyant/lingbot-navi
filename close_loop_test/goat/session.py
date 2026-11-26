"""GOAT task targets and Habitat/NavMesh executor."""
from close_loop_test.common.session import Session as CommonSession
from .goals import task_target
from .policy.habitat_follower import HabitatWaypointFollower

class Session(CommonSession):
    dataset = 'goat'

    def _task_target(self):
        return task_target(self.env, self.subtask_idx, self.output)

    def _new_endpoint_follower(self):
        config = self.config
        return HabitatWaypointFollower(self.sim, goal_radius=config.get('habitat_goal_radius_m', 0.2), max_projection_m=config.get('habitat_max_projection_m', 2.0), max_height_delta_m=config.get('habitat_max_height_delta_m', 0.5))
