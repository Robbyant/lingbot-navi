"""Small API bridge for official GOAT on newer Habitat-Sim; metrics are unchanged."""

def register_simulator(renderer_options):
    from habitat.core.registry import registry
    from goat_bench.task.simulator import GOATSim

    from data_toolkit.hm3d_render.simulator import CudaSimulator

    import habitat_sim
    from habitat.core.embodied_task import SimulatorTaskAction
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    class NativeLookAction(SimulatorTaskAction):
        def step(self, *args, **kwargs):
            action_id = getattr(HabitatSimActions, self.native_name)
            # Habitat-Lab's look action multiplies rotations directly. Use the
            # native simulator control, including its quaternion normalization,
            # to reproduce the qwen G01 sensor poses and millimetre depth bins.
            self._sim.get_agent(0).agent_config.action_space[action_id] = habitat_sim.agent.ActionSpec(
                self.native_name, habitat_sim.agent.ActuationSpec(amount=float(self._config.tilt_angle)))
            return self._sim.step(action_id)

    @registry.register_task_action(name='LingBotNativeLookUp')
    class NativeLookUp(NativeLookAction):
        native_name = 'look_up'

    @registry.register_task_action(name='LingBotNativeLookDown')
    class NativeLookDown(NativeLookAction):
        native_name = 'look_down'

    @registry.register_simulator(name='LingBotGoatSim-v0')
    class LingBotGoatSim(GOATSim, CudaSimulator):
        _cuda_options = renderer_options

        def recompute_navmesh(self, pathfinder, settings, include_static_objects=False):
            # Habitat-Sim 0.3 moved this keyword into NavMeshSettings.
            if hasattr(settings, 'include_static_objects'):
                settings.include_static_objects = include_static_objects
                return super().recompute_navmesh(pathfinder, settings)
            return super().recompute_navmesh(
                pathfinder, settings, include_static_objects=include_static_objects)


def apply_navigation_compatibility(sim):
    """Release a no-sliding boundary deadlock in Habitat-Sim 0.3.3+.

    Preserve native 0.3.2 execution. Only inward, reversible, on-ray numerical
    boundary corrections are allowed; goal and collision limits are unchanged.
    Call after env.reset(), which may recreate the agent controls.
    """
    import habitat_sim
    from .policy.navmesh_motion import install_boundary_filter
    version = tuple(int(part) for part in habitat_sim.__version__.split('.')[:3])
    enabled = version >= (0, 3, 3) and not sim.config.sim_cfg.allow_sliding
    if enabled:
        install_boundary_filter(sim)
    return dict(habitat_sim_version=habitat_sim.__version__,
                boundary_filter='ray_constrained_1e-5m' if enabled else 'native')
