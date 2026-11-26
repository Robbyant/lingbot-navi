"""Habitat navigation with sensor nodes only; CUDA owns all RGB/depth rendering."""
import copy
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

import habitat_sim
from .worker import receive, send


class CudaSimulator(habitat_sim.Simulator):
    def __init__(self, configuration, options=None):
        options = options if options is not None else self._cuda_options
        self._cuda_io_lock = threading.RLock()
        self._cuda_process = None
        self._cuda_socket = None
        self._cuda_closed = False
        self._cuda_specs = [list(agent.sensor_specifications) for agent in configuration.agents]
        bare_agents = []
        for agent in configuration.agents:
            bare = copy.copy(agent)
            bare.sensor_specifications = []
            bare_agents.append(bare)
        # Simulator's sanitizer enables EGL whenever sensors are present at construction.
        # Construct without sensors, then create pose/action nodes directly on each agent.
        sim_cfg = configuration.sim_cfg
        sim_cfg.create_renderer = False
        sim_cfg.requires_textures = False
        bare = habitat_sim.Configuration(sim_cfg, bare_agents)
        super().__init__(bare)
        for index, agent in enumerate(configuration.agents):
            self.get_agent(index).reconfigure(agent)
        env = os.environ.copy()
        paths = [str(Path(__file__).resolve().parents[2])]
        paths.extend(options.get('python_paths', []))
        if env.get('PYTHONPATH'):
            paths.append(env['PYTHONPATH'])
        env['PYTHONPATH'] = os.pathsep.join(paths)
        env.update({str(k): str(v) for k,v in options.get('env', {}).items()})
        parent, child = socket.socketpair()
        self._cuda_socket = parent
        try:
            self._cuda_process = subprocess.Popen(
                [options.get('python', sys.executable), '-m', options.get('worker_module', 'data_toolkit.hm3d_render.worker'), str(child.fileno())],
                pass_fds=(child.fileno(),), env=env)
            child.close()
            parent.settimeout(float(options.get('startup_timeout_s', 300)))
            send(parent, dict(scene=configuration.sim_cfg.scene_id,
                             gpu=configuration.sim_cfg.gpu_device_id,
                             basis_workers=options.get('basis_workers', 4),
                             atlas_max_side=options.get('atlas_max_side', 8192),
                             worker_options=options.get('worker_options', {})))
            self._response()
            parent.settimeout(float(options.get('frame_timeout_s', 60)))
        except BaseException:
            child.close()
            self.close()
            raise

    def _response(self):
        result = receive(self._cuda_socket)
        if not result['ok']:
            raise RuntimeError(result['error'])
        return result

    def render_cameras(self, cameras):
        """Synchronous optical-pose query; never changes the live agent."""
        try:
            with self._cuda_io_lock:
                send(self._cuda_socket, cameras)
                return self._response()['observations']
        except BaseException:
            self.close()
            raise

    def get_sensor_observations(self, agent_ids=0):
        single = isinstance(agent_ids, int)
        ids = [agent_ids] if single else list(agent_ids)
        observations = {}
        for index in ids:
            state = self.get_agent(index).get_state()
            request = {}
            for spec in self._cuda_specs[index]:
                pose = state.sensor_states[spec.uuid]
                q = pose.rotation
                request[spec.uuid] = dict(position=list(map(float, pose.position)),
                    rotation=[float(q.x),float(q.y),float(q.z),float(q.w)],
                    height=int(spec.resolution[0]), width=int(spec.resolution[1]),
                    hfov=float(spec.hfov), near=float(spec.near), far=float(spec.far),
                    depth=spec.sensor_type == habitat_sim.SensorType.DEPTH)
            try:
                with self._cuda_io_lock:
                    send(self._cuda_socket, request)
                    observations[index] = self._response()['observations']
            except BaseException:
                # A timed-out socket must never be reused: its next reply could be stale.
                self.close()
                raise
        return observations[ids[0]] if single else observations

    def close(self, *args, **kwargs):
        if getattr(self, '_cuda_closed', False):
            return
        self._cuda_closed = True
        sock = getattr(self, '_cuda_socket', None)
        proc = getattr(self, '_cuda_process', None)
        if sock is not None:
            try:
                send(sock, None)
            except OSError:
                pass
            sock.close()
        if proc is not None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
        if hasattr(self, "_initialized"):
            super().close(*args, **kwargs)
