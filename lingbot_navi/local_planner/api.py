"""Single-stream learned local planning from RGB, odometry and a world route."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image
import torch

from .checkpoint import load_model
from .controller import ControllerDecision, LocalPrediction, LocalTrajectoryController
from .history import PoseKeyframes
from .route import rotation_matrix, route_condition_input


@dataclass(frozen=True)
class LocalPlannerOptions:
    reference_interval_steps: int = 3
    observation_interval_s: float = 0.1
    trajectory_step_m: float = 0.25
    turn_angle_deg: float = 15.0
    arrival_threshold: float = 0.5
    arrival_distance_m: float = 0.1

    def __post_init__(self):
        if type(self.reference_interval_steps) is not int or self.reference_interval_steps < 1:
            raise ValueError("reference_interval_steps must be a positive integer")
        for name in ("observation_interval_s", "trajectory_step_m", "turn_angle_deg", "arrival_distance_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.arrival_threshold < 1 or self.turn_angle_deg > 180:
            raise ValueError("Invalid arrival threshold or turn angle")


@dataclass(frozen=True)
class LocalPlannerStep:
    prediction: LocalPrediction
    decision: ControllerDecision

    def to_dict(self):
        return {"prediction": self.prediction.to_dict(), "action": self.decision.action,
                "controller": dict(self.decision.diagnostics)}


class LocalPlanner:
    """Load local weights once and start a new session for each global plan."""

    def __init__(self, model, processor, route_config, backend, options):
        self._model = model
        self._processor = processor
        self._route_config = route_config
        self._backend = backend
        self._options = options

    @classmethod
    def from_pretrained(cls, checkpoint, vision_path, *, device="cuda:0", options=None, expected_sha256=None):
        loaded = load_model(checkpoint, vision_path, torch.device(device), expected_sha256)
        return cls(*loaded, options or LocalPlannerOptions())

    def start(self, route_world):
        """Start a new plan. Route and observation poses must use the same metric Z-up world."""
        return LocalPlannerSession(self, route_world)


class LocalPlannerSession:
    def __init__(self, planner, route_world):
        route = np.array(route_world, dtype=np.float32, copy=True)
        if route.ndim != 2 or route.shape[1] != 3 or len(route) < 2 or not np.isfinite(route).all():
            raise ValueError("World route requires finite [N>=2,3] points")
        route.setflags(write=False)
        self._planner = planner
        self._route = route
        self._history = PoseKeyframes(planner._route_config)
        self._controller = LocalTrajectoryController(planner._options)
        self._step = 0
        self._reference = -1
        self._frames = {}

    @torch.inference_mode()
    def step(self, image, *, position, rotation_xyzw):
        """Predict ten slots from one PIL image and choose the first action.

        The quaternion rotates body axes (X forward, Y left, Z up) into the
        same world frame as the route. Each successful call consumes one observation.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")
        position = np.asarray(position, dtype=np.float64)
        rotation = np.asarray(rotation_xyzw, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position requires finite world XYZ")
        rotation_matrix(rotation)
        rotation = rotation / np.linalg.norm(rotation)

        planner = self._planner
        step = self._step
        options = planner._options
        timestamp = step * options.observation_interval_s
        refreshed = self._reference < 0 or step - self._reference >= options.reference_interval_steps
        reference = step if refreshed else self._reference
        history = deepcopy(self._history)
        history.observe(step, position, rotation)
        indices, roles = history.select(reference, step)
        frames = {**self._frames, step: image.copy()}
        route, window = route_condition_input(self._route, position, rotation, planner._route_config)
        encoded = planner._processor(images=[frames[i] for i in indices], return_tensors="pt")
        frame_times = [i * options.observation_interval_s - timestamp for i in indices]
        batch = dict(pixel_values=encoded["pixel_values"], image_grid_thw=encoded["image_grid_thw"],
                     image_ids=[list(range(len(indices)))], frame_times=[np.asarray(frame_times, np.float32)],
                     frame_roles=[roles])
        model = planner._model
        conditions = model.encode_observations(batch)
        device = next(model.head.parameters()).device
        attention = nullcontext()
        if planner._backend == "math":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            attention = sdpa_kernel(SDPBackend.MATH)
        with attention:
            output = model.head(torch.from_numpy(route[None]).to(device), **conditions)
        prediction = LocalPrediction(
            output["xyz"][0].float().cpu().numpy(),
            output["action_logits"][0].float().softmax(-1).cpu().numpy(),
            output["arrival_logits"][0].float().sigmoid().cpu().numpy(),
            self._route, route, dict(position=position.tolist(), rotation_xyzw=rotation.tolist()),
            dict(reference_step=reference, reference_refreshed=refreshed, observed_frame_indices=indices,
                 timestamp_s=timestamp, history=history.diagnostics(), route_input_window=window,
                 observed_frame_times_s=frame_times))
        decision = self._controller.act(prediction)
        keep = set(history.indices) | {reference, step}
        self._frames = {i: frames[i] for i in keep}
        self._step, self._reference, self._history = step + 1, reference, history
        return LocalPlannerStep(prediction, decision)
