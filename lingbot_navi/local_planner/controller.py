"""Execute one current-ego local slot, then observe and replan."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from collections.abc import Mapping

STOP, MOVE_FORWARD = 0, 1


@dataclass(frozen=True)
class ControllerDecision:
    action: int
    diagnostics: Mapping[str, Any]

# Habitat action IDs. Local head class IDs are a different namespace: 0/1/2.
TURN_LEFT = 2
TURN_RIGHT = 3


@dataclass(frozen=True)
class LocalPrediction:
    local_trajectory_ego: np.ndarray
    action_probabilities: np.ndarray
    arrival_probabilities: np.ndarray
    global_trajectory_world: np.ndarray
    global_trajectory_ego: np.ndarray
    current_pose_world: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        action_shape = np.asarray(self.action_probabilities).shape
        if action_shape != (10, 3):
            raise ValueError("action probabilities require [10,3]")
        shapes = {
            "local_trajectory_ego": (10, 3),
            "action_probabilities": action_shape,
            "arrival_probabilities": (10,),
        }
        for name in ("global_trajectory_world", "global_trajectory_ego"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != 3 or len(array) < 2:
                raise ValueError(f"{name} requires [N>=2,3]")
            shapes[name] = array.shape
        for name, shape in shapes.items():
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, array)
        for array in (self.action_probabilities, self.arrival_probabilities):
            if ((array < 0) | (array > 1)).any():
                raise ValueError("prediction probabilities must be in [0,1]")
        if not np.allclose(self.action_probabilities.sum(-1), 1., atol=1e-5):
            raise ValueError("action probabilities must sum to one")

    def to_dict(self):
        return {
            "trajectory_source": "local",
            **{name: getattr(self, name).tolist() for name in (
                "local_trajectory_ego", "action_probabilities", "arrival_probabilities",
                "global_trajectory_world", "global_trajectory_ego",
            )},
            "current_pose_world": self.current_pose_world,
            "diagnostics": self.diagnostics,
        }


class LocalTrajectoryController:
    def __init__(self, options):
        self.options = options

    def act(self, prediction: LocalPrediction) -> ControllerDecision:
        target = prediction.local_trajectory_ego[0, :2]
        distance = float(np.linalg.norm(target))
        arrival = float(prediction.arrival_probabilities[0])
        local_action = int(np.argmax(prediction.action_probabilities[0]))
        path = np.vstack((np.zeros((1, 3)), prediction.local_trajectory_ego))
        base = {
            **prediction.diagnostics,
            "trajectory_source": "local", "executed_slot": 0,
            "local_action_class": local_action,
            "arrival_probability": arrival,
            "arrival_threshold": self.options.arrival_threshold,
            "arrival_distance_m": self.options.arrival_distance_m,
            "target_ego_xy": target.tolist(),
            "target_distance_xy_m": distance,
            "path_length_xy_m": float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()),
            "goal_straight_distance_xy_m": float(np.linalg.norm(prediction.global_trajectory_ego[-1, :2])),
            "requested_path_distance_m": self.options.trajectory_step_m,
            "selected_path_distance_m": 0.0,
            "move_distance_m": 0.0,
            "target_heading_deg": 0.0,
        }
        # Arrival refers to the position after the first predicted action.
        if arrival >= self.options.arrival_threshold and distance <= self.options.arrival_distance_m:
            return ControllerDecision(STOP, {**base, "reason": "local_arrival"})
        if local_action in (0, 1):
            angle = self.options.turn_angle_deg * (1 if local_action == 0 else -1)
            return ControllerDecision(TURN_LEFT if local_action == 0 else TURN_RIGHT, {
                **base, "reason": "local_turn", "target_heading_deg": angle,
            })
        move = min(distance, self.options.trajectory_step_m)
        selected = target * (move / distance) if distance > 0 else target
        return ControllerDecision(MOVE_FORWARD, {
            **base, "reason": "local_forward", "move_distance_m": move,
            "selected_path_distance_m": move,
            "target_ego_xy": selected.tolist(),
            "target_heading_deg": math.degrees(math.atan2(float(target[1]), float(target[0]))) if distance else 0.,
        })
