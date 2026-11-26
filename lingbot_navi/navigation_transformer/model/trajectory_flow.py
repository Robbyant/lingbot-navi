"""Conditional Flow Matching in metric first-block coordinates."""

import math

import torch
from torch import nn


class FlowTrajectoryHead(nn.Module):
    """Evolve sequences of 78 interior points conditioned on shared trajectory queries."""

    steps = 16
    scale_m = 5.0

    def __init__(self, context_dim):
        super().__init__()
        width = 256
        self.context = nn.Linear(context_dim, width)
        self.point = nn.Linear(3, width)
        self.endpoints = nn.Linear(9, width)
        self.time = nn.Linear(5, width)
        self.position = nn.Embedding(78, width)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(width, 8, dim_feedforward=4 * width, dropout=0.,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(4)
        )
        self.velocity = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        nn.init.normal_(self.velocity[-1].weight, std=.001)
        nn.init.zeros_(self.velocity[-1].bias)
        self.register_buffer("progress", torch.linspace(0., 1., 80)[1:-1, None], persistent=False)

    def _condition(self, hidden, condition):
        start, goal = condition[:, :3].detach().float()
        delta = goal - start
        chord = start + self.progress * delta
        endpoints = self.endpoints(torch.cat((start, goal, delta)) / self.scale_m)
        context = self.context(hidden.float()).unsqueeze(0)
        return start, goal, chord, endpoints, context

    def _velocity(self, state, time, endpoints, context):
        phase = math.pi * time
        embedding = torch.stack((time, phase.sin(), phase.cos(), (2 * phase).sin(), (2 * phase).cos()), dim=-1)
        tokens = self.point(state) + self.position.weight + endpoints + self.time(embedding)[:, None]
        context = context.expand(state.size(0), -1, -1)
        for layer in self.layers:
            tokens = layer(tokens, context)
        return self.velocity(tokens)

    def _paths(self, state, start, goal, chord):
        samples = state.size(0)
        return torch.cat((start.expand(samples, 1, -1), chord + self.scale_m * state,
                          goal.expand(samples, 1, -1)), dim=1)

    @torch.no_grad()
    def sample(self, hidden, condition):
        """Integrate from evenly spaced points on the start-to-goal chord."""
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            start, goal, chord, endpoints, context = self._condition(hidden, condition)
            # The flow state is the normalized offset from the chord.
            state = torch.zeros_like(chord).unsqueeze(0)
            step_size = 1. / self.steps
            for step in range(self.steps):
                time = state.new_tensor([step * step_size])
                state = state + step_size * self._velocity(state, time, endpoints, context)
            return self._paths(state, start, goal, chord)[0]
