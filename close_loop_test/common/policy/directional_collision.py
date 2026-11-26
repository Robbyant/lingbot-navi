"""Directional contact evidence, separate from measured depth occupancy.

All positions and unit approach vectors use the (Habitat X, -Z) plane.
A contact constrains translation into it from a matching heading; it never
paints a cell FREE or OCCUPIED and does not constrain rotation or retreat.
"""
from dataclasses import dataclass
import math
import numpy as np
DEFAULT_HALF_ANGLE_DEG = 30.0

@dataclass(eq=False)
class DirectionalCollision:
    position: np.ndarray
    approach: np.ndarray
    radius: float
    half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG

    @classmethod
    def at_stopped_edge(cls, stopped_xy, approach, body_radius, patch_radius):
        xy = np.asarray(stopped_xy, dtype=float).reshape(2)
        vector = np.asarray(approach, dtype=float).reshape(2)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(xy).all() or not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError('collision position and approach must be finite; approach must be nonzero')
        vector = vector / norm
        return cls(xy + float(body_radius) * vector, vector, float(patch_radius))

    def blocks_segment(self, start, end, body_radius):
        (start, end) = (np.asarray(start, dtype=float), np.asarray(end, dtype=float))
        delta = end - start
        squared = float(delta @ delta)
        if squared <= 1e-18:
            return False
        heading = delta / math.sqrt(squared)
        if float(heading @ self.approach) + 1e-09 < math.cos(math.radians(self.half_angle_deg)):
            return False
        offset = self.position - start
        if float(offset @ delta) <= 1e-12:
            return False
        fraction = float(np.clip(offset @ delta / squared, 0.0, 1.0))
        distance = float(np.linalg.norm(self.position - (start + fraction * delta)))
        return distance <= float(body_radius) + self.radius + 1e-09


def remember_contact(contacts, stopped_xy, heading, body_radius, patch_radius):
    contact = DirectionalCollision.at_stopped_edge(stopped_xy, heading, body_radius, patch_radius)
    for old in contacts:
        if np.linalg.norm(old.position - contact.position) < max(patch_radius, 1e-06) and float(old.approach @ contact.approach) > 1.0 - 1e-06:
            return old
    contacts.append(contact)
    return contact
