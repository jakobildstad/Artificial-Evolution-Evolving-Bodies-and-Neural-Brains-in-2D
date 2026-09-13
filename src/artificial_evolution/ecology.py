"""Pymunk construction, body-relative sensing, and energy transfers."""

from dataclasses import dataclass

import numpy as np
import pymunk

from .creature import Creature
from .genome import Genome

WIDTH, HEIGHT = 1000, 700
DT = 1 / 60
SIGHT = 180.0
FOOD_CAPACITY = 22.0
DEFAULT_PLANTS = 90
FOOD_RESPAWN_DELAY = (30.0, 60.0)
BITE_REACH = 7.0


@dataclass
class Food:
    x: float
    y: float
    energy: float
    renewable: bool = True
    respawn_in: float = 0.0


def make_space() -> pymunk.Space:
    space = pymunk.Space()
    space.damping = 0.3
    corners = [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        wall = pymunk.Segment(space.static_body, a, b, 4)
        wall.friction = 0.6
        space.add(wall)
    return space


def make_body(
    space: pymunk.Space, genome: Genome, position: tuple[float, float], angle: float
) -> tuple[pymunk.Body, list[pymunk.Poly]]:
    body = pymunk.Body()
    body.position = position
    body.angle = angle
    shapes = [pymunk.Poly(body, vertices) for vertices in genome.polygons()]
    for shape in shapes:
        # Pymunk derives mass, center of gravity, and inertia from the actual geometry.
        shape.density = 0.01
        shape.friction = 0.5
        shape.elasticity = 0.1
    space.add(body, *shapes)
    return body, shapes


def relative_target(creature: Creature, positions: np.ndarray) -> tuple[float, float]:
    if len(positions) == 0:
        return 0.0, 0.0
    offsets = positions - np.array(creature.mouth)
    distances = np.linalg.norm(offsets, axis=1)
    index = int(np.argmin(distances))
    distance = float(distances[index])
    if distance > SIGHT:
        return 0.0, 0.0
    direction = pymunk.Vec2d(*offsets[index]).rotated(-creature.body.angle)
    signal = direction / max(distance, 1) * (1 - distance / SIGHT)
    return signal.x, signal.y


def sense(
    creature: Creature, food_positions: np.ndarray, creature_positions: np.ndarray
) -> np.ndarray:
    """All directional channels and motors stay in the first segment's frame."""
    body = creature.body
    velocity = body.velocity.rotated(-body.angle)
    wall_distances = []
    for offset in (0, np.pi / 2, np.pi, -np.pi / 2):
        direction = pymunk.Vec2d(1, 0).rotated(body.angle + offset)
        distances = [SIGHT]
        for position, component, bound in (
            (body.position.x, direction.x, WIDTH),
            (body.position.y, direction.y, HEIGHT),
        ):
            if abs(component) > 1e-8:
                distances.append(((bound if component > 0 else 0) - position) / component)
        wall_distances.append(1 - np.clip(min(distances), 0, SIGHT) / SIGHT)
    return np.array(
        [
            *relative_target(creature, food_positions),
            *relative_target(creature, creature_positions),
            *wall_distances,
            creature.energy / creature.capacity,
            creature.tissue / creature.genome.tissue,
            np.tanh(velocity.x / 60),
            np.tanh(velocity.y / 60),
            np.tanh(body.angular_velocity / 3),
            min(creature.age / 180, 1),
            min(creature.cooldown / 8, 1),
            1.0,
        ]
    )


def consume(eater: Creature, available: float, amount: float) -> tuple[float, float]:
    """Return removed energy and digestion loss, respecting stomach capacity."""
    removed = max(0.0, min(available, amount, (eater.capacity - eater.energy) / 0.8))
    eater.energy += removed * 0.8
    return removed, removed * 0.2
