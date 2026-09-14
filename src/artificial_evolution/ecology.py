"""Pymunk construction, body-relative sensing, and energy transfers."""

from dataclasses import dataclass

import numpy as np
import pymunk

from .creature import Creature
from .genome import Genome
from .terrain import CREATURE, TERRAIN, TERRAIN_FILTER, WORLD_HEIGHT, WORLD_WIDTH, visible

WIDTH, HEIGHT = WORLD_WIDTH, WORLD_HEIGHT
DT = 1 / 60
SIGHT = 180.0
DEFAULT_PLANTS = 220
BITE_REACH = 7.0
PARENT_PROTECTION_TIME = 10.0


@dataclass(frozen=True)
class FoodType:
    label: str
    energy: float
    hardness: float
    respawn: tuple[float, float]
    color: tuple[int, int, int]


FOOD_TYPES = {
    "algae": FoodType("Soft algae", 10, 0.05, (18, 32), (140, 185, 94)),
    "seed": FoodType("Armored seeds", 42, 1.0, (55, 85), (212, 171, 83)),
    "weed": FoodType("Fibrous waterweed", 30, 0.25, (35, 55), (77, 155, 115)),
    "plankton": FoodType("Drifting plankton", 16, 0, (20, 40), (136, 211, 206)),
}


@dataclass
class Food:
    x: float
    y: float
    energy: float
    renewable: bool = True
    respawn_in: float = 0.0
    hardness: float = 0.0
    kind: str = "algae"
    drift_angle: float = 0.0


def food_resistance(food: Food, eater: Creature) -> float:
    resistance = 1 + 5 * food.hardness**2 / eater.strength
    if food.kind == "weed":
        # More segments provide more processing capacity for fibrous food.
        resistance += 5 / len(eater.genome.segments) ** 1.5
    return resistance


def make_space(width: int = WIDTH, height: int = HEIGHT) -> pymunk.Space:
    space = pymunk.Space()
    space.damping = 0.3
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        wall = pymunk.Segment(space.static_body, a, b, 4)
        wall.friction = 0.6
        wall.filter = pymunk.ShapeFilter(categories=TERRAIN)
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
        shape.filter = pymunk.ShapeFilter(categories=CREATURE)
    space.add(body, *shapes)
    return body, shapes


def relative_target(
    creature: Creature, positions: np.ndarray, space: pymunk.Space | None = None
) -> tuple[float, float]:
    if len(positions) == 0:
        return 0.0, 0.0
    offsets = positions - np.array(creature.mouth)
    distances = np.linalg.norm(offsets, axis=1)
    index = int(np.argmin(distances))
    if space is not None:
        index = next(
            (
                int(i)
                for i in np.argsort(distances)
                if distances[i] <= SIGHT and visible(space, creature.mouth, tuple(positions[i]))
            ),
            -1,
        )
        if index < 0:
            return 0.0, 0.0
    distance = float(distances[index])
    if distance > SIGHT:
        return 0.0, 0.0
    direction = pymunk.Vec2d(*offsets[index]).rotated(-creature.body.angle)
    signal = direction / max(distance, 1) * (1 - distance / SIGHT)
    return signal.x, signal.y


def sense(
    creature: Creature,
    food_positions: np.ndarray,
    creature_positions: np.ndarray,
    width: int = WIDTH,
    height: int = HEIGHT,
    space: pymunk.Space | None = None,
) -> np.ndarray:
    """All directional channels and motors stay in the first segment's frame."""
    body = creature.body
    velocity = body.velocity.rotated(-body.angle)
    wall_distances = []
    for offset in (0, np.pi / 2, np.pi, -np.pi / 2):
        direction = pymunk.Vec2d(1, 0).rotated(body.angle + offset)
        distances = [SIGHT]
        for position, component, bound in (
            (body.position.x, direction.x, width),
            (body.position.y, direction.y, height),
        ):
            if abs(component) > 1e-8:
                distances.append(((bound if component > 0 else 0) - position) / component)
        if space is not None:
            hit = space.segment_query_first(
                body.position, body.position + direction * SIGHT, 2, TERRAIN_FILTER
            )
            if hit is not None:
                distances.append(hit.alpha * SIGHT)
        wall_distances.append(1 - np.clip(min(distances), 0, SIGHT) / SIGHT)
    return np.array(
        [
            *relative_target(creature, food_positions, space),
            *relative_target(creature, creature_positions, space),
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


def sense_population(
    creatures: list[Creature],
    food_positions: np.ndarray,
    width: int = WIDTH,
    height: int = HEIGHT,
    space: pymunk.Space | None = None,
) -> np.ndarray:
    """The same 16 inputs as sense(), with distance and wall math batched in NumPy."""
    count = len(creatures)
    inputs = np.zeros((count, 16))
    if not count:
        return inputs
    positions = np.array([tuple(c.body.position) for c in creatures])
    mouths = np.array([tuple(c.mouth) for c in creatures])
    angles = np.array([c.body.angle for c in creatures])
    cosine, sine = np.cos(angles), np.sin(angles)

    def targets(points: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((count, 2))
        offsets = points[None, :, :] - mouths[:, None, :]
        squared = np.sum(offsets**2, axis=2)
        if exclude_self:
            np.fill_diagonal(squared, np.inf)
        nearest = np.argmin(squared, axis=1)
        if space is not None:
            # Search visible targets so food across a bank cannot pin a forager in place.
            for row in range(count):
                candidates = np.flatnonzero(squared[row] <= SIGHT**2)
                candidates = candidates[np.argsort(squared[row, candidates])]
                for index in candidates:
                    if visible(space, tuple(mouths[row]), tuple(points[index])):
                        nearest[row] = index
                        break
                else:
                    squared[row, nearest[row]] = np.inf
        distance = np.sqrt(squared[np.arange(count), nearest])
        delta = offsets[np.arange(count), nearest]
        strength = np.maximum(0, 1 - distance / SIGHT) / np.maximum(distance, 1)
        return (
            np.column_stack(
                (
                    delta[:, 0] * cosine + delta[:, 1] * sine,
                    -delta[:, 0] * sine + delta[:, 1] * cosine,
                )
            )
            * strength[:, None]
        )

    inputs[:, :2] = targets(food_positions)
    inputs[:, 2:4] = targets(positions, True)
    dx = np.column_stack((cosine, -sine, -cosine, sine))
    dy = np.column_stack((sine, cosine, -sine, -cosine))
    tx = np.full_like(dx, SIGHT)
    ty = np.full_like(dy, SIGHT)
    np.divide(
        np.where(dx > 0, width, 0) - positions[:, 0, None], dx, out=tx, where=np.abs(dx) > 1e-8
    )
    np.divide(
        np.where(dy > 0, height, 0) - positions[:, 1, None], dy, out=ty, where=np.abs(dy) > 1e-8
    )
    inputs[:, 4:8] = 1 - np.clip(np.minimum(tx, ty), 0, SIGHT) / SIGHT
    if space is not None:
        for row, creature in enumerate(creatures):
            start = creature.body.position
            for column in range(4):
                end = start + pymunk.Vec2d(dx[row, column], dy[row, column]) * SIGHT
                hit = space.segment_query_first(start, end, 2, TERRAIN_FILTER)
                if hit is not None:
                    inputs[row, 4 + column] = max(inputs[row, 4 + column], 1 - hit.alpha)
    velocity = np.array([tuple(c.body.velocity) for c in creatures])
    inputs[:, 8] = [c.energy / c.capacity for c in creatures]
    inputs[:, 9] = [c.tissue / c.genome.tissue for c in creatures]
    inputs[:, 10] = np.tanh((velocity[:, 0] * cosine + velocity[:, 1] * sine) / 60)
    inputs[:, 11] = np.tanh((-velocity[:, 0] * sine + velocity[:, 1] * cosine) / 60)
    inputs[:, 12] = np.tanh([c.body.angular_velocity / 3 for c in creatures])
    inputs[:, 13] = [min(c.age / 180, 1) for c in creatures]
    inputs[:, 14] = [min(c.cooldown / 8, 1) for c in creatures]
    inputs[:, 15] = 1
    return inputs
