"""Fixed-step ecosystem, reproduction, and ancestry records."""

import numpy as np
import pymunk

from .creature import Creature
from .ecology import (
    BITE_REACH,
    DT,
    FOOD_CAPACITY,
    HEIGHT,
    REGROWTH,
    WIDTH,
    Food,
    consume,
    make_body,
    make_space,
    sense,
)
from .genome import Genome


class Simulation:
    def __init__(
        self, seed: int = 1, population: int = 24, plants: int = 260, max_population: int = 180
    ):
        if not 0 <= population <= max_population or max_population < 1 or plants < 0:
            raise ValueError("Require 0 <= population <= max_population and plants >= 0")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.space = make_space()
        self.creatures: dict[int, Creature] = {}
        self.food: list[Food] = []
        self.ancestry: dict[int, dict] = {}
        self.max_population = max_population
        self.tick = 0
        self.next_id = 1
        self.births = 0
        self.deaths = 0
        self.energy_input = 0.0
        self.dissipated = 0.0
        self.ancestor = Genome.ancestral(self.rng)
        for _ in range(population):
            for attempt in range(1000):
                position = (
                    float(self.rng.uniform(40, WIDTH - 40)),
                    float(self.rng.uniform(40, HEIGHT - 40)),
                )
                if self.clear_position(position, 28):
                    break
            else:
                raise ValueError("Initial population does not fit in the arena")
            self.add_creature(
                self.ancestor.copy(), position, float(self.rng.uniform(-np.pi, np.pi)), 95.0
            )
        self.food = [
            Food(
                float(self.rng.uniform(12, WIDTH - 12)),
                float(self.rng.uniform(12, HEIGHT - 12)),
                FOOD_CAPACITY,
            )
            for _ in range(plants)
        ]
        self.initial_energy = self.total_energy()

    @property
    def time(self) -> float:
        return self.tick * DT

    def add_creature(
        self,
        genome: Genome,
        position: tuple[float, float],
        angle: float,
        energy: float,
        parent: Creature | None = None,
    ) -> Creature:
        body, shapes = make_body(self.space, genome, position, angle)
        creature = Creature(
            self.next_id,
            parent.id if parent else None,
            parent.generation + 1 if parent else 0,
            genome,
            body,
            shapes,
            energy,
            genome.tissue,
        )
        self.creatures[creature.id] = creature
        self.ancestry[creature.id] = {
            "parent_id": creature.parent_id,
            "generation": creature.generation,
            "born": self.time,
            "died": None,
        }
        self.next_id += 1
        return creature

    def clear_position(self, position: tuple[float, float], radius: float) -> bool:
        x, y = position
        return (
            radius + 5 < x < WIDTH - radius - 5
            and radius + 5 < y < HEIGHT - radius - 5
            and all(
                (c.body.position - position).length > radius + c.radius + 4
                for c in self.creatures.values()
            )
        )

    def reproduce(self, parent: Creature) -> Creature | None:
        """Pay for new tissue and reserves from the parent's existing energy."""
        if (
            len(self.creatures) >= self.max_population
            or parent.cooldown > 0
            or parent.age < 8
            or parent.energy < parent.capacity * 0.8
            or parent.tissue < parent.genome.tissue * 0.7
        ):
            return None
        genome = parent.genome.mutated(self.rng)
        reserve = genome.tissue * 1.6
        cost = genome.tissue + reserve
        if parent.energy - cost < parent.capacity * 0.2:
            return None
        radius = max(np.hypot(*v) for polygon in genome.polygons() for v in polygon)
        for _ in range(8):
            direction = float(self.rng.uniform(-np.pi, np.pi))
            position = parent.body.position + pymunk.Vec2d(parent.radius + radius + 8, 0).rotated(
                direction
            )
            if self.clear_position(tuple(position), radius):
                child = self.add_creature(genome, tuple(position), direction, reserve, parent)
                parent.energy -= cost
                parent.cooldown = 8.0
                self.births += 1
                return child
        return None

    def feed(self, eater: Creature, dt: float, nearby_food: list[Food] | None = None) -> None:
        """One mouth and one bite budget for every creature, regardless of ancestry."""
        budget = max(0.0, float(eater.actions[2])) * 40 * dt
        if budget <= 0 or eater.energy <= 0 or eater.tissue <= 0:
            return
        mouth = eater.mouth
        # Food gets the same digestion efficiency as tissue; no dietary classes exist.
        for food in self.food if nearby_food is None else nearby_food:
            if budget <= 0:
                break
            if (mouth - (food.x, food.y)).length <= BITE_REACH + 4:
                removed, loss = consume(eater, food.energy, budget)
                food.energy -= removed
                budget -= removed
                self.dissipated += loss
        if budget <= 0:
            return
        for victim in self.creatures.values():
            if victim is eater or victim.tissue <= 0:
                continue
            if (mouth - victim.body.position).length > victim.radius + BITE_REACH:
                continue
            if any(shape.point_query(mouth).distance <= BITE_REACH for shape in victim.shapes):
                removed, loss = consume(eater, victim.tissue, budget)
                victim.tissue -= removed
                budget -= removed
                self.dissipated += loss
                if budget <= 0:
                    break

    def kill(self, creature: Creature) -> None:
        """Remaining tissue and reserves become nonrenewable carrion."""
        self.food.append(
            Food(
                creature.body.position.x,
                creature.body.position.y,
                creature.energy + creature.tissue,
                False,
            )
        )
        self.space.remove(*creature.shapes, creature.body)
        del self.creatures[creature.id]
        self.ancestry[creature.id]["died"] = self.time
        self.deaths += 1

    def step(self) -> None:
        residents = list(self.creatures.values())
        if self.tick % 6 == 0:
            food_positions = np.array([(f.x, f.y) for f in self.food if f.energy > 1])
            positions = np.array([tuple(c.body.position) for c in residents]).reshape(-1, 2)
            for index, creature in enumerate(residents):
                creature.think(sense(creature, food_positions, np.delete(positions, index, axis=0)))
        for creature in residents:
            thrust, turn, bite, _ = creature.actions
            creature.age += DT
            creature.cooldown = max(0.0, creature.cooldown - DT)
            width = creature.genome.segments[0].width / 10
            creature.body.apply_force_at_local_point((float(thrust) * 85 * width, 0), (0, 0))
            creature.body.torque += float(turn) * 400 * width
            # Larger bodies pay upkeep and actuation costs, and have greater inertia.
            cost = DT * (
                0.3
                + creature.genome.area * 0.002
                + creature.body.mass * (abs(thrust) * 0.2 + abs(turn) * 0.1)
                + max(0, bite) * 0.12
            )
            spent = min(creature.energy, cost)
            creature.energy -= spent
            self.dissipated += spent
        self.space.step(DT)
        # Rotate feeding priority to avoid a persistent advantage for older IDs.
        if residents:
            offset = self.tick % len(residents)
            feeders = residents[offset:] + residents[:offset]
            # NumPy handles the many point distances; Pymunk handles actual body geometry.
            mouths = np.array([tuple(c.mouth) for c in feeders])
            food_points = np.array([(f.x, f.y) for f in self.food]).reshape(-1, 2)
            distances_squared = np.sum((mouths[:, None, :] - food_points[None, :, :]) ** 2, axis=2)
            for creature, distances in zip(feeders, distances_squared):
                nearby = np.flatnonzero(distances <= (BITE_REACH + 4) ** 2)
                self.feed(creature, DT, [self.food[i] for i in nearby])
        self.tick += 1
        for creature in residents:
            if (
                creature.energy <= 0
                or creature.tissue < creature.genome.tissue * 0.2
                or creature.age >= 240
            ):
                self.kill(creature)
            elif self.tick % 30 == 0 and creature.actions[3] > 0:
                self.reproduce(creature)
        for food in self.food:
            if food.renewable:
                growth = min(FOOD_CAPACITY - food.energy, REGROWTH * DT)
                food.energy += growth
                self.energy_input += growth
            else:
                decay = min(food.energy, food.energy * 0.008 * DT)
                food.energy -= decay
                self.dissipated += decay
        remaining = []
        for food in self.food:
            if food.renewable or food.energy > 0.01:
                remaining.append(food)
            else:
                self.dissipated += food.energy
        self.food = remaining

    def total_energy(self) -> float:
        return sum(c.energy + c.tissue for c in self.creatures.values()) + sum(
            f.energy for f in self.food
        )

    def stats(self) -> dict:
        residents = list(self.creatures.values())
        return {
            "seed": self.seed,
            "tick": self.tick,
            "seconds": round(self.time, 2),
            "population": len(residents),
            "births": self.births,
            "deaths": self.deaths,
            "max_generation": max((a["generation"] for a in self.ancestry.values()), default=0),
            "mean_segments": round(float(np.mean([len(c.genome.segments) for c in residents])), 2)
            if residents
            else 0,
            "mean_energy": round(float(np.mean([c.energy for c in residents])), 2)
            if residents
            else 0,
            "energy_balance_error": self.total_energy()
            + self.dissipated
            - self.initial_energy
            - self.energy_input,
        }
