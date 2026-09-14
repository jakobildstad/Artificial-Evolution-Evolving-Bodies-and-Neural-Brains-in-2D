"""A creature's mutable lifetime state and inherited recurrent controller."""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pymunk

from .genome import INPUTS, MEMORY, Genome


@dataclass
class Creature:
    id: int
    parent_id: int | None
    generation: int
    genome: Genome
    body: pymunk.Body
    shapes: list[pymunk.Poly]
    energy: float
    tissue: float
    age: float = 0.0
    cooldown: float = 0.0
    memory: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY))
    actions: np.ndarray = field(default_factory=lambda: np.zeros(4))
    sensors: np.ndarray = field(default_factory=lambda: np.zeros(INPUTS))
    group_id: int = 0
    plant_eaten: float = 0.0
    meat_eaten: float = 0.0
    food_eaten: dict[str, float] = field(default_factory=dict)

    @cached_property
    def capacity(self) -> float:
        # A fixed reserve lets a parent afford offspring that add a whole segment.
        return 40 + self.genome.tissue * 4 + 20 * (len(self.genome.segments) - 1)

    @cached_property
    def area(self) -> float:
        return self.genome.area

    @cached_property
    def strength(self) -> float:
        return (self.area / 220) ** 0.5 * (self.genome.segments[0].width / 10)

    @cached_property
    def armor(self) -> float:
        return sum(s.width for s in self.genome.segments) / len(self.genome.segments) / 9

    @cached_property
    def motor_force(self) -> float:
        return 85 * (self.area / 220) ** 0.85 * (self.genome.segments[0].width / 10) ** 0.5

    @cached_property
    def vertices(self) -> list:
        return [list(shape.get_vertices()) for shape in self.shapes]

    @property
    def mouth(self) -> pymunk.Vec2d:
        return self.body.local_to_world((self.genome.segments[0].length / 2 + 2, 0))

    @cached_property
    def radius(self) -> float:
        return max(v.length for shape in self.shapes for v in shape.get_vertices())

    def think(self, inputs: np.ndarray) -> None:
        self.sensors = inputs.copy()
        genome = self.genome
        self.memory = np.tanh(
            genome.input_weights @ inputs
            + genome.recurrent_weights @ self.memory
            + genome.hidden_bias
        )
        self.actions = np.tanh(genome.output_weights @ self.memory + genome.output_bias)
