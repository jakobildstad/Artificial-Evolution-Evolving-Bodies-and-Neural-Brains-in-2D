"""A creature's mutable lifetime state and inherited recurrent controller."""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pymunk

from .genome import MEMORY, Genome


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

    @property
    def capacity(self) -> float:
        return self.genome.tissue * 4

    @property
    def mouth(self) -> pymunk.Vec2d:
        return self.body.local_to_world((self.genome.segments[0].length / 2 + 2, 0))

    @cached_property
    def radius(self) -> float:
        return max(v.length for shape in self.shapes for v in shape.get_vertices())

    def think(self, inputs: np.ndarray) -> None:
        genome = self.genome
        self.memory = np.tanh(
            genome.input_weights @ inputs
            + genome.recurrent_weights @ self.memory
            + genome.hidden_bias
        )
        self.actions = np.tanh(genome.output_weights @ self.memory + genome.output_bias)
