"""Heritable geometry and fixed-size recurrent neural weights."""

from dataclasses import dataclass

import numpy as np

INPUTS, MEMORY, ACTIONS = 16, 12, 4
MAX_SEGMENTS = 5


@dataclass
class Segment:
    length: float
    width: float
    angle: float = 0.0


@dataclass
class Genome:
    segments: list[Segment]
    input_weights: np.ndarray
    recurrent_weights: np.ndarray
    output_weights: np.ndarray
    hidden_bias: np.ndarray
    output_bias: np.ndarray

    @classmethod
    def ancestral(cls, rng: np.random.Generator) -> "Genome":
        """One random brain shared by all founders; biases encourage basic activity."""
        return cls(
            [Segment(14, 10), Segment(10, 8)],
            rng.normal(0, 0.3, (MEMORY, INPUTS)),
            rng.normal(0, 0.18, (MEMORY, MEMORY)),
            rng.normal(0, 0.12, (ACTIONS, MEMORY)),
            np.zeros(MEMORY),
            np.array([0.5, 0.0, 0.8, 0.8]),
        )

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.input_weights,
            self.recurrent_weights,
            self.output_weights,
            self.hidden_bias,
            self.output_bias,
        )

    def copy(self) -> "Genome":
        return Genome(
            [Segment(s.length, s.width, s.angle) for s in self.segments],
            *(a.copy() for a in self.arrays()),
        )

    def mutated(self, rng: np.random.Generator) -> "Genome":
        """Sparse weight noise plus bounded dimension and segment-count mutations."""
        child = self.copy()
        for array in child.arrays():
            mask = rng.random(array.shape) < 0.08
            array += mask * rng.normal(0, 0.15, array.shape)
            np.clip(array, -4, 4, out=array)
        for segment in child.segments:
            if rng.random() < 0.3:
                segment.length = float(np.clip(segment.length + rng.normal(0, 2), 6, 22))
                segment.width = float(np.clip(segment.width + rng.normal(0, 1.5), 4, 16))
                segment.angle = float(np.clip(segment.angle + rng.normal(0, 0.2), -0.8, 0.8))
        operation = rng.random()
        if operation < 0.06 and len(child.segments) > 1:
            child.segments.pop(int(rng.integers(1, len(child.segments))))
        elif operation < 0.12 and len(child.segments) < MAX_SEGMENTS:
            source = child.segments[int(rng.integers(len(child.segments)))]
            child.segments.append(Segment(source.length, source.width, source.angle))
        elif operation < 0.18 and len(child.segments) < MAX_SEGMENTS:
            child.segments.append(
                Segment(
                    float(rng.uniform(6, 18)),
                    float(rng.uniform(4, 12)),
                    float(rng.uniform(-0.8, 0.8)),
                )
            )
        # The first segment defines the shared sensor/motor frame.
        child.segments[0].angle = 0.0
        return child

    @property
    def area(self) -> float:
        return sum(s.length * s.width for s in self.segments)

    @property
    def tissue(self) -> float:
        return self.area * 0.12

    def polygons(self) -> list[list[tuple[float, float]]]:
        """A connected chain extending behind the first segment, all on one rigid body."""
        polygons = []
        start = np.array([self.segments[0].length / 2, 0.0])
        angle = 0.0
        for segment in self.segments:
            angle += segment.angle
            along = np.array([np.cos(angle), np.sin(angle)])
            across = np.array([-along[1], along[0]]) * segment.width / 2
            end = start - along * segment.length
            polygons.append(
                [tuple(v) for v in (start + across, start - across, end - across, end + across)]
            )
            start = end + along * 1.0  # Slight overlap keeps bent segments connected.
        return polygons
