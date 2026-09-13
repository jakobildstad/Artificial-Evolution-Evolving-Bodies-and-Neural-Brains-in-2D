"""Offline imitation of basic foraging examples using a small NumPy training loop."""

import argparse
import json
from pathlib import Path

import numpy as np

from .genome import INPUTS, MEMORY, Genome


def examples(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic sensor states and a smooth steering teacher, used only offline."""
    inputs = rng.uniform(-1, 1, (count, INPUTS))
    angle = rng.uniform(-np.pi, np.pi, count)
    proximity = rng.uniform(0, 1, count)
    proximity[rng.random(count) < 0.15] = 0
    inputs[:, 0] = np.cos(angle) * proximity
    inputs[:, 1] = np.sin(angle) * proximity
    inputs[:, 4:8] = rng.uniform(0, 1, (count, 4)) ** 6
    inputs[:, 8:10] = rng.uniform(0.1, 1, (count, 2))
    inputs[:, 13:15] = rng.uniform(0, 1, (count, 2))
    inputs[:, 15] = 1
    forward, sideways = inputs[:, 0], inputs[:, 1]
    front_wall, right_wall, left_wall = inputs[:, 4], inputs[:, 5], inputs[:, 7]
    turn = np.tanh(3.5 * sideways / (proximity + 0.3) + 3 * front_wall * (left_wall - right_wall))
    thrust = 0.6 * (0.2 + 0.8 * np.maximum(forward / (proximity + 0.1), 0))
    thrust[proximity == 0] = 0.45
    thrust *= (1 - 0.7 * proximity**6) * (1 - 0.8 * front_wall)
    targets = np.column_stack((thrust, turn, np.full(count, 0.9), np.full(count, 0.8)))
    # Vary prior memory so inherited memory cannot destabilize basic steering.
    memory = rng.normal(0, 0.4, (count, MEMORY))
    return inputs, memory, targets


def train(seed: int = 42, steps: int = 3000) -> tuple[Genome, dict]:
    """Fit all five weight arrays with analytic gradients and Adam updates."""
    rng = np.random.default_rng(seed)
    genome = Genome.ancestral(rng)
    arrays = genome.arrays()
    first_moments = [np.zeros_like(a) for a in arrays]
    second_moments = [np.zeros_like(a) for a in arrays]
    for step in range(1, steps + 1):
        inputs, memory, targets = examples(rng, 512)
        hidden = np.tanh(
            inputs @ genome.input_weights.T
            + memory @ genome.recurrent_weights.T
            + genome.hidden_bias
        )
        outputs = np.tanh(hidden @ genome.output_weights.T + genome.output_bias)
        output_gradient = 2 * (outputs - targets) * (1 - outputs**2) / outputs.size
        hidden_gradient = (output_gradient @ genome.output_weights) * (1 - hidden**2)
        gradients = (
            hidden_gradient.T @ inputs,
            hidden_gradient.T @ memory,
            output_gradient.T @ hidden,
            hidden_gradient.sum(axis=0),
            output_gradient.sum(axis=0),
        )
        for array, gradient, first, second in zip(arrays, gradients, first_moments, second_moments):
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient**2
            array -= (
                0.003 * (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            )
            np.clip(array, -4, 4, out=array)
    # Evaluation examples come from a separate, held-out random stream.
    inputs, memory, targets = examples(np.random.default_rng(seed + 1), 4096)
    hidden = np.tanh(
        inputs @ genome.input_weights.T + memory @ genome.recurrent_weights.T + genome.hidden_bias
    )
    outputs = np.tanh(hidden @ genome.output_weights.T + genome.output_bias)
    return genome, {
        "method": "offline synthetic foraging imitation",
        "seed": seed,
        "steps": steps,
        "batch_size": 512,
        "validation_mse": float(np.mean((outputs - targets) ** 2)),
        "limitations": "A foraging prior, not a guarantee of ecosystem survival.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--output", type=Path, default=Path("src/artificial_evolution/data/ancestor.npz")
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    genome, report = train(args.seed, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        np.savez_compressed(file, **{f"weights_{i}": a for i, a in enumerate(genome.arrays())})
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
