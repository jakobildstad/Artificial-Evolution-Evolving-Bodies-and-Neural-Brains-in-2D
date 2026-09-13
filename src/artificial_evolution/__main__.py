"""Desktop and reproducible headless entry points."""

import argparse
import json
from pathlib import Path

from .ecology import DEFAULT_PLANTS
from .persistence import load, save
from .simulation import Simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolving bodies and neural brains in 2D")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--plants", type=int, default=DEFAULT_PLANTS)
    parser.add_argument(
        "--brain",
        choices=("pretrained", "random"),
        default="pretrained",
        help="Ancestral brain for a new world (default: pretrained)",
    )
    parser.add_argument("--max-population", type=int, default=180)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=3600, help="Fixed steps in a headless run")
    parser.add_argument("--load", type=Path, help="Resume a JSON snapshot")
    parser.add_argument("--save", type=Path, help="Save final state on exit")
    parser.add_argument(
        "--save-path", type=Path, default=Path("saves/world.json"), help="Desktop S/L snapshot path"
    )
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    try:
        sim = (
            load(args.load)
            if args.load
            else Simulation(
                args.seed, args.population, args.plants, args.max_population, args.brain
            )
        )
        if args.headless:
            for _ in range(args.steps):
                sim.step()
        else:
            from .rendering import run

            sim = run(sim, args.save_path)
        if args.save:
            save(sim, args.save)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(sim.stats(), indent=2))


if __name__ == "__main__":
    main()
