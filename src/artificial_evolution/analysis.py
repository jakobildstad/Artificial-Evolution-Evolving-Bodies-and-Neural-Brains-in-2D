"""Descriptive genetic groups, bounded population history, and portable CSV reports."""

import colorsys
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .ecology import FOOD_TYPES
from .genome import Genome

if TYPE_CHECKING:
    from .simulation import Simulation

MAX_SAMPLES = 6000


@dataclass
class Group:
    id: int
    born: float
    body: list[float]
    brain: list[float]


def body_features(genome: Genome) -> np.ndarray:
    widths = [s.width for s in genome.segments]
    return np.array(
        [
            np.log(genome.area / 220),
            (len(genome.segments) - 2) * 0.5,
            np.log(sum(s.length for s in genome.segments) / np.mean(widths) / (24 / 9)),
            np.mean([abs(s.angle) for s in genome.segments]),
            (genome.segments[0].width - 10) / 10,
        ]
    )


def classify(genome: Genome, groups: dict[int, Group], time: float) -> int:
    """Use fixed prototypes so group IDs keep their meaning across deaths and saves."""
    body = body_features(genome)
    brain = np.concatenate([a.ravel() for a in genome.arrays()])
    distances = {
        key: float(
            np.linalg.norm(body - group.body) + 0.5 * np.sqrt(np.mean((brain - group.brain) ** 2))
        )
        for key, group in groups.items()
    }
    if distances:
        nearest = min(distances, key=distances.get)
        if distances[nearest] <= 0.8:
            return nearest
    key = max(groups, default=0) + 1
    groups[key] = Group(key, time, body.tolist(), brain.tolist())
    return key


def group_color(group_id: int) -> tuple[int, int, int]:
    """Pond greens, amber and turquoise; these labels have no mechanical effect."""
    hue = (0.22 + (group_id - 1) * 0.073) % 0.5 + 0.06
    rgb = colorsys.hsv_to_rgb(hue, 0.42 + (group_id % 3) * 0.08, 0.83)
    return tuple(int(v * 255) for v in rgb)


def sample(sim: "Simulation") -> dict:
    creatures = list(sim.creatures.values())
    counts = Counter(c.group_id for c in creatures)
    return {
        "seconds": sim.time,
        "population": len(creatures),
        "births": sim.births,
        "deaths": sim.deaths,
        "mean_area": float(np.mean([c.area for c in creatures])) if creatures else 0.0,
        "mean_segments": float(np.mean([len(c.genome.segments) for c in creatures]))
        if creatures
        else 0.0,
        "mean_energy": float(np.mean([c.energy for c in creatures])) if creatures else 0.0,
        "available_food": int(sum(f.renewable and f.energy > 0.5 for f in sim.food)),
        **{
            f"food_{kind}": int(
                sum(f.renewable and f.kind == kind and f.energy > 0.5 for f in sim.food)
            )
            for kind in FOOD_TYPES
        },
        "groups": dict(counts),
    }


def record(sim: "Simulation") -> None:
    row = sample(sim)
    if sim.history and sim.history[-1]["seconds"] == sim.time:
        sim.history[-1] = row
    else:
        sim.history.append(row)
    if len(sim.history) > MAX_SAMPLES:
        # Keep the start and newest point; plots use actual timestamps after thinning.
        sim.history = sim.history[::2]


def group_rows(sim: "Simulation") -> list[dict]:
    rows = []
    for key, group in sim.groups.items():
        living = [c for c in sim.creatures.values() if c.group_id == key]
        births = sum(a.get("group_id") == key for a in sim.ancestry.values())
        plants = sum(c.plant_eaten for c in living)
        meat = sum(c.meat_eaten for c in living)
        diet = {kind: sum(c.food_eaten.get(kind, 0) for c in living) for kind in FOOD_TYPES}
        total = plants + meat
        main_food = max(diet, key=diet.get) if any(diet.values()) else "unknown"
        if meat > max(diet.values()):
            main_food = "meat"
        rows.append(
            {
                "group": key,
                "first_seen": group.born,
                "alive": len(living),
                "total_born": births,
                "mean_area": float(np.mean([c.area for c in living])) if living else 0.0,
                "mean_segments": float(np.mean([len(c.genome.segments) for c in living]))
                if living
                else 0.0,
                "mean_energy": float(np.mean([c.energy for c in living])) if living else 0.0,
                "meat_fraction": meat / (plants + meat) if plants + meat else 0.0,
                "main_food": main_food,
                **{
                    f"{kind}_fraction": amount / total if total else 0
                    for kind, amount in diet.items()
                },
            }
        )
    return sorted(rows, key=lambda r: (-r["alive"], r["group"]))


def export_csv(sim: "Simulation", path: Path) -> None:
    """Write history and group summaries beside a named JSON snapshot."""
    rows = list(sim.history)
    current = sample(sim)
    if not rows or rows[-1]["seconds"] != sim.time:
        rows.append(current)
    fields = [key for key in current if key != "groups"]
    group_fields = [f"group_{key:03d}" for key in sim.groups]
    with path.with_suffix(".population.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields + group_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fields},
                    **{f"group_{key:03d}": row["groups"].get(key, 0) for key in sim.groups},
                }
            )
    summaries = group_rows(sim)
    with path.with_suffix(".groups.csv").open("w", newline="", encoding="utf-8") as file:
        fields = [
            "group",
            "first_seen",
            "alive",
            "total_born",
            "mean_area",
            "mean_segments",
            "mean_energy",
            "meat_fraction",
            "main_food",
            *(f"{kind}_fraction" for kind in FOOD_TYPES),
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
