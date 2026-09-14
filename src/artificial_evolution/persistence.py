"""Versioned JSON snapshots; never deserialize executable Python objects."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .analysis import Group
from .analysis import record as record_history
from .ecology import FOOD_TYPES, Food
from .genome import ACTIONS, INPUTS, MAX_SEGMENTS, MEMORY, Genome, Segment
from .simulation import Simulation

VERSION = 5
WEIGHTS = ("input_weights", "recurrent_weights", "output_weights", "hidden_bias", "output_bias")


def encode_genome(genome: Genome) -> dict:
    return {
        "segments": [asdict(s) for s in genome.segments],
        **{name: array.tolist() for name, array in zip(WEIGHTS, genome.arrays())},
    }


def decode_genome(data: dict) -> Genome:
    segments = [Segment(**s) for s in data["segments"]]
    arrays = [np.asarray(data[name], dtype=float) for name in WEIGHTS]
    shapes = [(MEMORY, INPUTS), (MEMORY, MEMORY), (ACTIONS, MEMORY), (MEMORY,), (ACTIONS,)]
    if not 1 <= len(segments) <= MAX_SEGMENTS:
        raise ValueError("Invalid segment count")
    if (
        any(
            not (6 <= s.length <= 22 and 4 <= s.width <= 16 and -0.8 <= s.angle <= 0.8)
            for s in segments
        )
        or segments[0].angle != 0
    ):
        raise ValueError("Invalid segment geometry")
    if any(a.shape != shape or not np.isfinite(a).all() for a, shape in zip(arrays, shapes)):
        raise ValueError("Invalid neural weights")
    return Genome(segments, *arrays)


def snapshot(sim: Simulation) -> dict:
    return {
        "version": VERSION,
        "seed": sim.seed,
        "width": sim.width,
        "height": sim.height,
        "islands": sim.islands,
        "brain_source": sim.brain_source,
        "run_name": sim.run_name,
        "groups": [asdict(group) for group in sim.groups.values()],
        "history": sim.history,
        "tick": sim.tick,
        "next_id": sim.next_id,
        "max_population": sim.max_population,
        "births": sim.births,
        "deaths": sim.deaths,
        "initial_energy": sim.initial_energy,
        "energy_input": sim.energy_input,
        "dissipated": sim.dissipated,
        "rng": sim.rng.bit_generator.state,
        "ancestor": encode_genome(sim.ancestor),
        "ancestry": sim.ancestry,
        "food": [asdict(food) for food in sim.food],
        "creatures": [
            {
                "id": c.id,
                "parent_id": c.parent_id,
                "generation": c.generation,
                "genome": encode_genome(c.genome),
                "energy": c.energy,
                "tissue": c.tissue,
                "age": c.age,
                "cooldown": c.cooldown,
                "memory": c.memory.tolist(),
                "actions": c.actions.tolist(),
                "sensors": c.sensors.tolist(),
                "group_id": c.group_id,
                "plant_eaten": c.plant_eaten,
                "meat_eaten": c.meat_eaten,
                "food_eaten": c.food_eaten.copy(),
                "position": list(c.body.position),
                "velocity": list(c.body.velocity),
                "angle": c.body.angle,
                "angular_velocity": c.body.angular_velocity,
            }
            for c in sim.creatures.values()
        ],
    }


def save(sim: Simulation, path: str | Path) -> None:
    """Replace the destination only after a complete snapshot has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot(sim), allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load(path: str | Path) -> Simulation:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") not in (1, 2, 3, 4, VERSION):
        raise ValueError("Unsupported save version")
    sim = Simulation(
        seed=data["seed"],
        population=0,
        plants=0,
        max_population=data["max_population"],
        brain="random",
        width=data.get("width", 1000),
        height=data.get("height", 700),
        islands=data.get("islands", []),
    )
    sim.ancestor = decode_genome(data["ancestor"])
    sim.brain_source = data.get("brain_source", "random")
    sim.tick = data["tick"]
    sim.run_name = data.get("run_name", Path(path).stem)
    sim.groups = {g["id"]: Group(**g) for g in data.get("groups", [])}
    for record in data["creatures"]:
        sim.next_id = record["id"]
        creature = sim.add_creature(
            decode_genome(record["genome"]), record["position"], record["angle"], record["energy"]
        )
        for name in ("parent_id", "generation", "tissue", "age", "cooldown"):
            setattr(creature, name, record[name])
        creature.memory = np.asarray(record["memory"], dtype=float)
        creature.actions = np.asarray(record["actions"], dtype=float)
        creature.sensors = np.asarray(record.get("sensors", [0] * INPUTS), dtype=float)
        creature.group_id = record.get("group_id", creature.group_id)
        creature.plant_eaten = record.get("plant_eaten", 0.0)
        creature.meat_eaten = record.get("meat_eaten", 0.0)
        creature.food_eaten = record.get("food_eaten", {})
        if creature.memory.shape != (MEMORY,) or creature.actions.shape != (ACTIONS,):
            raise ValueError("Invalid brain state")
        creature.body.velocity = record["velocity"]
        creature.body.angular_velocity = record["angular_velocity"]
    for food in data["food"]:
        if "kind" not in food:
            food["kind"] = (
                ("seed" if food.get("hardness", 0) > 0.6 else "algae")
                if food.get("renewable", True)
                else "carrion"
            )
        if food["kind"] not in (*FOOD_TYPES, "carrion"):
            raise ValueError("Unknown food kind")
    sim.food = [Food(**food) for food in data["food"]]
    sim.ancestry = {int(key): value for key, value in data["ancestry"].items()}
    for creature in sim.creatures.values():
        sim.ancestry[creature.id]["group_id"] = creature.group_id
    sim.max_generation = max((a["generation"] for a in sim.ancestry.values()), default=0)
    for name in (
        "tick",
        "next_id",
        "births",
        "deaths",
        "initial_energy",
        "energy_input",
        "dissipated",
    ):
        setattr(sim, name, data[name])
    sim.rng.bit_generator.state = data["rng"]
    sim.history = data.get("history", [])
    for row in sim.history:
        row["groups"] = {int(key): count for key, count in row["groups"].items()}
    if not sim.history:
        record_history(sim)
    return sim
