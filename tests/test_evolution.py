"""Behavioral checks for heredity, energy flows, reproducibility, and snapshots."""

import numpy as np
import pytest

from artificial_evolution.ecology import DT, Food, make_body
from artificial_evolution.genome import MAX_SEGMENTS
from artificial_evolution.persistence import load, save, snapshot
from artificial_evolution.simulation import Simulation


def test_founders_are_independent_copies_and_mutations_remain_viable():
    sim = Simulation(seed=7, population=3, plants=0)
    first, second, _ = sim.creatures.values()
    for a, b in zip(first.genome.arrays(), second.genome.arrays()):
        np.testing.assert_array_equal(a, b)
        assert not np.shares_memory(a, b)
    original = first.genome.copy()
    genome = first.genome
    counts = set()
    for _ in range(400):
        genome = genome.mutated(sim.rng)
        counts.add(len(genome.segments))
        assert 1 <= len(genome.segments) <= MAX_SEGMENTS
        body, shapes = make_body(sim.space, genome, (400, 350), 0)
        assert body.mass > 0 and body.moment > 0
        sim.space.remove(*shapes, body)
    assert len(counts) >= 3
    for before, after in zip(original.arrays(), first.genome.arrays()):
        np.testing.assert_array_equal(before, after)
    assert first.genome.segments == original.segments


def test_reproduction_pays_for_child_tissue_and_energy():
    sim = Simulation(population=1, plants=0)
    parent = next(iter(sim.creatures.values()))
    parent.body.position = (500, 350)
    parent.age = 10
    parent.energy = parent.capacity
    before = sim.total_energy()
    child = sim.reproduce(parent)
    assert child is not None
    assert child.parent_id == parent.id and child.generation == 1
    assert sim.total_energy() == pytest.approx(before)
    assert parent.energy < parent.capacity
    assert not np.shares_memory(child.genome.input_weights, parent.genome.input_weights)
    assert sim.reproduce(parent) is None


def test_biting_live_tissue_and_death_conserve_energy():
    sim = Simulation(population=2, plants=0)
    eater, victim = sim.creatures.values()
    eater.body.position, eater.body.angle = (400, 350), 0
    victim.body.position, victim.body.angle = (421, 350), 0
    sim.space.reindex_shapes_for_body(eater.body)
    sim.space.reindex_shapes_for_body(victim.body)
    eater.energy = 10
    eater.actions[2] = 1
    before = sim.total_energy() + sim.dissipated
    tissue = victim.tissue
    sim.feed(eater, 1)
    assert victim.tissue < tissue
    assert eater.energy > 10
    assert sim.total_energy() + sim.dissipated == pytest.approx(before)
    sim.kill(victim)
    assert not sim.food[-1].renewable
    assert sim.total_energy() + sim.dissipated == pytest.approx(before)


def test_plant_digestion_capacity_and_regrowth():
    sim = Simulation(population=1, plants=0)
    eater = next(iter(sim.creatures.values()))
    eater.energy = eater.capacity - 1
    eater.actions[2] = 1
    sim.food = [Food(*eater.mouth, 10)]
    sim.feed(eater, 1)
    assert eater.energy == pytest.approx(eater.capacity)
    assert sim.food[0].energy == pytest.approx(8.75)
    assert sim.dissipated == pytest.approx(0.25)
    eater.actions[2] = -1
    sim.tick = 1  # Hold the manually supplied action for this tick.
    remaining = sim.food[0].energy
    sim.step()
    assert sim.food[0].energy > remaining


def test_fixed_step_loop_energy_ledger_and_seed():
    left = Simulation(seed=5, population=4, plants=12)
    right = Simulation(seed=5, population=4, plants=12)
    for _ in range(360):
        left.step()
        right.step()
    assert left.time == 360 * DT
    assert abs(left.stats()["energy_balance_error"]) < 1e-8
    assert snapshot(left) == snapshot(right)
    for creature in left.creatures.values():
        assert np.isfinite(creature.memory).all()
        assert np.isfinite(creature.body.position).all()


def test_save_load_restores_full_state_and_can_continue(tmp_path):
    sim = Simulation(seed=9, population=2, plants=6)
    parent = next(iter(sim.creatures.values()))
    parent.body.position = (500, 350)
    parent.age, parent.energy = 10, parent.capacity
    assert sim.reproduce(parent) is not None
    for _ in range(42):
        sim.step()
    path = tmp_path / "nested" / "world.json"
    save(sim, path)
    restored = load(path)
    assert snapshot(restored) == snapshot(sim)
    assert restored.rng.random() == sim.rng.random()
    balance = restored.stats()["energy_balance_error"]
    for _ in range(120):
        restored.step()
    assert restored.stats()["energy_balance_error"] == pytest.approx(balance, abs=1e-8)
    path.write_text('{"version": 999}')
    with pytest.raises(ValueError, match="version"):
        load(path)


def test_extinct_world_keeps_running():
    sim = Simulation(population=0, plants=0)
    for _ in range(12):
        sim.step()
    assert sim.stats()["population"] == 0
    assert sim.stats()["energy_balance_error"] == 0
