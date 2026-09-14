"""Checks for ecology tradeoffs, analysis fidelity, and named run persistence."""

import csv

import numpy as np
import pytest

from artificial_evolution.analysis import MAX_SAMPLES, classify, export_csv, group_rows, record
from artificial_evolution.ecology import Food, sense, sense_population
from artificial_evolution.genome import Segment
from artificial_evolution.persistence import load, save, snapshot
from artificial_evolution.run_dialog import RunDialog, named_path
from artificial_evolution.simulation import Simulation


def test_groups_are_stable_descriptions_and_survive_save_load(tmp_path):
    sim = Simulation(population=1, plants=0)
    original = next(iter(sim.creatures.values()))
    original_weights = original.genome.input_weights.copy()
    assert classify(original.genome.copy(), sim.groups, 10) == original.group_id
    tiny = original.genome.copy()
    tiny.segments = [Segment(6, 4)]
    child = sim.add_creature(tiny, (500, 350), 0, 20, original)
    assert child.group_id != original.group_id
    assert classify(tiny, sim.groups, 20) == child.group_id
    np.testing.assert_array_equal(original_weights, original.genome.input_weights)
    sim.initial_energy = sim.total_energy()
    for _ in range(61):
        sim.step()
    sim.run_name = "Quiet reeds"
    path = tmp_path / "Quiet reeds.json"
    save(sim, path)
    restored = load(path)
    assert snapshot(restored) == snapshot(sim)
    export_csv(restored, path)
    with path.with_suffix(".population.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) >= 2
    assert int(rows[-1]["population"]) == len(sim.creatures)
    assert f"group_{child.group_id:03d}" in rows[-1]
    assert group_rows(restored) == group_rows(sim)


def test_history_thinning_keeps_start_and_latest():
    sim = Simulation(population=0, plants=0)
    for tick in range(1, MAX_SAMPLES + 1):
        sim.tick = tick * 60
        record(sim)
    assert len(sim.history) <= MAX_SAMPLES
    assert sim.history[0]["seconds"] == 0
    assert sim.history[-1]["seconds"] == sim.time


def test_batched_sensors_match_scalar_reference():
    sim = Simulation(seed=12, population=7, plants=12)
    creatures = list(sim.creatures.values())
    for index, creature in enumerate(creatures):
        creature.body.angle = index * 0.8
        creature.body.velocity = (index * 4, -index * 2)
    points = np.array([(f.x, f.y) for f in sim.food])
    positions = np.array([tuple(c.body.position) for c in creatures])
    reference = np.array(
        [sense(c, points, np.delete(positions, i, axis=0)) for i, c in enumerate(creatures)]
    )
    np.testing.assert_allclose(sense_population(creatures, points), reference, atol=1e-12)
    single = sense_population(creatures[:1], np.empty((0, 2)))
    np.testing.assert_array_equal(single[0, :4], np.zeros(4))
    assert np.isfinite(single).all()


def test_extra_segment_offspring_can_be_funded(monkeypatch):
    sim = Simulation(population=1, plants=0)
    parent = next(iter(sim.creatures.values()))
    parent.body.position = (500, 350)
    parent.age, parent.energy = 10, parent.capacity
    bigger = parent.genome.copy()
    bigger.segments.append(Segment(14, 10))
    monkeypatch.setattr(parent.genome, "mutated", lambda rng: bigger)
    total = sim.total_energy()
    child = sim.reproduce(parent)
    assert child is not None
    assert len(child.genome.segments) == 3
    assert sim.total_energy() == pytest.approx(total)


def test_large_mouth_processes_hard_food_faster_and_healing_costs_energy():
    sim = Simulation(population=0, plants=0)
    small_genome, large_genome = sim.ancestor.copy(), sim.ancestor.copy()
    small_genome.segments, large_genome.segments = [Segment(6, 4)], [Segment(20, 16)]
    small = sim.add_creature(small_genome, (200, 350), 0, 10)
    large = sim.add_creature(large_genome, (700, 350), 0, 10)
    sim.food = [Food(*small.mouth, 22, hardness=0.9), Food(*large.mouth, 22, hardness=0.9)]
    small.actions[2] = large.actions[2] = 1
    sim.feed(small, 1)
    sim.feed(large, 1)
    assert large.energy - 10 > 5 * (small.energy - 10)
    large.tissue *= 0.8
    large.energy = large.capacity
    before_tissue = large.tissue
    sim.initial_energy = sim.total_energy() + sim.dissipated - sim.energy_input
    sim.step()
    assert large.tissue > before_tissue
    assert abs(sim.stats()["energy_balance_error"]) < 1e-8


def test_named_save_rejects_paths_and_requires_overwrite_confirmation(tmp_path):
    import pygame

    for name in ("", "../outside", "a/b", "CON", "a" * 65):
        with pytest.raises(ValueError):
            named_path(tmp_path, name)
    sim = Simulation(population=0, plants=0)
    path = named_path(tmp_path, "My pond")
    path.write_text("existing run")
    dialog = RunDialog("save", tmp_path, "My pond")
    enter = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)
    assert dialog.handle_event(enter, sim) is None
    assert path.read_text() == "existing run"
    assert dialog.replace
    assert dialog.handle_event(enter, sim) == (sim, path)
    assert load(path).run_name == "My pond"
