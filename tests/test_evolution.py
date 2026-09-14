"""Behavioral checks for heredity, energy flows, reproducibility, and snapshots."""

import numpy as np
import pytest

from artificial_evolution.ecology import (
    DT,
    FOOD_TYPES,
    HEIGHT,
    PARENT_PROTECTION_TIME,
    WIDTH,
    Food,
    make_body,
)
from artificial_evolution.genome import MAX_SEGMENTS, Genome
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


@pytest.mark.parametrize("restore", [False, True])
def test_parent_cannot_bite_child_until_grace_period_expires(tmp_path, restore):
    sim = Simulation(population=1, plants=0)
    parent = next(iter(sim.creatures.values()))
    parent.body.position, parent.body.angle = (400, 350), 0
    sim.space.reindex_shapes_for_body(parent.body)
    parent.energy = 10
    parent.actions[2] = 1
    child = sim.add_creature(parent.genome.copy(), (421, 350), 0, 40, parent)
    tissue = child.tissue
    before = sim.total_energy() + sim.dissipated

    for age in (0, PARENT_PROTECTION_TIME - DT):
        child.age = age
        if restore:
            path = tmp_path / "young-family.json"
            save(sim, path)
            sim = load(path)
            parent, child = sim.creatures[parent.id], sim.creatures[child.id]
        sim.feed(parent, 1)
        assert child.tissue == tissue
        assert parent.energy == 10
        assert parent.meat_eaten == 0
        assert sim.total_energy() + sim.dissipated == pytest.approx(before)

    child.age = PARENT_PROTECTION_TIME
    sim.feed(parent, 1)
    assert child.tissue < tissue
    assert parent.energy > 10
    assert sim.total_energy() + sim.dissipated == pytest.approx(before)


def test_newborn_can_still_be_bitten_by_an_unrelated_creature():
    sim = Simulation(population=2, plants=0)
    parent, predator = sim.creatures.values()
    predator.body.position, predator.body.angle = (400, 350), 0
    sim.space.reindex_shapes_for_body(predator.body)
    predator.energy = 10
    predator.actions[2] = 1
    child = sim.add_creature(parent.genome.copy(), (421, 350), 0, 40, parent)
    tissue = child.tissue
    before = sim.total_energy() + sim.dissipated
    sim.feed(predator, DT, nearby_creatures=[child])
    assert child.age == 0
    assert child.tissue < tissue
    assert predator.energy > 10
    assert sim.total_energy() + sim.dissipated == pytest.approx(before)


def test_plant_digestion_capacity_without_regrowth_in_place():
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
    assert sim.food[0].energy == remaining


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


def test_nearby_feeding_matches_full_scan_during_collisions(monkeypatch):
    fast = Simulation(seed=5, population=4, plants=0)
    reference = Simulation(seed=5, population=4, plants=0)
    for sim in (fast, reference):
        for index, creature in enumerate(sim.creatures.values()):
            creature.body.position = (400 + index * 20, 350)
            creature.body.angle = 0
            sim.space.reindex_shapes_for_body(creature.body)
            creature.energy = 10
            creature.genome.output_weights.fill(0)
            creature.genome.output_bias[:] = (0.3, 0, 1, -1)
        sim.food = [Food(408, 350, 1), Food(470, 350, 2)]
        sim.initial_energy = sim.total_energy()
    full_scan = reference.feed
    monkeypatch.setattr(reference, "feed", lambda eater, dt, *_: full_scan(eater, dt))
    tissue_eaten = False
    for _ in range(120):
        fast.step()
        reference.step()
        tissue_eaten |= any(c.tissue < c.genome.tissue for c in fast.creatures.values())
    assert tissue_eaten
    assert snapshot(fast) == snapshot(reference)


def test_food_respawns_elsewhere_and_restores_pending_timer(tmp_path):
    sim = Simulation(seed=17, population=0, plants=1)
    plant = sim.food[0]
    old_position = (plant.x, plant.y)
    plant.energy = 0.005
    sim.food.append(Food(100, 200, 10, False))
    sim.initial_energy = sim.total_energy()
    sim.update_food()
    assert plant.energy == 0
    assert (
        FOOD_TYPES[plant.kind].respawn[0] <= plant.respawn_in <= FOOD_TYPES[plant.kind].respawn[1]
    )
    assert (plant.x, plant.y) == old_position
    assert sim.energy_input == 0

    path = tmp_path / "pending.json"
    save(sim, path)
    restored = load(path)
    assert snapshot(restored) == snapshot(sim)
    for _ in range(int(np.ceil(plant.respawn_in / DT)) + 1):
        sim.step()
        restored.step()
    assert snapshot(restored) == snapshot(sim)
    assert plant.energy == pytest.approx(FOOD_TYPES[plant.kind].energy)
    assert (plant.x, plant.y) != old_position
    assert 12 <= plant.x <= WIDTH - 12 and 12 <= plant.y <= HEIGHT - 12
    assert sim.energy_input == plant.energy
    carrion = sim.food[1]
    assert (carrion.x, carrion.y) == (100, 200)
    assert 0 < carrion.energy < 10
    assert carrion.respawn_in == 0
    assert abs(sim.stats()["energy_balance_error"]) < 1e-8


def test_healthy_creature_survives_old_age_cutoff_but_can_starve():
    sim = Simulation(population=1, plants=0)
    creature = next(iter(sim.creatures.values()))
    creature.age = 240 - DT
    creature.energy = creature.capacity / 2
    for _ in range(6):
        sim.step()
    assert creature.id in sim.creatures
    assert creature.age > 240
    creature.energy = 0.0001
    sim.step()
    assert creature.id not in sim.creatures
    assert sim.deaths == 1


def test_version_one_saves_load_without_respawn_timers(tmp_path):
    import json

    data = snapshot(Simulation(population=0, plants=2))
    data["version"] = 1
    del data["brain_source"]
    for food in data["food"]:
        del food["respawn_in"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))
    sim = load(path)
    assert sim.brain_source == "random"
    assert all(food.respawn_in == 0 for food in sim.food)
    sim.step()


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


def test_pretrained_is_default_and_random_is_selectable():
    trained = Simulation(population=2, plants=0)
    other_seed = Simulation(seed=22, population=1, plants=0)
    random = Simulation(population=1, plants=0, brain="random")
    assert trained.brain_source == "pretrained"
    np.testing.assert_array_equal(trained.ancestor.input_weights, other_seed.ancestor.input_weights)
    assert not np.array_equal(trained.ancestor.input_weights, random.ancestor.input_weights)
    first, second = trained.creatures.values()
    first.genome.input_weights[0, 0] += 1
    np.testing.assert_array_equal(second.genome.input_weights, Genome.pretrained().input_weights)
    with pytest.raises(ValueError, match="Brain"):
        Simulation(population=0, brain="invalid")


def test_pretrained_forager_survives_by_eating_with_low_initial_energy():
    sim = Simulation(seed=11, population=1, max_population=1)
    creature = next(iter(sim.creatures.values()))
    creature.energy = 15
    sim.initial_energy = sim.total_energy()
    for _ in range(10800):
        sim.step()
    assert creature.id in sim.creatures
    assert creature.energy > 15
    assert sim.energy_input > 0
    assert abs(sim.stats()["energy_balance_error"]) < 1e-7
