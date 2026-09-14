"""Resource tradeoffs, connected terrain, camera interaction, and habitat snapshots."""

import json
from collections import deque

import numpy as np
import pytest

from artificial_evolution.analysis import record
from artificial_evolution.ecology import (
    DT,
    FOOD_TYPES,
    Food,
    food_resistance,
    sense,
    sense_population,
)
from artificial_evolution.genome import Segment
from artificial_evolution.persistence import load, save, snapshot
from artificial_evolution.simulation import Simulation
from artificial_evolution.terrain import TERRAIN_FILTER, open_water, visible


def test_islands_separate_basins_but_channels_connect_open_water():
    sim = Simulation(seed=3)
    assert (sim.width, sim.height) == (1600, 1100)
    assert not open_water(sim.space, (560, 200), 5)
    assert not visible(sim.space, (430, 200), (680, 200))
    assert visible(sim.space, (430, 360), (680, 360))
    for creature in sim.creatures.values():
        assert open_water(sim.space, tuple(creature.body.position), creature.radius)
    for food in sim.food:
        assert open_water(sim.space, (food.x, food.y), 10)
    # A body with a 25-unit radius can traverse all three basins, with several detours.
    water = {
        (x, y)
        for x in range(40, sim.width, 40)
        for y in range(40, sim.height, 40)
        if open_water(sim.space, (x, y), 25)
    }
    visited, queue = {(200, 200)}, deque([(200, 200)])
    while queue:
        x, y = queue.popleft()
        for target in ((x + 40, y), (x - 40, y), (x, y + 40), (x, y - 40)):
            if (
                target in water
                and target not in visited
                and sim.space.segment_query_first((x, y), target, 25, TERRAIN_FILTER) is None
            ):
                visited.add(target)
                queue.append(target)
    assert (800, 200) in visited and (1440, 200) in visited
    assert visited == water


def test_terrain_collisions_sensors_and_occlusion_agree():
    sim = Simulation(population=0, plants=0, islands=[[(500, 0), (530, 0), (530, 450), (500, 450)]])
    creature = sim.add_creature(sim.ancestor.copy(), (460, 200), 0, 80)
    # Hidden food is closer than visible food, but must not attract the creature through rock.
    points = np.array([(540, 200), (390, 290)])
    actual = sense_population([creature], points, sim.width, sim.height, sim.space)
    reference = sense(creature, points, np.empty((0, 2)), sim.width, sim.height, sim.space)
    np.testing.assert_allclose(actual[0], reference, atol=1e-12)
    assert actual[0, 0] < 0 and actual[0, 1] > 0
    assert actual[0, 4] > 0.7
    creature.body.velocity = (300, 0)
    for _ in range(60):
        sim.space.step(DT)
    assert creature.body.position.x < 500


def test_resource_types_have_different_costs_drift_and_preserve_energy(tmp_path):
    sim = Simulation(population=0, plants=0, islands=[])
    single_genome = sim.ancestor.copy()
    single_genome.segments = [Segment(20, 10)]
    long_genome = single_genome.copy()
    long_genome.segments = [Segment(10, 10), Segment(10, 10)]
    single = sim.add_creature(single_genome, (200, 200), 0, 10)
    long = sim.add_creature(long_genome, (400, 200), 0, 10)
    weed = Food(*single.mouth, FOOD_TYPES["weed"].energy, kind="weed", hardness=0.25)
    assert single.area == long.area
    assert food_resistance(weed, long) < food_resistance(weed, single)
    seed = Food(*single.mouth, 42, kind="seed", hardness=1)
    assert food_resistance(seed, single) > food_resistance(Food(*single.mouth, 10), single)
    single.actions[2] = 1
    # Capacity-limited feeding can produce NumPy scalars in resource energy.
    single.energy = np.float64(single.capacity - 0.1)
    sim.food = [weed, Food(800, 600, 16, kind="plankton", drift_angle=0)]
    before = sim.total_energy()
    sim.feed(single, 0.1)
    assert single.food_eaten["weed"] == pytest.approx(30 - weed.energy)
    assert sim.total_energy() + sim.dissipated == pytest.approx(before)
    sim.update_food()
    record(sim)
    assert sim.food[1].x > 800 and sim.food[1].y == 600
    path = tmp_path / "habitats.json"
    save(sim, path)
    restored = load(path)
    assert snapshot(restored) == snapshot(sim)
    for _ in range(60):
        restored.update_food()
        sim.update_food()
    assert snapshot(restored) == snapshot(sim)


def test_all_food_types_respawn_in_open_water_and_legacy_world_stays_open(tmp_path):
    sim = Simulation(population=0, plants=0)
    for kind in FOOD_TYPES:
        food = Food(100, 100, 0, respawn_in=DT, kind=kind)
        sim.food.append(food)
    sim.update_food()
    for food in sim.food:
        assert food.energy == FOOD_TYPES[food.kind].energy
        assert open_water(sim.space, (food.x, food.y), 10)
        assert (food.x, food.y) != (100, 100)
    assert abs(sim.stats()["energy_balance_error"]) < 1e-8
    path = tmp_path / "islands.json"
    save(sim, path)
    restored = load(path)
    assert snapshot(restored) == snapshot(sim)
    assert not open_water(restored.space, (560, 200), 5)
    legacy = snapshot(sim)
    legacy["version"] = 4
    for key in ("width", "height", "islands"):
        del legacy[key]
    for food in legacy["food"]:
        food.pop("kind")
        food.pop("drift_angle")
    path.write_text(json.dumps(legacy))
    restored = load(path)
    assert (restored.width, restored.height, restored.islands) == (1000, 700, [])
    assert open_water(restored.space, (560, 200), 5)


def test_zoom_anchor_pan_bounds_and_minimap_do_not_change_simulation():
    import pygame

    from artificial_evolution.pond_view import MINIMAP, Camera

    sim = Simulation(population=0, plants=0)
    before = snapshot(sim)
    camera = Camera(sim.width, sim.height)
    anchor = (500, 350)
    world_point = camera.to_world(anchor)
    camera.zoom_at(anchor, 3)
    assert camera.to_world(anchor).distance_to(world_point) < 1e-10
    camera.pan((100, 40))
    point = (830, 420)
    assert camera.to_world(camera.to_screen(point)).distance_to(point) < 1e-10
    camera.pan((-10000, 10000))
    assert camera.to_world((0, 0)).x >= -1e-8
    assert camera.to_world((1000, 700)).y <= sim.height + 1e-8
    click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=MINIMAP.center)
    assert camera.handle_event(click)
    assert camera.center.distance_to((sim.width / 2, sim.height / 2)) < 1e-10
    assert snapshot(sim) == before
