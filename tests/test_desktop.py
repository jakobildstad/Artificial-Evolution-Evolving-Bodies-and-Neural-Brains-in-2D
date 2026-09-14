"""Exercise desktop controls with SDL's offscreen driver."""

import pytest

from artificial_evolution.persistence import load
from artificial_evolution.simulation import Simulation


def test_pause_step_inspect_speed_and_save_load(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from artificial_evolution import rendering

    sim = Simulation(population=1, plants=2)
    creature = next(iter(sim.creatures.values()))
    camera = rendering.Camera(sim.width, sim.height)

    def key(code):
        return pygame.event.Event(pygame.KEYDOWN, key=code)

    events = iter(
        [
            [
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN,
                    button=1,
                    pos=tuple(camera.to_screen(creature.body.position)),
                ),
                key(pygame.K_SPACE),
            ],
            [key(pygame.K_n)],
            [key(pygame.K_s)],
            [pygame.event.Event(pygame.TEXTINPUT, text="Reed pond")],
            [key(pygame.K_RETURN)],
            [key(pygame.K_n)],
            [key(pygame.K_l)],
            [key(pygame.K_RETURN)],
            [key(pygame.K_TAB)],
            [pygame.event.Event(pygame.QUIT)],
        ]
    )
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    frames = []
    original_draw = rendering.draw

    def record_draw(screen, world, selected, paused, speed, *args):
        frames.append((world.tick, selected, paused, speed))
        original_draw(screen, world, selected, paused, speed, *args)

    monkeypatch.setattr(rendering, "draw", record_draw)
    path = tmp_path / "world.json"
    restored = rendering.run(sim, path, frame_limit=12)
    assert frames[0] == (0, creature.id, True, 1)
    assert frames[5][0] == 2
    assert frames[-1] == (1, None, True, 4)
    named = tmp_path / "Reed pond.json"
    assert restored.tick == load(named).tick == 1
    assert restored.run_name == "Reed pond"
    assert named.with_suffix(".population.csv").exists()
    assert named.with_suffix(".groups.csv").exists()


def test_info_button_pauses_and_blocks_world_actions(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from artificial_evolution import rendering

    sim = Simulation(population=1, plants=1)
    events = iter(
        [
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)],
            [
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, button=1, pos=rendering.INFO_BUTTON.center
                )
            ],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_i)],
            [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rendering.INFO_CLOSE.center)],
            [pygame.event.Event(pygame.QUIT)],
        ]
    )
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    frames = []
    original = rendering.draw

    def record(screen, world, selected, paused, speed, font, heading, notice, show_info, camera):
        frames.append((world.tick, paused, show_info))
        original(screen, world, selected, paused, speed, font, heading, notice, show_info, camera)

    monkeypatch.setattr(rendering, "draw", record)
    rendering.run(sim, tmp_path / "world.json", frame_limit=10)
    assert frames[1:4] == [(0, True, True), (0, True, True), (0, True, False)]
    assert frames[5:7] == [(1, True, True), (1, True, False)]
    assert sim.tick == 1


def test_analysis_pages_and_brain_navigation_pause_the_world(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from artificial_evolution import rendering

    sim = Simulation(population=2, plants=2)
    keys = [
        pygame.K_SPACE,
        pygame.K_a,
        pygame.K_2,
        pygame.K_4,
        pygame.K_3,
        pygame.K_RIGHT,
        pygame.K_n,
        pygame.K_ESCAPE,
        pygame.K_n,
    ]
    events = iter(
        [[pygame.event.Event(pygame.KEYDOWN, key=key)] for key in keys]
        + [[pygame.event.Event(pygame.QUIT)]]
    )
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    rendering.run(sim, tmp_path / "world.json", frame_limit=12)
    assert sim.tick == 1


def test_camera_zoom_drag_and_selection_while_paused(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from artificial_evolution import rendering

    sim = Simulation(population=1, plants=4)
    creature = next(iter(sim.creatures.values()))
    creature.body.position = (800, 550)
    sim.space.reindex_shapes_for_body(creature.body)
    expected = rendering.Camera(sim.width, sim.height)
    expected.zoom_at((500, 350), 1.2**3)
    expected.pan((-100, -30))
    position = tuple(expected.to_screen(creature.body.position))
    events = iter(
        [
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)],
            [pygame.event.Event(pygame.MOUSEWHEEL, y=3)],
            [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(500, 350))],
            [pygame.event.Event(pygame.MOUSEMOTION, rel=(100, 30))],
            [pygame.event.Event(pygame.MOUSEBUTTONUP, button=3, pos=(600, 380))],
            [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=position)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_HOME)],
            [pygame.event.Event(pygame.QUIT)],
        ]
    )
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (500, 350))
    frames = []
    original = rendering.draw

    def record(screen, world, selected, paused, speed, *args):
        camera = args[-1]
        frames.append((selected, camera.zoom, tuple(camera.center)))
        original(screen, world, selected, paused, speed, *args)

    monkeypatch.setattr(rendering, "draw", record)
    rendering.run(sim, tmp_path / "world.json", frame_limit=10)
    assert frames[5][0] == creature.id
    assert frames[5][1] == pytest.approx(expected.zoom)
    assert frames[6][2] == (800, 550)
    assert frames[7][1] == pytest.approx(expected.minimum_zoom)
    assert sim.tick == 0
