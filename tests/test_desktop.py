"""Exercise desktop controls with SDL's offscreen driver."""

from artificial_evolution.persistence import load
from artificial_evolution.simulation import Simulation


def test_pause_step_inspect_speed_and_save_load(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from artificial_evolution import rendering

    sim = Simulation(population=1, plants=2)
    creature = next(iter(sim.creatures.values()))

    def key(code):
        return pygame.event.Event(pygame.KEYDOWN, key=code)

    events = iter(
        [
            [
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, button=1, pos=tuple(creature.body.position)
                ),
                key(pygame.K_SPACE),
            ],
            [key(pygame.K_n)],
            [key(pygame.K_s)],
            [key(pygame.K_n)],
            [key(pygame.K_l)],
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
    restored = rendering.run(sim, path, frame_limit=10)
    assert frames[0] == (0, creature.id, True, 1)
    assert frames[3][0] == 2
    assert frames[-1] == (1, None, True, 4)
    assert restored.tick == load(path).tick == 1


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

    def record(screen, world, selected, paused, speed, font, heading, notice, show_info):
        frames.append((world.tick, paused, show_info))
        original(screen, world, selected, paused, speed, font, heading, notice, show_info)

    monkeypatch.setattr(rendering, "draw", record)
    rendering.run(sim, tmp_path / "world.json", frame_limit=10)
    assert frames[1:4] == [(0, True, True), (0, True, True), (0, True, False)]
    assert frames[5:7] == [(1, True, True), (1, True, False)]
    assert sim.tick == 1
