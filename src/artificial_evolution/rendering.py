"""Pygame desktop view; wall-clock rendering never changes the physics timestep."""

import colorsys
import time
from pathlib import Path

import pygame

from .ecology import DT, FOOD_CAPACITY, HEIGHT, WIDTH
from .persistence import load, save
from .simulation import Simulation

PANEL = 300
BACKGROUND = (15, 23, 28)
TEXT = (216, 230, 226)
MUTED = (132, 156, 157)


def draw(
    screen: pygame.Surface,
    sim: Simulation,
    selected: int | None,
    paused: bool,
    speed: int,
    font: pygame.font.Font,
    heading: pygame.font.Font,
    notice: str = "",
) -> None:
    screen.fill(BACKGROUND)
    for food in sim.food:
        if food.energy > 0.5:
            radius = max(2, min(8, int(2 + food.energy / FOOD_CAPACITY * 3)))
            pygame.draw.circle(
                screen,
                (84, 160, 105) if food.renewable else (179, 110, 74),
                (int(food.x), int(food.y)),
                radius,
            )
    for creature in sim.creatures.values():
        # Color shows lineage depth, never a species or mechanical trait preset.
        rgb = colorsys.hsv_to_rgb(
            (0.49 + creature.generation * 0.085) % 1,
            0.45,
            0.45 + 0.5 * creature.energy / creature.capacity,
        )
        color = tuple(int(v * 255) for v in rgb)
        for shape in creature.shapes:
            points = [creature.body.local_to_world(v) for v in shape.get_vertices()]
            pygame.draw.polygon(screen, color, points)
            pygame.draw.polygon(
                screen,
                (245, 222, 151) if creature.id == selected else (39, 68, 71),
                points,
                2 if creature.id == selected else 1,
            )
        pygame.draw.circle(screen, (244, 218, 161), creature.mouth, 2)
        if creature.actions[2] > 0.1 and creature.id == selected:
            pygame.draw.circle(screen, (161, 131, 88), creature.mouth, 7, 1)
    pygame.draw.rect(screen, (69, 89, 89), (0, 0, WIDTH, HEIGHT), 3)
    pygame.draw.rect(screen, (23, 34, 40), (WIDTH, 0, PANEL, HEIGHT))

    def label(text: str, y: int, color: tuple = TEXT, large: bool = False) -> None:
        screen.blit((heading if large else font).render(text, True, color), (WIDTH + 18, y))

    label("ARTIFICIAL EVOLUTION", 20, large=True)
    label(f"{'PAUSED' if paused else 'LIVE'}  /  {speed}x  /  {sim.time:.1f}s", 52, MUTED)
    stats = sim.stats()
    for y, text in zip(
        range(90, 216, 25),
        [
            f"Population       {stats['population']} / {sim.max_population}",
            f"Births / deaths   {sim.births} / {sim.deaths}",
            f"Deepest generation   {stats['max_generation']}",
            f"Mean segments     {stats['mean_segments']}",
            f"Mean energy       {stats['mean_energy']}",
        ],
    ):
        label(text, y)
    creature = sim.creatures.get(selected)
    label("CREATURE INSPECTOR", 237, large=True)
    if creature:
        lines = [
            f"ID {creature.id}  /  generation {creature.generation}",
            f"Parent: {creature.parent_id or 'ancestral genome'}",
            f"Age: {creature.age:.1f}s",
            f"Energy: {creature.energy:.1f} / {creature.capacity:.1f}",
            f"Tissue: {creature.tissue:.1f} / {creature.genome.tissue:.1f}",
            f"Segments: {len(creature.genome.segments)}",
            f"Mass: {creature.body.mass:.2f}",
            "Thrust / turn / bite / reproduce:",
            "  ".join(f"{a:+.2f}" for a in creature.actions),
        ]
        lineage = []
        ancestor = creature.id
        while ancestor is not None and len(lineage) < 5:
            lineage.append(str(ancestor))
            ancestor = sim.ancestry[ancestor]["parent_id"]
        lines.append(" < ".join(lineage) + (" ..." if ancestor else ""))
    elif selected in sim.ancestry:
        record = sim.ancestry[selected]
        lines = [
            f"Creature {selected} died at {record['died']:.1f}s",
            f"Parent: {record['parent_id']}",
            "Click another creature to inspect.",
        ]
    else:
        lines = [
            "Click a creature to inspect it.",
            "Green: plants   /   rust: carrion",
            "Body color follows generation.",
        ]
    for y, text in zip(range(270, 491, 22), lines):
        label(text, y, MUTED)
    for y, text in zip(
        range(523, 649, 23),
        [
            "Space   pause / resume",
            "Tab       cycle 1x / 4x / 12x",
            "N          one step while paused",
            "S / L     save / load snapshot",
            "Esc       quit",
        ],
    ):
        label(text, y)
    label(notice[:35] if notice else f"Seed {sim.seed}  |  fixed 60 Hz physics", 669, MUTED)


def run(sim: Simulation, save_path: Path, frame_limit: int | None = None) -> Simulation:
    """Return the live state on exit; frame_limit supports desktop smoke checks."""
    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH + PANEL, HEIGHT))
        pygame.display.set_caption("Artificial Evolution: Evolving Bodies and Neural Brains in 2D")
        font = pygame.font.Font(None, 21)
        heading = pygame.font.Font(None, 23)
        clock = pygame.time.Clock()
        paused, running = False, True
        selected = None
        speeds, speed_index = [1, 4, 12], 0
        accumulator = 0.0
        notice, notice_until = "", 0.0
        frames = 0
        while running:
            elapsed = min(clock.tick(60) / 1000, 0.25)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if event.pos[0] < WIDTH:
                        selected = next(
                            (
                                c.id
                                for c in sim.creatures.values()
                                if any(s.point_query(event.pos).distance <= 3 for s in c.shapes)
                            ),
                            None,
                        )
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        accumulator = 0
                    elif event.key == pygame.K_TAB:
                        speed_index = (speed_index + 1) % len(speeds)
                    elif event.key == pygame.K_n and paused:
                        sim.step()
                    elif event.key in (pygame.K_s, pygame.K_l):
                        try:
                            if event.key == pygame.K_s:
                                save(sim, save_path)
                                notice = "Snapshot saved"
                            else:
                                sim = load(save_path)
                                selected, accumulator = None, 0
                                notice = "Snapshot loaded"
                        except (OSError, ValueError, KeyError, TypeError) as error:
                            notice = f"Snapshot error: {error}"
                            print(notice)
                        notice_until = time.monotonic() + 5
            if not paused:
                accumulator += elapsed * speeds[speed_index]
                # Limit work per frame; a busy desktop slows down rather than enlarging dt.
                for _ in range(min(int(accumulator / DT), 48)):
                    sim.step()
                    accumulator -= DT
                accumulator = min(accumulator, DT * 48)
            draw(
                screen,
                sim,
                selected,
                paused,
                speeds[speed_index],
                font,
                heading,
                notice if time.monotonic() < notice_until else "",
            )
            pygame.display.flip()
            frames += 1
            if frame_limit is not None and frames >= frame_limit:
                break
    finally:
        pygame.quit()
    return sim
