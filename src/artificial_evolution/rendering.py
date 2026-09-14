"""Pygame desktop view; wall-clock rendering never changes the physics timestep."""

import time
from pathlib import Path

import pygame

from .analysis_view import CLOSE as ANALYSIS_CLOSE
from .analysis_view import Dashboard
from .ecology import DEFAULT_PLANTS, DT, PARENT_PROTECTION_TIME
from .pond_view import VIEW_HEIGHT as HEIGHT
from .pond_view import VIEW_WIDTH as WIDTH
from .pond_view import Camera, draw_pond
from .run_dialog import RunDialog
from .simulation import Simulation

PANEL = 300
BACKGROUND = (18, 44, 39)
TEXT = (222, 233, 205)
MUTED = (144, 170, 146)
INFO_BUTTON = pygame.Rect(WIDTH + 211, 46, 72, 28)
INFO_CLOSE = pygame.Rect(1110, 88, 90, 30)
ANALYZE_BUTTON = pygame.Rect(WIDTH + 18, 500, 264, 32)
SAVE_BUTTON = pygame.Rect(WIDTH + 18, 544, 126, 30)
LOAD_BUTTON = pygame.Rect(WIDTH + 156, 544, 126, 30)


def draw_info(screen: pygame.Surface, font: pygame.font.Font, heading: pygame.font.Font) -> None:
    """Explain mechanical tradeoffs while the world is paused behind the overlay."""
    shade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    shade.fill((0, 0, 0, 185))
    screen.blit(shade, (0, 0))
    pygame.draw.rect(screen, (23, 34, 40), (70, 70, 1160, 590), border_radius=12)
    screen.blit(heading.render("BODIES, BRAINS & SURVIVAL", True, TEXT), (100, 95))
    pygame.draw.rect(screen, (49, 73, 77), INFO_CLOSE, border_radius=5)
    screen.blit(font.render("Close", True, TEXT), (1135, 95))
    sections = [
        (
            "Size & mass",
            [
                "More body area adds mass, edible tissue and energy capacity.",
                "It also raises upkeep, bite strength and movement costs.",
                "More mass needs more force for the same acceleration.",
            ],
        ),
        (
            "Length & shape",
            [
                "Long or bent bodies change collisions and the reach of tissue.",
                "Mass farther from the center makes turning harder.",
                "Extra segments add storage; all form one rigid body.",
            ],
        ),
        (
            "Head width & movement",
            [
                "The first segment sets the forward direction and mouth.",
                "A wider head supplies stronger bites and turning torque.",
                "The brain chooses thrust, turning, biting and reproduction.",
            ],
        ),
        (
            "Segments & mutation",
            [
                "Bodies have 1-5 segments. Offspring can gain, lose or copy one.",
                "Dimensions and bends can mutate along with neural weights.",
                "Larger offspring require more parental energy to build.",
            ],
        ),
        (
            "Tissue & energy",
            [
                "Energy reserves fuel movement and upkeep; zero means death.",
                "Other creatures can bite tissue and gain energy from it.",
                "Below 20% tissue is fatal. Well-fed bodies spend energy healing.",
            ],
        ),
        (
            "Food & scarcity",
            [
                f"{DEFAULT_PLANTS} food slots: algae, seeds, waterweed and plankton.",
                "Seeds favor strong bites; extra segments process waterweed.",
                "Plankton drifts. Food renews elsewhere after 18-85 seconds.",
            ],
        ),
        (
            "Birth & age",
            [
                "Maturity starts at 8 seconds; upkeep rises with age.",
                "A parent pays for its child's tissue and starting energy.",
                f"Parents cannot bite newborns for {PARENT_PROTECTION_TIME:g} seconds.",
            ],
        ),
        (
            "Brains & colors",
            [
                "Founders share a pretrained foraging brain by default.",
                "Use --brain random for an untrained ancestor. Both evolve.",
                "Color shows genetic group; brightness shows energy reserves.",
            ],
        ),
    ]
    for index, (title, lines) in enumerate(sections):
        x = 100 if index < 4 else 660
        y = 150 + (index % 4) * 110
        screen.blit(heading.render(title, True, (148, 207, 187)), (x, y))
        for offset, line in enumerate(lines):
            screen.blit(font.render(line, True, TEXT), (x, y + 28 + offset * 21))
    screen.blit(
        font.render("Paused while open. Click Close, or press I / Esc to return.", True, MUTED),
        (100, 628),
    )


def draw(
    screen: pygame.Surface,
    sim: Simulation,
    selected: int | None,
    paused: bool,
    speed: int,
    font: pygame.font.Font,
    heading: pygame.font.Font,
    notice: str = "",
    show_info: bool = False,
    camera: Camera | None = None,
) -> None:
    screen.fill(BACKGROUND)
    camera = camera or Camera(sim.width, sim.height)
    draw_pond(screen, sim, camera, selected, font)
    pygame.draw.rect(screen, (27, 43, 34), (WIDTH, 0, PANEL, HEIGHT))

    def label(text: str, y: int, color: tuple = TEXT, large: bool = False) -> None:
        screen.blit((heading if large else font).render(text, True, color), (WIDTH + 18, y))

    label("ARTIFICIAL EVOLUTION", 20, large=True)
    label(f"{'PAUSED' if paused else 'LIVE'}  /  {speed}x  /  {sim.time:.1f}s", 52, MUTED)
    pygame.draw.rect(screen, (49, 73, 77), INFO_BUTTON, border_radius=5)
    screen.blit(font.render("i  Info", True, TEXT), (INFO_BUTTON.x + 13, INFO_BUTTON.y + 6))
    stats = sim.stats()
    for y, text in zip(
        range(90, 216, 25),
        [
            f"Population       {stats['population']} / {sim.max_population}",
            f"Births / deaths   {sim.births} / {sim.deaths}",
            f"Groups {stats['groups']}  /  generation {stats['max_generation']}",
            f"Mean segments     {stats['mean_segments']}",
            f"Mean energy       {stats['mean_energy']}",
        ],
    ):
        label(text, y)
    creature = sim.creatures.get(selected)
    label("CREATURE INSPECTOR", 237, large=True)
    if creature:
        lines = [
            f"ID {creature.id} / G{creature.group_id:03d} / gen {creature.generation}",
            f"Parent: {creature.parent_id or 'ancestral genome'}",
            f"Age: {creature.age:.1f}s",
            f"Energy: {creature.energy:.1f} / {creature.capacity:.1f}",
            f"Tissue: {creature.tissue:.1f} / {creature.genome.tissue:.1f}",
            f"Area: {creature.area:.0f} / segments: {len(creature.genome.segments)}",
            f"Mass: {creature.body.mass:.2f} / bite: {creature.strength:.2f}",
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
            "F focuses the selected creature.",
            "Body color follows genetic group.",
        ]
    for y, text in zip(range(270, 491, 22), lines):
        label(text, y, MUTED)
    for rect, title in (
        (ANALYZE_BUTTON, "Analyze pond  [A]"),
        (SAVE_BUTTON, "Save run [S]"),
        (LOAD_BUTTON, "Load run [L]"),
    ):
        pygame.draw.rect(screen, (55, 82, 53), rect, border_radius=5)
        screen.blit(font.render(title, True, TEXT), (rect.x + 12, rect.y + 8))
    for y, text in zip(
        (591, 616, 641),
        ["Space pause / Tab speed", "N step / B inspect brain", "I info / Esc quit"],
    ):
        label(text, y)
    label(notice[:35] if notice else f"Seed {sim.seed}  |  {sim.brain_source} brain", 669, MUTED)
    if show_info:
        draw_info(screen, font, heading)


def run(sim: Simulation, save_path: Path, frame_limit: int | None = None) -> Simulation:
    """Render independently; all dialogs pause without changing the user's pause state."""
    pygame.init()
    pygame.key.stop_text_input()
    try:
        screen = pygame.display.set_mode((WIDTH + PANEL, HEIGHT))
        pygame.display.set_caption("Artificial Evolution: The Evolving Pond")
        font = pygame.font.Font(None, 21)
        heading = pygame.font.Font(None, 23)
        clock = pygame.time.Clock()
        camera = Camera(sim.width, sim.height)
        paused, running, info_open = False, True, False
        selected = None
        dashboard = None
        dialog = None
        speeds, speed_index = [1, 4, 12], 0
        accumulator = 0.0
        notice, notice_until = "", 0.0
        frames = 0
        while running:
            elapsed = min(clock.tick(60) / 1000, 0.25)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if dialog is not None:
                    result = dialog.handle_event(event, sim)
                    if result is not None:
                        mode = dialog.mode
                        sim, path = result
                        dialog = None
                        pygame.key.stop_text_input()
                        if path is not None:
                            save_path = path
                            notice = f"Run {'saved' if mode == 'save' else 'loaded'}: {path.stem}"
                            notice_until = time.monotonic() + 5
                            if mode == "load":
                                selected, dashboard, info_open = None, None, False
                                camera = Camera(sim.width, sim.height)
                        accumulator = 0
                    continue
                key = event.key if event.type == pygame.KEYDOWN else None
                click = (
                    event.pos
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                    else None
                )
                if dashboard is not None:
                    if key in (pygame.K_ESCAPE, pygame.K_a, pygame.K_b) or (
                        click is not None and ANALYSIS_CLOSE.collidepoint(click)
                    ):
                        dashboard = None
                    elif key in (pygame.K_s, pygame.K_l):
                        dialog = RunDialog(
                            "save" if key == pygame.K_s else "load", save_path.parent, sim.run_name
                        )
                        pygame.key.start_text_input()
                    else:
                        selected = dashboard.handle_event(event, sim, selected)
                    continue
                if info_open:
                    if key in (pygame.K_ESCAPE, pygame.K_i) or (
                        click is not None and INFO_CLOSE.collidepoint(click)
                    ):
                        info_open = False
                    continue
                if camera.handle_event(event):
                    continue
                if key == pygame.K_f and selected in sim.creatures:
                    camera.center.update(sim.creatures[selected].body.position)
                    camera.zoom = max(camera.zoom, 1.5)
                    camera.clamp()
                elif key == pygame.K_ESCAPE:
                    running = False
                elif key == pygame.K_i or (click is not None and INFO_BUTTON.collidepoint(click)):
                    info_open = True
                    accumulator = 0
                elif key in (pygame.K_a, pygame.K_b) or (
                    click is not None and ANALYZE_BUTTON.collidepoint(click)
                ):
                    dashboard = Dashboard(page=2 if key == pygame.K_b else 0)
                    dashboard.refresh(sim)
                    accumulator = 0
                elif key in (pygame.K_s, pygame.K_l) or (
                    click is not None
                    and (SAVE_BUTTON.collidepoint(click) or LOAD_BUTTON.collidepoint(click))
                ):
                    saving = key == pygame.K_s or (
                        click is not None and SAVE_BUTTON.collidepoint(click)
                    )
                    dialog = RunDialog("save" if saving else "load", save_path.parent, sim.run_name)
                    pygame.key.start_text_input()
                    accumulator = 0
                elif key == pygame.K_SPACE:
                    paused = not paused
                    accumulator = 0
                elif key == pygame.K_TAB:
                    speed_index = (speed_index + 1) % len(speeds)
                elif key == pygame.K_n and paused:
                    sim.step()
                elif click is not None and click[0] < WIDTH:
                    point = camera.to_world(click)
                    selected = next(
                        (
                            c.id
                            for c in sim.creatures.values()
                            if any(
                                s.point_query(tuple(point)).distance <= 3 / camera.zoom
                                for s in c.shapes
                            )
                        ),
                        None,
                    )
            modal = info_open or dashboard is not None or dialog is not None
            if modal:
                camera.dragging = False
            else:
                keys = pygame.key.get_pressed()
                camera.pan(
                    (
                        (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 500 * elapsed,
                        (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * 500 * elapsed,
                    )
                )
            blocked = paused or modal
            if not blocked:
                accumulator += elapsed * speeds[speed_index]
                for _ in range(min(int(accumulator / DT), 48)):
                    sim.step()
                    accumulator -= DT
                accumulator = min(accumulator, DT * 48)
            draw(
                screen,
                sim,
                selected,
                blocked,
                speeds[speed_index],
                font,
                heading,
                notice if time.monotonic() < notice_until else "",
                info_open,
                camera,
            )
            if dashboard is not None:
                dashboard.draw(screen, sim, selected, font, heading)
            if dialog is not None:
                dialog.draw(screen, font, heading)
            pygame.display.flip()
            frames += 1
            if frame_limit is not None and frames >= frame_limit:
                break
    finally:
        pygame.key.stop_text_input()
        pygame.quit()
    return sim
