"""Camera transforms and pond drawing, independent of simulation state."""

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pygame

from .analysis import group_color
from .ecology import FOOD_TYPES

VIEW_WIDTH, VIEW_HEIGHT = 1000, 700
VIEWPORT = pygame.Rect(0, 0, VIEW_WIDTH, VIEW_HEIGHT)
MINIMAP = pygame.Rect(16, VIEW_HEIGHT - 126, 160, 110)


@dataclass
class Camera:
    width: int
    height: int
    center: pygame.Vector2 = field(default_factory=pygame.Vector2)
    zoom: float = 1.0
    dragging: bool = False

    def __post_init__(self):
        self.reset()

    @property
    def minimum_zoom(self):
        return min(VIEW_WIDTH / self.width, VIEW_HEIGHT / self.height)

    def reset(self):
        self.zoom = self.minimum_zoom
        self.center.update(self.width / 2, self.height / 2)

    def to_screen(self, point):
        return (pygame.Vector2(point) - self.center) * self.zoom + VIEWPORT.center

    def to_world(self, point):
        return (pygame.Vector2(point) - VIEWPORT.center) / self.zoom + self.center

    def clamp(self):
        for axis, (size, pixels) in enumerate(
            ((self.width, VIEW_WIDTH), (self.height, VIEW_HEIGHT))
        ):
            half = pixels / (2 * self.zoom)
            self.center[axis] = (
                size / 2 if half * 2 >= size else min(size - half, max(half, self.center[axis]))
            )

    def pan(self, delta):
        self.center += pygame.Vector2(delta) / self.zoom
        self.clamp()

    def zoom_at(self, point, factor):
        anchor = self.to_world(point)
        self.zoom = min(4.0, max(self.minimum_zoom, self.zoom * factor))
        self.center += anchor - self.to_world(point)
        self.clamp()

    def handle_event(self, event):
        """Return whether a camera gesture consumed the event."""
        if event.type == pygame.MOUSEWHEEL:
            mouse = pygame.mouse.get_pos()
            if VIEWPORT.collidepoint(mouse):
                self.zoom_at(mouse, 1.2 ** max(-8, min(8, event.y)))
                return True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and MINIMAP.collidepoint(event.pos):
                self.center.update(
                    (event.pos[0] - MINIMAP.x) / MINIMAP.width * self.width,
                    (event.pos[1] - MINIMAP.y) / MINIMAP.height * self.height,
                )
                self.clamp()
                return True
            if event.button in (2, 3) and VIEWPORT.collidepoint(event.pos):
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button in (2, 3):
            self.dragging = False
            return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.pan(-pygame.Vector2(event.rel))
            return True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_HOME:
                self.reset()
                return True
            if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_MINUS):
                self.zoom_at(VIEWPORT.center, 1 / 1.2 if event.key == pygame.K_MINUS else 1.2)
                return True
        elif event.type == pygame.WINDOWFOCUSLOST:
            self.dragging = False
        return False


@lru_cache(maxsize=2)
def pond_background(seed, width, height, islands):
    """Cache shorelines and banks; decorative randomness never affects evolution."""
    x, y = np.mgrid[:width, :height]
    shore = np.minimum.reduce((x, width - 1 - x, y, height - 1 - y))
    depth = np.clip(shore / 120, 0, 1)[..., None]
    edge, center = np.array((51, 76, 47)), np.array((19, 56, 51))
    surface = pygame.surfarray.make_surface((edge + (center - edge) * depth).astype(np.uint8))
    rng = np.random.default_rng(seed)
    for _ in range(80):
        px, py = int(rng.integers(40, width - 40)), int(rng.integers(40, height - 40))
        radius = int(rng.integers(10, 40))
        pygame.draw.arc(surface, (29, 67, 56), (px, py, radius * 2, radius), 0.2, 2.7, 1)
    for polygon in islands:
        pygame.draw.polygon(surface, (61, 83, 54), polygon)
        pygame.draw.lines(surface, (89, 108, 62), True, polygon, 12)
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            for fraction in np.linspace(0, 1, max(2, int(math.dist(start, end) / 14))):
                point = pygame.Vector2(start).lerp(end, fraction)
                length = int(rng.integers(8, 23))
                pygame.draw.line(surface, (123, 135, 73), point, point + (4, -length), 2)
    pygame.draw.rect(surface, (92, 106, 58), surface.get_rect(), 5)
    return surface


def draw_pond(screen, sim, camera, selected, font):
    background = pond_background(
        sim.seed,
        sim.width,
        sim.height,
        tuple(tuple(tuple(p) for p in poly) for poly in sim.islands),
    )
    old_clip = screen.get_clip()
    screen.set_clip(VIEWPORT)
    screen.fill((15, 35, 31), VIEWPORT)
    top_left, bottom_right = camera.to_world((0, 0)), camera.to_world(VIEWPORT.bottomright)
    source = pygame.Rect(
        math.floor(top_left.x),
        math.floor(top_left.y),
        math.ceil(bottom_right.x - top_left.x) + 2,
        math.ceil(bottom_right.y - top_left.y) + 2,
    ).clip(background.get_rect())
    if source.width and source.height:
        # Scale only the visible crop, never a huge image of the whole zoomed world.
        scaled = pygame.transform.scale(
            background.subsurface(source),
            (math.ceil(source.width * camera.zoom), math.ceil(source.height * camera.zoom)),
        )
        screen.blit(scaled, camera.to_screen(source.topleft))
    for food in sim.food:
        if food.energy <= 0.5:
            continue
        point = camera.to_screen((food.x, food.y))
        if not VIEWPORT.inflate(40, 40).collidepoint(point):
            continue
        radius = max(2, round(min(8, 2 + food.energy * 0.14) * camera.zoom))
        color = FOOD_TYPES[food.kind].color if food.renewable else (151, 108, 65)
        if food.kind == "weed":
            pygame.draw.line(screen, color, point - (radius, radius), point + (radius, radius), 3)
            pygame.draw.line(screen, color, point + (-radius, radius), point + (radius, -radius), 3)
        elif food.kind == "seed":
            pygame.draw.polygon(
                screen,
                color,
                [point + p for p in ((-radius, 0), (0, -radius), (radius, 0), (0, radius))],
            )
            pygame.draw.circle(screen, (113, 98, 55), point, max(1, radius // 3))
        else:
            pygame.draw.circle(screen, color, point, radius)
            if food.kind == "plankton":
                pygame.draw.circle(screen, (212, 241, 221), point, max(1, radius // 2), 1)
            elif food.renewable:
                pygame.draw.line(screen, (188, 205, 129), point - (radius, 0), point + (radius, 0))
    for creature in sim.creatures.values():
        point = camera.to_screen(creature.body.position)
        margin = creature.radius * camera.zoom * 2
        if not VIEWPORT.inflate(margin, margin).collidepoint(point):
            continue
        brightness = 0.5 + 0.5 * creature.energy / creature.capacity
        color = tuple(int(v * brightness) for v in group_color(creature.group_id))
        centers = []
        for vertices in creature.vertices:
            points = [camera.to_screen(creature.body.local_to_world(v)) for v in vertices]
            centers.append(sum(points, pygame.Vector2()) / len(points))
            pygame.draw.polygon(screen, color, points)
            pygame.draw.polygon(
                screen,
                (245, 222, 151) if creature.id == selected else (110, 151, 104),
                points,
                2 if creature.id == selected else 1,
            )
        if len(centers) > 1:
            pygame.draw.lines(screen, (139, 171, 116), False, centers, 1)
        head = creature.genome.segments[0]
        for side in (-1, 1):
            eye = camera.to_screen(
                creature.body.local_to_world((head.length * 0.28, head.width * 0.23 * side))
            )
            pygame.draw.circle(screen, (228, 232, 178), eye, max(1, round(2 * camera.zoom)))
            pygame.draw.circle(screen, (20, 38, 26), eye, max(1, round(camera.zoom)))
        mouth = camera.to_screen(creature.mouth)
        pygame.draw.circle(screen, (244, 218, 161), mouth, max(1, round(2 * camera.zoom)))
        if creature.actions[2] > 0.1 and creature.id == selected:
            pygame.draw.circle(screen, (161, 131, 88), mouth, max(2, round(7 * camera.zoom)), 1)
    screen.blit(pygame.transform.scale(background, MINIMAP.size), MINIMAP)
    for creature in sim.creatures.values():
        point = (
            MINIMAP.x + creature.body.position.x / sim.width * MINIMAP.width,
            MINIMAP.y + creature.body.position.y / sim.height * MINIMAP.height,
        )
        pygame.draw.circle(
            screen,
            (245, 222, 151) if creature.id == selected else group_color(creature.group_id),
            point,
            2,
        )
    view = pygame.Rect(
        MINIMAP.x + max(0, top_left.x) / sim.width * MINIMAP.width,
        MINIMAP.y + max(0, top_left.y) / sim.height * MINIMAP.height,
        min(sim.width, VIEW_WIDTH / camera.zoom) / sim.width * MINIMAP.width,
        min(sim.height, VIEW_HEIGHT / camera.zoom) / sim.height * MINIMAP.height,
    )
    pygame.draw.rect(screen, (224, 218, 166), view.clip(MINIMAP), 1)
    pygame.draw.rect(screen, (137, 157, 104), MINIMAP, 1)
    help_text = (
        f"{camera.zoom:.2f}x  |  Scroll zoom / right-drag or arrows pan / Home overview / F focus"
    )
    label = font.render(help_text, True, (222, 233, 205), (23, 43, 35))
    screen.blit(label, (16, 14))
    legend = "Algae: green  /  Seeds: gold  /  Waterweed: branched green  /  Plankton: cyan"
    screen.blit(font.render(legend, True, (195, 213, 171), (23, 43, 35)), (16, 38))
    screen.set_clip(old_clip)
