"""Paused population charts, genetic-group summaries, and neural inspection."""

from dataclasses import dataclass, field

import numpy as np
import pygame

from .analysis import group_color, group_rows, sample
from .ecology import FOOD_TYPES
from .genome import Genome

TEXT = (220, 232, 206)
MUTED = (137, 166, 148)
GOLD = (221, 186, 112)
CLOSE = pygame.Rect(1130, 76, 90, 30)
TABS = [pygame.Rect(95 + i * 175, 115, 165, 30) for i in range(4)]
SENSOR_NAMES = [
    "Food forward",
    "Food sideways",
    "Creature forward",
    "Creature sideways",
    "Bank front",
    "Bank right",
    "Bank back",
    "Bank left",
    "Energy",
    "Tissue",
    "Forward speed",
    "Sideways speed",
    "Turning speed",
    "Age",
    "Birth cooldown",
    "Bias",
]
ACTION_NAMES = ["Thrust", "Turn", "Bite", "Reproduce"]


def text(screen, font, value, position, color=TEXT):
    screen.blit(font.render(str(value), True, color), position)


def chart(screen, font, rect, title, times, series):
    """Draw real-time coordinates, including irregularly thinned history samples."""
    text(screen, font, title, (rect.x, rect.y - 25))
    pygame.draw.rect(screen, (15, 36, 34), rect, border_radius=6)
    maximum = max(1, max((max(values, default=0) for _, _, values in series), default=1))
    start, end = (times[0], times[-1]) if times else (0, 1)
    plot = rect.inflate(-50, -36)
    for fraction in (0, 0.5, 1):
        y = plot.bottom - fraction * plot.height
        pygame.draw.line(screen, (40, 66, 56), (plot.left, y), (plot.right, y))
        text(screen, font, f"{maximum * fraction:.0f}", (rect.x + 2, y - 8), MUTED)
    for label, color, values in series:
        points = [
            (
                plot.x + (t - start) / max(end - start, 1) * plot.width,
                plot.bottom - value / maximum * plot.height,
            )
            for t, value in zip(times, values)
        ]
        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, 2)
        elif points:
            pygame.draw.circle(screen, color, points[0], 3)
    text(screen, font, f"{start:.0f}s", (plot.left, rect.bottom - 18), MUTED)
    text(screen, font, f"{end:.0f}s", (plot.right - 40, rect.bottom - 18), MUTED)


def weight_map(screen, font, matrix, title, origin, row_labels=None):
    x, y = origin
    cell = 17
    text(screen, font, title, (x, y - 25))
    hover = None
    mouse = pygame.mouse.get_pos()
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            amount = min(abs(value) / 4, 1)
            target = np.array(GOLD if value >= 0 else (78, 163, 194))
            color = tuple((np.array((29, 49, 43)) * (1 - amount) + target * amount).astype(int))
            rect = pygame.Rect(x + column * cell, y + row * cell, cell - 1, cell - 1)
            pygame.draw.rect(screen, color, rect)
            if rect.collidepoint(mouse):
                hover = f"{title}: row {row}, column {column} = {value:+.4f}"
                pygame.draw.rect(screen, TEXT, rect, 1)
        if row_labels:
            text(screen, font, row_labels[row], (x + matrix.shape[1] * cell + 8, y + row * cell))
    return hover


@dataclass
class Dashboard:
    page: int = 0
    scroll: int = 0
    group_id: int | None = None
    rows: list = field(default_factory=list)
    history: list = field(default_factory=list)

    def refresh(self, sim):
        self.rows = group_rows(sim)
        self.history = list(sim.history)
        current = sample(sim)
        if not self.history or self.history[-1]["seconds"] != sim.time:
            self.history.append(current)

    def handle_event(self, event, sim, selected):
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                self.page = event.key - pygame.K_1
            elif event.key == pygame.K_TAB:
                self.page = (self.page + 1) % 4
            elif self.page == 2 and event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                ids = list(sim.creatures)
                if ids:
                    index = ids.index(selected) if selected in ids else 0
                    selected = ids[(index + (1 if event.key == pygame.K_RIGHT else -1)) % len(ids)]
                    self.group_id = None
        elif event.type == pygame.MOUSEWHEEL and self.page == 1:
            self.scroll = min(max(0, self.scroll - event.y), max(0, len(self.rows) - 11))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, rect in enumerate(TABS):
                if rect.collidepoint(event.pos):
                    self.page = index
            if self.page == 1 and 205 <= event.pos[1] < 557 and 95 <= event.pos[0] < 1195:
                index = self.scroll + (event.pos[1] - 205) // 32
                if index < len(self.rows):
                    self.group_id = self.rows[index]["group"]
                    selected = next(
                        (c.id for c in sim.creatures.values() if c.group_id == self.group_id), None
                    )
                    self.page = 2
        return selected

    def draw(self, screen, sim, selected, font, heading):
        shade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 190))
        screen.blit(shade, (0, 0))
        pygame.draw.rect(screen, (24, 46, 40), (65, 60, 1170, 600), border_radius=12)
        text(screen, heading, f"POND OBSERVATORY  /  {sim.run_name[:55]}", (95, 80))
        pygame.draw.rect(screen, (55, 80, 64), CLOSE, border_radius=5)
        text(screen, font, "Close", (1154, 83))
        for index, name in enumerate(("1  Population", "2  Groups", "3  Brain", "4  Food")):
            pygame.draw.rect(
                screen,
                (74, 105, 77) if self.page == index else (35, 62, 50),
                TABS[index],
                border_radius=5,
            )
            text(screen, font, name, (TABS[index].x + 12, TABS[index].y + 6))
        if self.page == 0:
            self.draw_population(screen, sim, font)
        elif self.page == 1:
            self.draw_groups(screen, font)
        elif self.page == 2:
            self.draw_brain(screen, sim, selected, font)
        else:
            self.draw_food(screen, sim, font)

    def draw_food(self, screen, sim, font):
        rows = [row for row in self.history if "food_algae" in row] or [sample(sim)]
        times = [row["seconds"] for row in rows]
        for index, (kind, properties) in enumerate(FOOD_TYPES.items()):
            text(screen, font, properties.label, (100 + 270 * index, 158), properties.color)
        chart(
            screen,
            font,
            pygame.Rect(95, 220, 1090, 165),
            "Available resources by type",
            times,
            [
                (kind, properties.color, [row[f"food_{kind}"] for row in rows])
                for kind, properties in FOOD_TYPES.items()
            ],
        )
        text(screen, font, "Current members' lifetime diet (six largest living groups)", (95, 405))
        colors = {kind: properties.color for kind, properties in FOOD_TYPES.items()}
        colors["meat"] = (176, 121, 86)
        for index, row in enumerate([row for row in self.rows if row["alive"]][:6]):
            y = 439 + index * 26
            text(screen, font, f"G{row['group']:03d}", (100, y), group_color(row["group"]))
            pygame.draw.rect(screen, (53, 69, 60), (185, y, 850, 17))
            x = 185.0
            for kind, color in colors.items():
                width = row[f"{kind}_fraction"] * 850
                if width > 0:
                    pygame.draw.rect(screen, color, (round(x), y, max(1, round(width)), 17))
                x += width
            text(screen, font, row["main_food"], (1055, y), MUTED)
        text(
            screen,
            font,
            "Brown = meat / gray = no intake or unrecorded legacy diet.",
            (95, 610),
            MUTED,
        )
        text(
            screen,
            font,
            "Seeds favor bite strength; waterweed favors segments; plankton drifts.",
            (95, 632),
            MUTED,
        )

    def draw_population(self, screen, sim, font):
        times = [row["seconds"] for row in self.history]
        text(
            screen,
            font,
            f"Born {sim.births}   Died {sim.deaths}   "
            f"Living groups {len({c.group_id for c in sim.creatures.values()})}   "
            f"History samples {len(times)}",
            (95, 155),
            MUTED,
        )
        chart(
            screen,
            font,
            pygame.Rect(95, 212, 520, 155),
            "Population (green) / available plants (gold)",
            times,
            [
                ("Alive", (154, 204, 150), [r["population"] for r in self.history]),
                ("Plants", GOLD, [r["available_food"] for r in self.history]),
            ],
        )
        chart(
            screen,
            font,
            pygame.Rect(665, 212, 520, 155),
            "Mean body area (square units)",
            times,
            [("Area", (135, 189, 183), [r["mean_area"] for r in self.history])],
        )
        keys = [row["group"] for row in self.rows[:6]]
        for index, key in enumerate(keys):
            text(screen, font, f"G{key:03d}", (100 + index * 110, 386), group_color(key))
        chart(
            screen,
            font,
            pygame.Rect(95, 440, 1090, 165),
            "Group abundance (six largest groups now)",
            times,
            [
                (str(key), group_color(key), [r["groups"].get(key, 0) for r in self.history])
                for key in keys
            ],
        )
        text(
            screen,
            font,
            "Saves include this history plus population and group CSV reports.",
            (95, 628),
            MUTED,
        )

    def draw_groups(self, screen, font):
        text(
            screen,
            font,
            "Descriptive genetic groups, not biological species. Labels never affect survival.",
            (95, 155),
            MUTED,
        )
        columns = [
            (110, "Group"),
            (235, "Alive"),
            (340, "Ever born"),
            (475, "Area"),
            (600, "Segments"),
            (745, "Energy"),
            (875, "Meat %"),
            (1000, "Main food"),
        ]
        for x, label in columns:
            text(screen, font, label, (x, 180), GOLD)
        for index, row in enumerate(self.rows[self.scroll : self.scroll + 11]):
            y = 205 + index * 32
            if index % 2 == 0:
                pygame.draw.rect(screen, (29, 55, 44), (95, y, 1090, 31))
            values = [
                f"G{row['group']:03d}",
                row["alive"],
                row["total_born"],
                f"{row['mean_area']:.0f}",
                f"{row['mean_segments']:.1f}",
                f"{row['mean_energy']:.1f}",
                f"{row['meat_fraction'] * 100:.0f}%",
                row["main_food"],
            ]
            for (x, _), value in zip(columns, values):
                text(screen, font, value, (x, y + 7), group_color(row["group"]))
        text(
            screen,
            font,
            "Click a group to inspect a living brain, or its founding prototype if extinct.",
            (95, 579),
            MUTED,
        )
        text(
            screen,
            font,
            "Meat % describes current members' lifetime diet, including carrion. Scroll for more.",
            (95, 606),
            MUTED,
        )
        text(
            screen,
            font,
            "Classifier: body distance + 0.5 x neural-weight RMS; a new group starts above 0.8.",
            (95, 632),
            MUTED,
        )

    def draw_brain(self, screen, sim, selected, font):
        creature = sim.creatures.get(selected)
        if creature is None and self.group_id is None:
            creature = next(iter(sim.creatures.values()), None)
        if creature:
            genome, sensors, memory, actions = (
                creature.genome,
                creature.sensors,
                creature.memory,
                creature.actions,
            )
            title = (
                f"Creature {creature.id} / G{creature.group_id:03d}"
                f" / generation {creature.generation}"
            )
        elif self.group_id in sim.groups:
            genome = Genome.ancestral(np.random.default_rng(0))
            offset = 0
            for array in genome.arrays():
                array[:] = np.array(
                    sim.groups[self.group_id].brain[offset : offset + array.size]
                ).reshape(array.shape)
                offset += array.size
            sensors, memory, actions = np.zeros(16), np.zeros(12), np.zeros(4)
            title = f"G{self.group_id:03d} founding brain / no live activations"
        else:
            text(screen, font, "No brain available. Select a creature or group.", (95, 180), MUTED)
            return
        text(screen, font, title + "   |   Left / Right: browse living creatures", (95, 155), MUTED)
        hovered = []
        hovered.append(
            weight_map(
                screen,
                font,
                np.column_stack((genome.input_weights, genome.hidden_bias)),
                "Inputs + bias -> memory",
                (100, 215),
            )
        )
        hovered.append(
            weight_map(
                screen, font, genome.recurrent_weights, "Previous memory -> memory", (460, 215)
            )
        )
        hovered.append(
            weight_map(
                screen,
                font,
                np.column_stack((genome.output_weights, genome.output_bias)),
                "Memory + bias -> actions",
                (790, 215),
                ACTION_NAMES,
            )
        )
        for index, name in enumerate(ACTION_NAMES):
            text(screen, font, f"{name:10} {actions[index]:+.3f}", (790, 303 + index * 23), GOLD)
        text(screen, font, "Memory:", (100, 435), MUTED)
        for index, value in enumerate(memory):
            text(screen, font, f"{index}:{value:+.2f}", (190 + index * 77, 435), GOLD)
        for index, (name, value) in enumerate(zip(SENSOR_NAMES, sensors)):
            x = 100 + (index % 4) * 275
            y = 475 + (index // 4) * 32
            text(screen, font, f"{index:02} {name}  {value:+.2f}", (x, y))
            pygame.draw.rect(screen, (41, 69, 54), (x, y + 20, 235, 3))
            width = int(abs(value) * 115)
            pygame.draw.rect(
                screen,
                GOLD if value >= 0 else (78, 163, 194),
                (x + 117 if value >= 0 else x + 117 - width, y + 20, width, 3),
            )
        tooltip = next(
            (value for value in hovered if value),
            "Weights: blue negative / gold positive, scale -4 to +4. Hover a cell for its value.",
        )
        text(screen, font, tooltip, (95, 628), MUTED)
