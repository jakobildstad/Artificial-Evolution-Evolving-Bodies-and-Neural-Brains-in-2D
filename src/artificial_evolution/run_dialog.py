"""Small native pygame save-name and saved-run chooser dialogs."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pygame

from .analysis import export_csv
from .persistence import load, save

PANEL = pygame.Rect(330, 190, 640, 320)
ACCEPT = pygame.Rect(710, 456, 110, 34)
CANCEL = pygame.Rect(835, 456, 110, 34)


def named_path(folder: Path, name: str) -> Path:
    name = name.strip()
    if not name or len(name) > 64 or not re.fullmatch(r"[\w -]+", name):
        raise ValueError("Use 1-64 letters, numbers, spaces, hyphens or underscores.")
    if name.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *[f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)],
    }:
        raise ValueError("Please choose a different run name.")
    return folder / f"{name}.json"


@dataclass
class RunDialog:
    mode: str
    folder: Path
    name: str = ""
    files: list[Path] = field(default_factory=list)
    index: int = 0
    message: str = ""
    select_all: bool = True
    replace: bool = False

    def __post_init__(self):
        if self.mode == "load":
            self.files = sorted(
                self.folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            )

    def handle_event(self, event, sim):
        """Return (world, path) when closed, or None while the dialog remains open."""
        accept = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return sim, None
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                accept = True
            elif self.mode == "save":
                if event.key == pygame.K_BACKSPACE:
                    self.name = "" if self.select_all else self.name[:-1]
                    self.select_all, self.replace = False, False
                elif event.key == pygame.K_a and getattr(event, "mod", 0) & (
                    pygame.KMOD_CTRL | pygame.KMOD_META
                ):
                    self.select_all = True
            elif event.key in (pygame.K_UP, pygame.K_DOWN) and self.files:
                self.index = (self.index + (1 if event.key == pygame.K_DOWN else -1)) % len(
                    self.files
                )
        elif event.type == pygame.TEXTINPUT and self.mode == "save":
            self.name = (("" if self.select_all else self.name) + event.text)[:64]
            self.select_all, self.replace, self.message = False, False, ""
        elif event.type == pygame.MOUSEWHEEL and self.mode == "load" and self.files:
            self.index = min(max(0, self.index - event.y), len(self.files) - 1)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if CANCEL.collidepoint(event.pos):
                return sim, None
            accept = ACCEPT.collidepoint(event.pos)
            if self.mode == "load" and 350 <= event.pos[0] <= 950 and 254 <= event.pos[1] < 414:
                index = (self.index // 5) * 5 + (event.pos[1] - 254) // 32
                if index < len(self.files):
                    self.index = index
        if accept:
            try:
                if self.mode == "load":
                    if not self.files:
                        self.message = "No saved runs in this folder yet."
                        return None
                    path = self.files[self.index]
                    return load(path), path
                path = named_path(self.folder, self.name)
                if path.exists() and not self.replace:
                    self.replace = True
                    self.message = "This run exists. Press Enter again to replace it."
                    return None
                previous_name = sim.run_name
                sim.run_name = self.name.strip()
                try:
                    save(sim, path)
                except (OSError, ValueError):
                    sim.run_name = previous_name
                    raise
                # The complete JSON is already safe if an optional report cannot be written.
                try:
                    export_csv(sim, path)
                except OSError as error:
                    print(f"Run saved, but CSV export failed: {error}")
                return sim, path
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.message = str(error)
        return None

    def draw(self, screen, font, heading):
        shade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 180))
        screen.blit(shade, (0, 0))
        pygame.draw.rect(screen, (27, 48, 40), PANEL, border_radius=12)
        screen.blit(
            heading.render(
                "NAME THIS RUN" if self.mode == "save" else "OPEN A SAVED RUN",
                True,
                (224, 231, 203),
            ),
            (350, 211),
        )
        if self.mode == "save":
            pygame.draw.rect(screen, (14, 34, 28), (350, 260, 600, 42), border_radius=4)
            if self.select_all:
                pygame.draw.rect(
                    screen, (62, 100, 71), (358, 268, min(580, font.size(self.name)[0]), 24)
                )
            # Keep the typed end visible when a long name exceeds the field width.
            visible = self.name
            while font.size(visible)[0] > 570:
                visible = visible[1:]
            screen.blit(font.render(visible + "|", True, (224, 231, 203)), (360, 272))
            for y, line in (
                (328, "Saves the world, ancestry, brains and population history."),
                (355, "Population and group CSV reports are saved beside the JSON."),
                (382, "Enter to save. Esc to cancel. Ctrl/Cmd+A selects the name."),
            ):
                screen.blit(font.render(line, True, (144, 168, 143)), (350, y))
        else:
            start = (self.index // 5) * 5
            for offset, path in enumerate(self.files[start : start + 5]):
                y = 254 + offset * 32
                if start + offset == self.index:
                    pygame.draw.rect(screen, (59, 89, 62), (350, y, 600, 31))
                screen.blit(font.render(path.stem[:55], True, (224, 231, 203)), (360, y + 6))
            if not self.files:
                screen.blit(font.render("No saved runs yet.", True, (144, 168, 143)), (350, 265))
        screen.blit(font.render(self.message[:75], True, (225, 184, 110)), (350, 423))
        for rect, label in (
            (ACCEPT, "Replace" if self.replace else self.mode.title()),
            (CANCEL, "Cancel"),
        ):
            pygame.draw.rect(screen, (61, 92, 64), rect, border_radius=5)
            screen.blit(font.render(label, True, (224, 231, 203)), (rect.x + 22, rect.y + 9))
