# Artificial Evolution: Evolving Bodies and Neural Brains in 2D

A desktop artificial-life pond where bodies and recurrent neural brains evolve
through mutation, competition, and reproduction. Every founder shares one ancestral
genome. There are no predefined species or scripted predator controllers.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync
uv run artificial-evolution
```

uv manages Python 3.12, the virtual environment, and the dependencies pinned in
`uv.lock`. The only runtime libraries are Pymunk, pygame-ce, and NumPy.

```sh
uv run artificial-evolution --brain random                 # untrained ancestor
uv run artificial-evolution --plants 150                   # scarcer food
uv run artificial-evolution --save-dir saves/experiments    # desktop run folder
uv run artificial-evolution --headless --seed 1 --steps 36000 \
  --name "Reed pond" --save "saves/Reed pond.json"
uv run artificial-evolution --load "saves/Reed pond.json"
uv run pytest
uv run ruff check .
```

Defaults: pretrained brain, 24 founders, 220 renewable food slots, and a
180-creature population cap. Headless runs do not initialize pygame;
`--steps` counts additional 1/60-second
steps. Initialization flags such as `--brain`, `--seed`, and `--plants` apply to new
worlds. Loaded runs retain their stored map, organisms, and food. Extinct populations
are not automatically repopulated.

## Controls

| Control | Action |
| --- | --- |
| Space / Tab | Pause/resume / cycle requested speed: 1×, 4×, 12× |
| N | One physics step while paused |
| Left click | Inspect a creature; click the minimap to look elsewhere |
| Mouse wheel / + / - | Zoom (the wheel anchors the world under the cursor) |
| Right/middle drag / arrow keys | Pan the camera |
| Home / F | Fit the whole pond / focus the selected creature |
| Analyze pond / A | Open population charts, groups, and brain analysis |
| B | Open the selected creature's brain |
| Info / I | Explain physical traits and ecology |
| Save run / S | Enter a run name; Enter saves, Esc cancels |
| Load run / L | Choose a saved run with clicks, arrows, or scrolling; Enter opens |
| Esc | Close the current overlay, or quit the simulation |

Overlays pause the world and preserve the previous pause state. In analysis,
1/2/3/4 or Tab switches pages. Click a group to inspect a member's brain. Left/Right
browses living brains; scrolling browses longer group lists.

Saving writes `Your run.json`, `Your run.population.csv`, and `Your run.groups.csv`
in the chosen folder. The JSON contains the world, current brains, ancestry, group
prototypes, and population history. CSV files are convenient for outside analysis.
An existing name requires a second confirmation before replacement. Ctrl/Cmd+A
selects the proposed name. `--save PATH` also saves on exit, including CSV reports.

## The pond

New runs use a **1600×1100** map with islands and open channels between three
basins. Migration requires navigation and energy. Islands block movement, sight,
and bites. Old saves keep their original smaller, open map; start a new run to
explore the islands.

| Food | Energy | Respawn | Mechanical opportunity |
| --- | --- | --- | --- |
| Soft algae | 10 | 18–32 s | Small, easy meals suit low-upkeep bodies |
| Armored seeds | 42 | 55–85 s | Strong bites process hard shells faster |
| Fibrous waterweed | 30 | 35–55 s | Extra segments improve processing |
| Drifting plankton | 16 | 20–40 s | Moving food rewards effective movement |

Food respawns at random open-water locations with overlapping habitat preferences.
All creatures can eat every food type, live tissue, and carrion. Parents cannot bite
their own living offspring for **10 simulated seconds** after birth. Reproduction
transfers the cost of all offspring tissue and reserves from the parent.

Bodies mutate in size, shape, and segment count alongside recurrent neural weights.
Larger bodies gain bite strength and reserves but pay greater upkeep and birth
costs. Founders share one pretrained foraging brain by default; `--brain random`
starts with an untrained ancestor. Live behavior always comes from inherited brains.

The **Analyze pond** dashboard shows population graphs, genetic groups, neural
weights and activations, food availability, and group diets. Groups are descriptive
clusters, not predefined biological species; labels never alter behavior. Colors
identify groups and brightness shows energy. Diet statistics describe the lifetime
intake of current group members, including scavenged meat.

## Architecture and checks

The compact `src/artificial_evolution/` package uses:

- `genome.py`, `creature.py`: inherited bodies, brains, and lifetime state.
- `ecology.py`, `terrain.py`, `simulation.py`: physics, resources, sensing, fixed steps.
- `analysis.py`, `analysis_view.py`: group classification, history, graphs, CSV export.
- `rendering.py`, `pond_view.py`, `run_dialog.py`: desktop controls, camera, named runs.
- `persistence.py`, `__main__.py`: versioned saves and command-line execution.
- `pretrain.py`, `data/`: reproducible offline training and bundled ancestral weights.

Physics advances at 60 Hz and brain decisions at 10 Hz, independently of rendering.
Tests cover inheritance, energy accounting, offspring protection, save/load,
resource tradeoffs, terrain connectivity, sensors, analysis, and desktop controls.
Ten-minute runs on three seeds retained 28–35 creatures across 5–7 groups; this is
an experiment, not a guarantee of lasting diversity.

## Limitations

Bodies are rigid compounds, food has no physical collider, and plankton is a drifting
particle. Brains sense the nearest visible food but not its type. Units and ecological
tradeoffs are simplified. Extinction and selection for small bodies remain possible.
The 1300×700 window supports camera zoom but is not resizable. Save/load reconstructs
Pymunk contacts, so resumed collision trajectories may differ slightly. Camera state
resets on load; old saves lack historical food-type intake.

See the [simulation guide](docs/simulation.md) for detailed mechanics, analysis,
pretraining, persistence, and verification notes.

## License

[MIT](LICENSE), copyright 2026 Jakob Ildstad.
