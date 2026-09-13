# Artificial Evolution: Evolving Bodies and Neural Brains in 2D

A runnable desktop artificial-life experiment: creatures evolve their bodies and
small recurrent neural brains through mutation, competition, and reproduction.
There are no predefined species or predator controllers. Every founder receives
an independent copy of the same randomly initialized ancestral genome.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync
uv run artificial-evolution
```

`.python-version` selects Python 3.12; uv manages the interpreter and `.venv`.
`pyproject.toml` declares dependencies and `uv.lock` pins their resolved versions.
The desktop needs a graphical session. Headless runs do not initialize pygame:

```sh
uv run artificial-evolution --headless --seed 1 --steps 18000 --save saves/run.json
uv run artificial-evolution --load saves/run.json
uv run artificial-evolution --headless --load saves/run.json --steps 600
uv run python -m artificial_evolution --seed 2 --population 30 --plants 240
uv run pytest
uv run ruff check .
```

Verified on macOS with Python 3.12: eight tests pass, the native window runs, and a
seed-1 run of 18,000 steps produces 41 births, reaches generation 3, and leaves
7 creatures alive. Its energy balance error is below `1.3e-7` units. These results
are a smoke check, not evidence of stable long-term diversity.

Headless `--steps` counts **additional** 1/60-second steps and prints JSON statistics.
`--max-population` defaults to 180. A loaded snapshot supplies all world settings;
initialization flags such as `--seed` only apply to new worlds. There is no automatic
repopulation after extinction. Try different seeds and resource densities.

## Controls

| Control | Action |
| --- | --- |
| Space | Pause/resume |
| Tab | Cycle requested speed: 1×, 4×, 12× |
| N | Advance one physics step while paused |
| Left click | Inspect a creature, its actions, and recent ancestor IDs |
| S / L | Save/load `saves/world.json` (override with `--save-path`) |
| Esc / close window | Quit; `--save PATH` writes the final state |

Green dots are renewable plants; rust dots are carrion. The yellow mouth marker
shows the sensor/motor frame. Creature color indicates generation, with brightness
showing energy. The panel reports population, births, deaths, and body statistics.
Snapshots retain parent IDs, birth/death times, and generations for every creature,
including dead ancestors. Founders have no parent and share the stored ancestor genome.

## Ecosystem and evolution

- Pymunk simulates a bounded, top-down arena without gravity. Each creature is one
  rigid body composed of 1–5 connected rectangles. Mutations resize, bend, add,
  remove, or duplicate segments. Geometry affects collisions, mass, rotational
  inertia, tissue, energy capacity, upkeep, and motor cost. The first segment
  defines the mouth, forward axis, and motor scale, regardless of segment count.
- A NumPy recurrent network has 16 inputs, 12 memory units, and 4 tanh outputs:
  signed thrust, signed turning, positive bite effort, and reproduction request.
  It runs at 10 Hz; actions are held between decisions. Inputs are nearest edible
  resource and creature direction/proximity (two body-relative channels each),
  four body-relative wall proximities, energy, tissue, local forward/lateral
  velocity, angular velocity, age, reproduction cooldown, and a constant bias.
  Resource sensing combines plants and carrion. Memory starts at zero at birth.
- Founders share random weights and general activity biases; they have no
  hand-written food-seeking or hunting rules. Offspring inherit sparse Gaussian
  weight mutations and occasional geometry mutations. There is no training,
  fitness score, crossover, speciation, or script that chooses a survival strategy.
- Positive bite effort can consume nearby plants, carrion, or another creature's
  tissue. One bite budget is shared across all targets. Digestion transfers 80%
  into reserves and dissipates the rest. Feeding order rotates each step. Tissue
  damage can kill a creature; attacking can yield energy and remove a competitor.
- A mature creature requesting reproduction needs sufficient energy, healthy
  tissue, an expired cooldown, and free space. The parent pays **all** child tissue
  and starting reserve energy. Unaffordable or crowded births are skipped.
- Upkeep and motor activity drain reserves. Starvation, severe tissue loss, or
  240 seconds of age cause death; remaining tissue and energy become decaying
  carrion. Plants regrow to a cap. The reported energy balance error checks
  `stored + dissipated = initial + plant growth`, including reproduction and death.

## Architecture

The `src/artificial_evolution/` package keeps responsibilities in simple modules:

| Module | Responsibility |
| --- | --- |
| `genome.py` | Segment geometry, ancestral weights, mutation |
| `creature.py` | Lifetime state and recurrent brain |
| `ecology.py` | Pymunk bodies, sensing, digestion |
| `simulation.py` | Fixed-step loop, feeding, births, deaths, statistics, ancestry |
| `rendering.py` | pygame-ce display, inspection, input, wall-clock accumulator |
| `persistence.py` | Versioned JSON snapshots and RNG restoration |
| `__main__.py` | CLI and headless execution |

Physics always uses 1/60 second. Rendering consumes an accumulator independently;
overloaded desktop runs slow down instead of increasing the timestep. Randomness
comes from one seeded NumPy generator. Repeating a fresh run with the same seed,
versions, and platform is reproducible. Snapshots include genomes, neural memory,
actions, positions, velocities, ecology, ancestry, counters, and RNG state.

## Simplifications and limitations

This is a small research toy, not a claim that complex strategies will reliably
evolve. A random ancestor may be poor at foraging, and extinction is possible.
Long-lived diversity and specialized predation are outcomes to investigate, not
guaranteed features. All animals share one mouth and two body-level motors; bodies
are rigid compounds without articulated limbs. Rectangles can overlap at bends;
overlap counts as tissue and mass. Tissue damage reduces health but does not shrink
collision shapes or change mass before death. Tissue does not heal.

Sensing uses the nearest resource/creature within 180 units, without occlusion.
Food is a point with a small feeding radius, not a physical collider. Plants are
stationary and renew in place. Nearby food is consumed before live tissue; there
is no dietary trait. Thrust, energy costs, aging, and digestion use arbitrary game
units, not a biological or thermodynamic calibration. Crowd limits and conservative
birth clearance bound cost and may bias selection against large bodies.

Pymunk contact caches are reconstructed on load, so resumed collision trajectories
can diverge slightly from uninterrupted runs; saves are not bit-exact physics
checkpoints. JSON saves are intended for this version's own snapshots. Sensing and
feeding use straightforward scans, suitable for hundreds rather than thousands of
creatures; ancestry and snapshots grow with total births. The fixed 1300×700 window
has no zoom or resizing. No assets, server, GPU, or external services are required.

Physics construction follows the [Pymunk API](https://www.pymunk.org/en/latest/pymunk.html);
the only runtime libraries are Pymunk, pygame-ce, and NumPy.
