# Simulation guide

[Setup and controls](../README.md)

## Observe the pond

- **Population:** population and available-food curves, mean body area, and abundance
  histories for the six largest groups. History is sampled each simulated second.
- **Groups:** living count, total births, mean area, segment count, energy, and diet.
  Meat percentage is the lifetime intake of *current members*, including carrion;
  it does not distinguish hunting from scavenging. Main food shows the largest
  recorded intake category, without assigning a diet or species to the creature.
- **Brains:** input, recurrent, and output weight heatmaps, biases, last sensor
  readings, recurrent memory, and actions. Hover a weight for its exact value.
  Extinct groups retain a founding brain prototype, without live activations.
- **Food:** resource counts by type and stacked diet bars for the six largest living
  groups. CSV exports include each resource count and diet fraction. Old saves have
  no historical per-type diet; that portion remains unclassified.
- **Appearance:** green water, reeds, and ripples decorate solid island banks.
  Organisms retain their actual collision polygons. Body color identifies the genetic group;
  brightness shows energy. Green dots are algae, gold diamonds are seeds, branched
  green marks are waterweed, cyan dots are plankton, and brown remnants are carrion.

Groups are a simple analysis aid, **not biological species**. A newborn joins the
nearest fixed prototype if its distance is at most 0.8; otherwise it starts a new
numbered group. Distance combines normalized body measurements (area, segment
count, elongation, bending, head width) with half the RMS neural-weight difference.
IDs remain stable after extinction and across saves. Convergent descendants can
share a group. Labels never affect physics, reproduction, or neural decisions.

## Ecology and body tradeoffs

New runs use a **1600×1100** pond (about 2.5 times the original area). Two staggered
chains of solid islands form three loosely separated basins. Multiple channels
remain open, including routes wide enough for large bodies. Crossing costs travel
time and energy; islands do not teleport creatures or prevent migration. Founders,
newborns, and renewable food are placed in open water. Four terrain sensors detect
banks and islands, and terrain blocks sight and bites. Pan and zoom affect only
observation, including while paused; the physics timestep stays fixed.

Pymunk simulates a bounded top-down arena without gravity. Each creature consists
of 1–5 connected rectangular segments on one rigid body. Mutations resize, bend,
add, remove, or duplicate segments and perturb neural weights.

Small bodies cost less to maintain, while larger bodies gain stronger bites and
more reserves. Thrust scales mostly with body area, so shrinking no longer keeps
an oversized motor. Broad tissue resists weaker bites. Extra segments provide
additional energy storage, but add mass, inertia, upkeep, and birth costs. Fixed
reserve requirements make reproduction less disproportionately cheap for tiny
organisms and let parents afford mutations that add a segment.

Four renewable resources offer different mechanical opportunities. Every creature
can eat all of them; group labels never grant access or a bonus.

| Food | Energy | Respawn delay | Tradeoff / habitat preference |
| --- | --- | --- | --- |
| Soft algae | 10 | 18–32 s | Easy to eat; frequent, small meals suit low-upkeep bodies; western water |
| Armored seeds | 42 | 55–85 s | Strong bites process hard shells faster; eastern water |
| Fibrous waterweed | 30 | 35–55 s | Extra segments improve processing but add upkeep and birth cost; middle basin |
| Drifting plankton | 16 | 20–40 s | Soft food moving at 12 units/s; catching it requires movement; central latitudes |

Food slots initially choose these types with probabilities 40/20/20/20%. A slot
keeps its type, but respawns at a fresh random open-water location. Habitat biases
overlap: all four resources can occur in every basin. Partially eaten food does
not refill. Plankton drifts passively and reflects from banks; it has no brain.
The larger food count maintains roughly the original density on the larger map.
Invisible crumbs are discarded and recycled, with their energy accounted as loss.

Hardness consumes more of a mouth's bite budget when bite strength is low.
Waterweed adds a processing resistance of `5 / segments**1.5`, representing the
benefit of more digestive tissue. This is a deliberate, simple ecological
tradeoff, not a detailed digestive model. All other foods use the same hardness
rule. Larger reserves can also bridge the longer waits between rich seed meals.

A shared bite budget and 80% digestion efficiency apply to food, carrion, and live
tissue. Attacks can provide energy and remove competitors. Well-fed creatures
spend reserve energy to repair tissue, also at 80% efficiency. Starvation or tissue
below 20% causes death; remaining tissue and reserves become decaying carrion.
Upkeep increases gradually with age rather than imposing a fixed lifespan.

Reproduction requires maturity, adequate reserves and tissue, an expired cooldown,
a positive neural reproduction request, and free space. The parent pays for **all**
child tissue and starting energy. For the first **10 simulated seconds** after
birth, a living child cannot be bitten by its own parent. Unrelated creatures can
still prey on it; this protection persists across save/load. Carrion remains edible
to everyone. The energy ledger checks:

`stored energy + dissipated energy = initial energy + new plant energy`

These tradeoffs create opportunities for different forms; they do not guarantee
permanent diversity or eliminate selection pressure and extinction.

## Brains and pretraining

The NumPy recurrent network has 16 inputs, 12 memory units, and 4 tanh outputs:
forward thrust, turning, bite effort, and reproduction request. Decisions run at
10 Hz; physics holds the most recent actions between decisions.

Inputs are nearest food and creature direction/proximity in the first segment's
frame, four bank/island proximities, energy, tissue, local velocity, angular velocity,
age, cooldown, and a bias. Memory starts at zero at birth. The sensor and motor
mapping stays consistent when body segments change. The nearest visible food is
sensed; brains do not receive a food-type label. Habitat preferences and body
mechanics create opportunities for differentiation without scripted diets.

`data/ancestor.npz` is a bundled foraging prior trained on synthetic steering
examples with randomized memory states. It encourages turning toward food, slowing
near food and walls, biting, and reproduction. It is imitation learning, not
reinforcement learning; predation is not taught. The teacher exists only in
`pretrain.py`. Live actions always come from inherited neural weights.

Rebuild the model from the repository root without extra dependencies:

```sh
uv run python -m artificial_evolution.pretrain --seed 42 --steps 3000
```

The script uses NumPy gradients and Adam and records settings and held-out error
in `data/ancestor.json`. Retraining affects new pretrained worlds, not stored brains
in existing saves. Random brains remain available with `--brain random`.

## Architecture and verification

The `src/artificial_evolution/` package separates responsibilities:

| Module | Responsibility |
| --- | --- |
| `genome.py`, `creature.py` | Heritable geometry/weights, lifetime state, recurrent brain |
| `ecology.py`, `terrain.py`, `simulation.py` | Food types, islands, physics, sensing, energy transfers, fixed-step loop |
| `analysis.py` | Genetic groups, population history, CSV export |
| `analysis_view.py` | Charts, group tables, neural inspection |
| `rendering.py`, `pond_view.py`, `run_dialog.py` | Camera, pond display, controls, named saves and run chooser |
| `persistence.py`, `__main__.py` | Versioned snapshots, command-line/headless execution |
| `pretrain.py`, `data/` | Reproducible offline training and bundled weights |

Physics always uses 1/60 second. Rendering consumes a wall-clock accumulator;
heavy desktop runs slow down instead of enlarging the timestep. Batched NumPy
sensing matches the scalar reference within numerical precision; a 120-creature
benchmark on the earlier open arena measured about 6× faster sensing (not 6×
for the whole simulation). The island map adds Pymunk queries for visibility and
terrain sensing. Offscreen creatures are culled and only the visible background
crop is scaled when zoomed in.
Immutable body measurements and pond decorations are cached. Headless runs avoid
rendering costs entirely.

Tests cover inheritance, energy conservation, healing, body-size tradeoffs,
expanded offspring, parental bite protection, sensor equivalence, stable groups,
history/export, save/load, connected channels, collisions/occlusion, food tradeoffs,
drift, camera transforms, selection after zoom/pan, and desktop dialogs. Seeded
randomness is independent of appearance and analysis.
Repeating a fresh run with the same versions and platform is reproducible.
Ten-minute island runs on seeds 1–3 ended with 31, 35, and 28 creatures across
6, 5, and 7 groups; energy ledger errors stayed below 0.0000001. This is a smoke
check, not a guarantee of lasting diversity.

## Limitations

This is a research toy, not a claim that stable species or complex hunting will
evolve. Bodies are rigid compounds; overlapping segments count as tissue and mass.
Injury does not shrink collision geometry. Food has no physical collider and
food type is not directly sensed. Terrain occludes sensing; creatures do not occlude
each other. Plankton is a moving food particle, not an evolving organism.
Mechanical and energy units are arbitrary. Group thresholds, population caps, and conservative birth clearance
can bias results.

Graphs keep at most 6,000 samples, thinning older history as needed while retaining
actual timestamps. Group prototypes and ancestry grow with total evolution. Saves
store living brains and group prototypes, not every deceased individual's brain or
every past activation. Earlier snapshot versions load, but missing history cannot
be reconstructed; old extinct ancestors may have no group assignment. Loaded runs
use the current ecology rules. Pre-island saves keep their original 1000×700 open
map. Their plants become algae or seeds based on stored hardness, keeping existing
energy until first respawn. Start a new run to get the larger island map.

Pymunk contact caches are reconstructed on load, so resumed collision trajectories
may diverge slightly from uninterrupted runs. Saves are not bit-exact physics
checkpoints. The desktop window remains 1300×700; the pond supports overview-to-4×
zoom, but the window is not resizable. This engine is intended for hundreds rather
than thousands of organisms. Camera state is a viewing preference and is reset
when loading a run.
