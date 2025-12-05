# Idrisi

Idrisi is a Python terrain generator that builds game-ready heightmaps. It combines Delaunay-based meshing, river-aware levelization, and slope-constrained height solving to export 1081×1081 greyscale maps tuned for Cities: Skylines, along with intermediate visualizations you can inspect or tweak.

## Highlights
- Delaunay grid seeded by Poisson-disc sampling to avoid clustered artifacts.
- Coastline + inland sea support with optional Koch-esque noisy shores.
- River network growth with spacing rules, segmented flow, and drainage-aware “wrinkle” feeder streams for believable wrinkles and ravines.
- Slope-constrained height solver that walks from sea level up through floodplains, foothills, shoulders, and peaks, keeping gradients within user-specified ranges.
- Exports game-ready heightmap PNGs and debugging overlays (rivers, levels, wrinkling) sized for Cities: Skylines (18 km map → 1081 px import).
- Deterministic seeds via `JRandom`, plus tqdm-powered progress when solving heights.

## Quick start
```bash
pip install scipy pillow tqdm
export PYTHONPATH=src
python -m idrisi.cs --help      # see all knobs
./makemap.sh                    # generate a default Cities: Skylines map
```

Outputs land in `out/`:
- `cs.major.png`, `cs.minor.png`, `cs.revis.png`: river growth passes (major, minor, re-run).
- `cs.wrink.png`: invisible feeder stream “wrinkle” overlay used to roughen terrain.
- `cs.heights.png`: colorized preview of computed heights.
- `cs.heightmap.png`: final 1081×1081 greyscale heightmap to import into the game.

## Useful knobs (selected)
- `--separate`: average spacing of the triangulation points; larger means fewer, chunkier cells.
- `--river_segment_length`, `--river_separation`: how wiggly and how far apart main rivers may be.
- `--river_source_grade`, `--river_min_grade`: starting and minimum river slopes.
- `--wrinkle_mean_length`, `--wrinkle_dev_length`, `--wrinkle_grade`: feeder stream shape and steepness to add fine detail.
- `--shore_width`, `--shore_grade`: coastal shelf width and underwater slope.
- `--flood_plain_width/grade`, `--foothill_*`, `--midhill_*`, `--shoulder_*`, `--peak_grade`: gradient envelope for climbing away from rivers toward peaks.
- `--variance_scale`: scales X/Y variation to break up large flat areas.

Adjust any combination in `./makemap.sh` or call `python -m idrisi.cs` with your own values; ranges can be passed as `min:max` to let Idrisi pick randomized variations per run.

## How it works
- **Grid + mesh**: sample points in world space, triangulate with SciPy Delaunay, and forbid long edges to keep the mesh well-behaved.
- **Sea placement**: mark hull/simplex areas as sea; levelization then measures steps from sea across the mesh.
- **River growth**: iteratively extend river segments along descending level paths while respecting spacing, then re-levelize so land distance is measured from either rivers or sea.
- **Wrinkles**: add short feeder paths to imply unseen creeks and carve subtle gullies.
- **Height solve**: propagate slope constraints outward from water to peaks, enforcing min/max slopes per band; apply optional annealing and river “punching” to dig channels.
- **Export**: color previews for inspection plus a Cities: Skylines-ready greyscale PNG.

## Importing into Cities: Skylines
1. Run `./makemap.sh` (or your tuned command) and grab `out/cs.heightmap.png`.
2. In the Cities: Skylines map editor, import the PNG as a heightmap (1081×1081).
3. Tweak water level and detailing in-game; the generator already respects an 18 km play area with extra padding.
