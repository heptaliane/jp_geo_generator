# jp_geo_generator

Generate a 3D-printable STL terrain model from Japanese elevation data published by the [Geospatial Information Authority of Japan (GSI)](https://www.gsi.go.jp/).

The script downloads DEM (Digital Elevation Model) tiles from GSI's [`cyberjapandata`](https://cyberjapandata.gsi.go.jp/) mesh data service, stitches together one or more tiles, and converts the elevation data into a solid, watertight STL mesh suitable for 3D printing.

## Requirements

- Python 3.12+ (managed via [uv](https://docs.astral.sh/uv/))
- Internet access to `cyberjapandata.gsi.go.jp`

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies (creates a virtual environment automatically)
uv sync
```

> `pyfqmr` (used for `--decimate_ratio`) is a compiled C++ extension. If installation fails on your platform, make sure you have a working C++ build toolchain, or omit `--decimate_ratio` (leave it at the default `1.0`) and skip adding `pyfqmr` if you don't need mesh simplification.

If you're starting from scratch (no `pyproject.toml` yet):

```bash
uv init
uv add click numpy numpy-stl requests scikit-image pyfqmr
```

## Usage

```bash
uv run main.py --x <X> --y <Y> --z <Z> [OPTIONS]
```

### Finding tile coordinates (x, y, z)

GSI's DEM tiles are addressed using standard Slippy Map tile coordinates (`x`, `y`, `z`), the same scheme used by Google Maps / OpenStreetMap.

Use GSI's official **[Tile Coordinate Checker](https://maps.gsi.go.jp/development/tileCoordCheck.html)** to find the `x`, `y`, `z` values for the area you want:

1. Open the [Tile Coordinate Checker](https://maps.gsi.go.jp/development/tileCoordCheck.html#5/35.362/138.731).
2. Pan/zoom the map to the location you want to model.
3. Click on the map — the `x`, `y`, and `z` (zoom level) values for that tile will be displayed.
4. Use those values as the `--x`, `--y`, and `--z` options below.

Note: not all zoom levels have DEM data available for every area. Zoom level 12–15 generally has the widest coverage; higher zoom levels (finer resolution) are only available for some regions.

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--x` | int | *required* | Tile X coordinate |
| `--y` | int | *required* | Tile Y coordinate |
| `--z` | int | `12` | Zoom level |
| `--offset` | int | `10` | Vertical offset (base height) added to elevation values, in the same scaled units as the model |
| `--output` | str | `geo.stl` | Output STL file path |
| `--blocks` | int int | `1 1` | Number of tiles to fetch and stitch in the X and Y directions, starting from `--x`/`--y` |
| `--size` | float | `50.0` | Target size (mm) of the longest edge of the output model |
| `--sample_rate` | int | `1` | Downsampling factor applied to the elevation grid before meshing (higher = coarser/faster) |
| `--z_exaggeration` | float | `1.0` | Multiplier applied on top of the true-to-scale vertical height. `1.0` keeps real-world x/y/z proportions (relief will look subtle, since actual terrain is quite flat relative to its horizontal extent); increase (e.g. `2.0`–`5.0`) to exaggerate relief for a more dramatic/visible 3D print |
| `--decimate_ratio` | float | `1.0` | Fraction of triangles to keep after adaptive mesh simplification (`0 < ratio < 1`). `1.0` disables simplification. Unlike `--sample_rate`, which uniformly coarsens the whole grid, this collapses triangles more in flat areas while preserving detail on ridges/peaks (uses `pyfqmr` quadric edge-collapse decimation) |

### Example

Generate a model of the area around Mt. Fuji at zoom level 14, stitching a 2×2 block of tiles, downsampled by 2×, scaled to 100mm:

```bash
uv run main.py --x 14552 --y 6451 --z 14 --blocks 2 2 --sample_rate 2 --size 100 --output fuji.stl
```

Same area, but with real-world relief exaggerated 3x so the terrain is more visually striking / easier to feel on a 3D print:

```bash
uv run main.py --x 14552 --y 6451 --z 14 --blocks 2 2 --sample_rate 2 --size 100 --z_exaggeration 3 --output fuji_exaggerated.stl
```

## How it works

1. **Download**: For each requested tile, elevation data is fetched as CSV text from GSI's `xyz/dem/{z}/{x}/{y}.txt` endpoint and cached locally (`.cache.npz`) to avoid re-downloading on subsequent runs.
2. **Stitch**: Multiple tiles (`--blocks`) are concatenated into a single elevation grid.
3. **Scale**: Elevation values (meters) are converted into the same unit as the x/y grid indices (pixels), based on each tile's real-world ground resolution at the given zoom level (`meters_per_pixel = equator_length_m / (2**z * 256)`). This keeps the model's vertical (z) axis proportional to its horizontal (x/y) axes — a `1.0` `--z_exaggeration` produces true-to-scale real-world proportions. The result is then multiplied by `--z_exaggeration` (useful since real terrain relief is often too subtle to see at 1:1 scale) and offset to create a solid base.
4. **Downsample**: The grid is optionally reduced (`--sample_rate`) using max-pooling to reduce mesh complexity. The vertical scale from step 3 is adjusted to account for this so the x/y/z ratio stays correct regardless of the sampling rate used.
5. **Mesh**: The elevation grid is triangulated into a closed, solid mesh (top surface, flat bottom, and side walls), correctly handling missing/no-data cells at tile edges.
6. **Export**: The mesh is scaled to the requested physical size (`--size`), optionally simplified (see below), and written out as an STL file.

### Reducing triangle count

Two independent knobs control the final triangle count, and can be combined:

- **`--sample_rate`**: coarsens the *input elevation grid* uniformly before meshing. Fast and simple, but reduces detail everywhere equally, including in areas with interesting terrain.
- **`--decimate_ratio`**: applies adaptive mesh simplification (quadric edge-collapse via `pyfqmr`) to the *already-built mesh*, keeping the given fraction of triangles. It collapses triangles more aggressively in flat/low-detail regions (e.g. flat sea or plains) while preserving detail in high-curvature regions (e.g. ridgelines, peaks), so it generally gives a better quality-to-triangle-count tradeoff than `--sample_rate` alone.

For example, to keep only 20% of the triangles after meshing:

```bash
uv run main.py --x 14552 --y 6451 --z 14 --blocks 2 2 --size 100 --decimate_ratio 0.2 --output fuji_simplified.stl
```

## Data Source & Attribution

Elevation data is provided by the **Geospatial Information Authority of Japan (国土地理院)** via their DEM tile service. Please refer to GSI's [terms of use](https://www.gsi.go.jp/kikakuchousei/kikakuchousei40182.html) regarding usage and attribution requirements.

## License

Add your license here.
