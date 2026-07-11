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

If you're starting from scratch (no `pyproject.toml` yet):

```bash
uv init
uv add click numpy numpy-stl requests scikit-image
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

### Example

Generate a model of the area around Mt. Fuji at zoom level 14, stitching a 2×2 block of tiles, downsampled by 2×, scaled to 100mm:

```bash
uv run main.py --x 14552 --y 6451 --z 14 --blocks 2 2 --sample_rate 2 --size 100 --output fuji.stl
```

## How it works

1. **Download**: For each requested tile, elevation data is fetched as CSV text from GSI's `xyz/dem/{z}/{x}/{y}.txt` endpoint and cached locally (`.cache.npz`) to avoid re-downloading on subsequent runs.
2. **Stitch**: Multiple tiles (`--blocks`) are concatenated into a single elevation grid.
3. **Scale**: Elevation values are scaled to be consistent with the tile's real-world ground resolution at the given zoom level, then offset to create a solid base.
4. **Downsample**: The grid is optionally reduced (`--sample_rate`) using max-pooling to reduce mesh complexity.
5. **Mesh**: The elevation grid is triangulated into a closed, solid mesh (top surface, flat bottom, and side walls), correctly handling missing/no-data cells at tile edges.
6. **Export**: The final mesh is scaled to the requested physical size and written out as an STL file.

## Data Source & Attribution

Elevation data is provided by the **Geospatial Information Authority of Japan (国土地理院)** via their DEM tile service. Please refer to GSI's [terms of use](https://www.gsi.go.jp/kikakuchousei/kikakuchousei40182.html) regarding usage and attribution requirements.

## License

Add your license here.
