# droplet-impact-cv

Automated side-view image analysis for droplet impact sequences. The CLI reads
16-bit grayscale TIFF frames, detects the substrate surface and impact frame, then
exports droplet spreading diameter as a function of time.

Each TIFF filename must end with a six-digit frame number immediately before the
extension. For example, `capture_000005.tif` is reported as frame 5 even when it
is the first file in the input directory.

## Usage

The project uses `uv` for dependency and environment management.

```bash
uv run droplet-impact-cv sourcedata/example
```

Default physical parameters:

- frame rate: `8000` fps
- pixel size: `0.00711883341` mm/px

Common overrides:

```bash
uv run droplet-impact-cv path/to/tiff_frames \
  --fps 8000 \
  --pixel-size-mm 0.00711883341 \
  --surface-frame 61 \
  -o outputs/example/spreading_diameter.csv \
  --debug-every 1
```

Use `--max-frame 120` to process only files whose filename frame number is at
most 120. If omitted, the full sequence is processed.

Use `--surface-frame` when a frame shows clear symmetry between the droplet and
its reflection. The detected waist between the droplet and its reflection is
then used as the center height of a fixed surface line for the full sequence.

Surface angles are measured clockwise from horizontal, with a default of `-0.6`
degrees (that is, `0.6` degrees counterclockwise). Use
`--surface-y` to manually override this calibration with the surface-line center
pixel coordinate.

## Outputs

By default, all outputs are grouped under a folder named after the input
folder. e.g., For the command `uv run droplet-impact-cv sourcedata/example`, the
CSV is written to `outputs/example/spreading_diameter.csv` and diagnostic
overlays are written to `outputs/example/debug_overlays`. Use `--debug-dir` to
choose a different directory for debug overlays.

The reported spreading diameter is the length of the liquid contour intersection
with the fixed surface line.

The CSV columns are:

- `filename`
- `frame`
- `time_ms`
- `diameter_px`
- `diameter_mm`
- `impact_frame`
- `surface_y_px`
- `component_area_px`
- `fps`
- `pixel-size-mm`
- `surface-frame`

By default, `time_ms = 0` at the automatically detected impact frame and
pre-impact rows are omitted. Use `--include-pre-impact` to keep all frames, or
`--time-zero first-frame` to report time from the first image.

## Project structure

- `droplet_impact_cv/cli.py`: command-line parsing and entry point
- `droplet_impact_cv/models.py`: configuration and result data models
- `droplet_impact_cv/imaging.py`: TIFF loading, segmentation, and surface calibration
- `droplet_impact_cv/analysis.py`: sequence-level analysis workflow
- `droplet_impact_cv/visualization.py`: debug overlay rendering
- `droplet_impact_cv/output.py`: CSV serialization
- `tests/`: unit tests grouped by the module under test

## Development

Synchronize the environment and run the test suite with:

```bash
uv sync
uv run python -m unittest discover -v
```
