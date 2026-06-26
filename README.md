# droplet-impact-cv

Automated side-view image analysis for droplet impact sequences. The CLI reads
16-bit grayscale TIFF frames, detects the solid surface and impact frame, then
exports droplet spreading diameter as a function of time.

## Usage

The project uses `uv` for dependency and environment management.

```bash
uv run droplet-impact-cv sourcedata/example -o outputs/example_spreading_diameter.csv
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
  -o outputs/spreading_diameter.csv
```

Use `--surface-frame` when a frame shows clear symmetry between the droplet and
its reflection. The detected waist between the droplet and its reflection is
then used as the center height of a fixed surface line for the full sequence.
The surface line is tilted `0.3` degrees counterclockwise from horizontal. Use
`--surface-y` to manually override this calibration with the surface-line center
pixel coordinate.

The reported spreading diameter is the length of the liquid contour intersection
with the fixed surface line.

Debug overlays are written by default to `outputs/debug_overlays`. Use
`--debug-dir` to choose a different directory.

The CSV columns are:

- `frame`
- `filename`
- `time_ms`
- `diameter_px`
- `diameter_mm`
- `component_area_px`
- `surface_y_px`
- `impact_frame`

By default, `time_ms = 0` at the automatically detected impact frame and
pre-impact rows are omitted. Use `--include-pre-impact` to keep all frames, or
`--time-zero first-frame` to report time from the first image.
