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
  --surface-y 687 \
  --debug-dir outputs/debug_overlays \
  -o outputs/spreading_diameter.csv
```

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
