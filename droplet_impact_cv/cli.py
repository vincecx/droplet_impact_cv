from __future__ import annotations

import argparse
import math
from pathlib import Path

from .analysis import analyze_sequence
from .models import (
    DEFAULT_FPS,
    DEFAULT_MIN_FOREGROUND_DELTA,
    DEFAULT_PIXEL_SIZE_MM,
    DEFAULT_SURFACE_ANGLE_DEG,
    AnalysisConfig,
)
from .output import write_csv


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def default_output_dir(input_dir: Path) -> Path:
    return Path("outputs") / input_dir.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure droplet spreading diameter from high-speed TIFF sequences."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("sourcedata/example"),
        help="Directory containing TIFF frames. Default: sourcedata/example",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: outputs/<input-folder>/spreading_diameter.csv",
    )
    parser.add_argument(
        "--max-frame",
        type=positive_int,
        default=None,
        help="Only process frames up to this 1-based frame number (inclusive). Default: no limit.",
    )
    parser.add_argument(
        "--fps",
        type=positive_float,
        default=DEFAULT_FPS,
        help="Camera frame rate in fps. Default: 8000",
    )
    parser.add_argument(
        "--pixel-size-mm",
        type=positive_float,
        default=DEFAULT_PIXEL_SIZE_MM,
        help="Physical length per pixel in mm. Default: 0.00711883341",
    )
    parser.add_argument(
        "--background-frames",
        type=int,
        default=8,
        help="Number of first frames used for median background.",
    )
    parser.add_argument(
        "--surface-y",
        type=nonnegative_int,
        default=None,
        help="Override detected surface y coordinate in pixels.",
    )
    parser.add_argument(
        "--surface-frame",
        type=nonnegative_int,
        default=None,
        help="Frame number used to calibrate surface y from the droplet/reflection waist.",
    )
    parser.add_argument(
        "--threshold",
        type=positive_float,
        default=None,
        help="Override dark foreground threshold in gray levels.",
    )
    parser.add_argument(
        "--min-foreground-delta",
        type=positive_float,
        default=DEFAULT_MIN_FOREGROUND_DELTA,
        help="Lower bound for automatic threshold. Default: 700",
    )
    parser.add_argument(
        "--min-area-px",
        type=positive_float,
        default=250,
        help="Minimum liquid component area in pixels.",
    )
    parser.add_argument(
        "--morphology-radius-px",
        type=nonnegative_int,
        default=3,
        help="Morphology radius for mask cleanup.",
    )
    parser.add_argument(
        "--measure-above-surface-px",
        type=nonnegative_int,
        default=20,
        help="Measurement window above detected surface.",
    )
    parser.add_argument(
        "--measure-below-surface-px",
        type=nonnegative_int,
        default=220,
        help="Measurement window below detected surface.",
    )
    parser.add_argument(
        "--touch-above-surface-px",
        type=nonnegative_int,
        default=20,
        help="Impact detection window above surface.",
    )
    parser.add_argument(
        "--touch-below-surface-px",
        type=nonnegative_int,
        default=170,
        help="Impact detection window below surface.",
    )
    parser.add_argument(
        "--min-touch-pixels",
        type=nonnegative_int,
        default=30,
        help="Minimum foreground pixels near surface to mark impact.",
    )
    parser.add_argument(
        "--include-pre-impact",
        action="store_true",
        help="Include frames before the detected impact frame in the CSV.",
    )
    parser.add_argument(
        "--time-zero",
        choices=("impact", "first-frame"),
        default="impact",
        help="Set time origin. Default: impact",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory for overlay PNG diagnostics. Default: outputs/<input-folder>/debug_overlays",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=25,
        help="Write one debug overlay every N frames when --debug-dir is used.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    output_dir = default_output_dir(args.input_dir)
    return AnalysisConfig(
        input_dir=args.input_dir,
        output_csv=args.output_csv or output_dir / "spreading_diameter.csv",
        fps=args.fps,
        pixel_size_mm=args.pixel_size_mm,
        background_frames=args.background_frames,
        surface_y=args.surface_y,
        surface_frame=args.surface_frame,
        surface_angle_deg=DEFAULT_SURFACE_ANGLE_DEG,
        threshold=args.threshold,
        min_foreground_delta=args.min_foreground_delta,
        min_area_px=int(args.min_area_px),
        morphology_radius_px=args.morphology_radius_px,
        measure_above_surface_px=args.measure_above_surface_px,
        measure_below_surface_px=args.measure_below_surface_px,
        touch_above_surface_px=args.touch_above_surface_px,
        touch_below_surface_px=args.touch_below_surface_px,
        min_touch_pixels=args.min_touch_pixels,
        include_pre_impact=args.include_pre_impact,
        time_zero=args.time_zero,
        debug_dir=args.debug_dir or output_dir / "debug_overlays",
        debug_every=args.debug_every,
        max_frame=args.max_frame,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    measurements = analyze_sequence(config)
    write_csv(config.output_csv, measurements)

    impact_frame = measurements[0].impact_frame if measurements else None
    valid = [row for row in measurements if not math.isnan(row.diameter_mm)]
    print(f"Wrote {len(measurements)} rows to {config.output_csv}")
    if valid:
        max_row = max(valid, key=lambda row: row.diameter_mm)
        print(f"Surface y: {valid[0].surface_y}px")
        print(f"Impact frame: {impact_frame if impact_frame is not None else 'not detected'}")
        print(f"Max diameter: {max_row.diameter_mm:.6f} mm at {max_row.time_ms:.6f} ms")
    else:
        print("No valid droplet diameter measurements were found.")
