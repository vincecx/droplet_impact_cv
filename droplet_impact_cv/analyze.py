from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi


DEFAULT_FPS = 8000.0
DEFAULT_PIXEL_SIZE_MM = 0.00711883341


@dataclass(frozen=True)
class AnalysisConfig:
    input_dir: Path
    output_csv: Path
    fps: float = DEFAULT_FPS
    pixel_size_mm: float = DEFAULT_PIXEL_SIZE_MM
    background_frames: int = 8
    surface_y: int | None = None
    threshold: float | None = None
    min_foreground_delta: float = 120.0
    min_area_px: int = 250
    morphology_radius_px: int = 3
    surface_search_start_px: int = 50
    surface_drop_delta: float = 100.0
    measure_above_surface_px: int = 20
    measure_below_surface_px: int = 220
    touch_above_surface_px: int = 20
    touch_below_surface_px: int = 170
    min_touch_pixels: int = 30
    include_pre_impact: bool = False
    time_zero: str = "impact"
    debug_dir: Path | None = None
    debug_every: int = 25


@dataclass(frozen=True)
class FrameMeasurement:
    frame_number: int
    filename: str
    time_ms: float
    diameter_px: float
    diameter_mm: float
    component_area_px: int
    surface_y: int
    impact_frame: int | None


def natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def find_tiff_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = sorted([*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")], key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No .tif or .tiff images found in: {input_dir}")
    return files


def read_image(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got shape {image.shape}: {path}")
    return image.astype(np.float32, copy=False)


def build_background(files: list[Path], frame_count: int) -> np.ndarray:
    if frame_count < 1:
        raise ValueError("--background-frames must be at least 1")
    selected = files[: min(frame_count, len(files))]
    stack = np.stack([read_image(path) for path in selected], axis=0)
    return np.median(stack, axis=0).astype(np.float32, copy=False)


def estimate_surface_y(background: np.ndarray, search_start_px: int, drop_delta: float) -> int:
    row_profile = np.median(background, axis=1)
    smooth = ndi.gaussian_filter1d(row_profile, sigma=3.0)
    bright_reference = float(np.percentile(smooth[: max(20, search_start_px)], 95))
    limit = bright_reference - drop_delta

    for y in range(max(0, search_start_px), len(smooth) - 20):
        window = smooth[y : y + 20]
        if np.mean(window < limit) >= 0.8:
            return y

    gradient = np.abs(np.diff(smooth))
    return int(np.argmax(gradient[search_start_px:]) + search_start_px)


def estimate_threshold(
    files: list[Path],
    background: np.ndarray,
    surface_y: int,
    background_frames: int,
    min_foreground_delta: float,
) -> float:
    sample_files = files[: min(background_frames, len(files))]
    if len(sample_files) < 2:
        return min_foreground_delta

    roi_bottom = max(10, min(surface_y - 30, background.shape[0]))
    noise_samples = []
    for path in sample_files:
        image = read_image(path)
        diff = np.abs(background[:roi_bottom] - image[:roi_bottom])
        noise_samples.append(diff.ravel())

    noise = np.concatenate(noise_samples)
    median = float(np.median(noise))
    mad = float(np.median(np.abs(noise - median)))
    sigma = 1.4826 * mad
    return max(min_foreground_delta, median + 8.0 * sigma)


def make_structure(radius_px: int) -> np.ndarray:
    radius_px = max(1, int(radius_px))
    size = radius_px * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def foreground_mask(
    image: np.ndarray,
    background: np.ndarray,
    surface_y: int,
    threshold: float,
    config: AnalysisConfig,
    structure: np.ndarray,
) -> np.ndarray:
    roi_bottom = min(image.shape[0], surface_y + config.measure_below_surface_px)
    mask = np.zeros(image.shape, dtype=np.uint8)
    dark_change = (background[:roi_bottom] - image[:roi_bottom]) >= threshold
    mask[:roi_bottom] = dark_change.astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, structure)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, structure)
    mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8)
    return mask


def component_measurement(
    mask: np.ndarray,
    surface_y: int,
    config: AnalysisConfig,
) -> tuple[float, int, tuple[int, int, int, int] | None]:
    labels, count = ndi.label(mask)
    if count == 0:
        return math.nan, 0, None

    touch_top = max(0, surface_y - config.touch_above_surface_px)
    touch_bottom = min(mask.shape[0], surface_y + config.touch_below_surface_px + 1)
    measure_top = max(0, surface_y - config.measure_above_surface_px)
    measure_bottom = min(mask.shape[0], surface_y + config.measure_below_surface_px + 1)

    best_label = 0
    best_score = -1
    best_area = 0

    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < config.min_area_px:
            continue
        touch_pixels = int(component[touch_top:touch_bottom].sum())
        score = touch_pixels * 1_000_000 + area
        if score > best_score:
            best_label = label
            best_score = score
            best_area = area

    if best_label == 0:
        return math.nan, 0, None

    component = labels == best_label
    measured = component[measure_top:measure_bottom]
    ys, xs = np.nonzero(measured)
    if len(xs) == 0:
        return math.nan, best_area, None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min() + measure_top)
    y_max = int(ys.max() + measure_top)
    diameter_px = float(x_max - x_min + 1)
    return diameter_px, best_area, (x_min, y_min, x_max, y_max)


def touches_surface(mask: np.ndarray, surface_y: int, config: AnalysisConfig) -> bool:
    y0 = max(0, surface_y - config.touch_above_surface_px)
    y1 = min(mask.shape[0], surface_y + config.touch_below_surface_px + 1)
    return int(mask[y0:y1].sum()) >= config.min_touch_pixels


def write_debug_overlay(
    path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    surface_y: int,
    bbox: tuple[int, int, int, int] | None,
    frame_number: int,
    diameter_mm: float,
) -> None:
    lo, hi = np.percentile(image, [0.5, 99.5])
    gray = np.clip((image - lo) / max(1.0, hi - lo) * 255.0, 0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = (0, 0, 255)
    cv2.line(overlay, (0, surface_y), (overlay.shape[1] - 1, surface_y), (0, 255, 255), 1)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
    text = f"frame={frame_number} diameter_mm={diameter_mm:.4f}"
    cv2.putText(overlay, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def analyze_sequence(config: AnalysisConfig) -> list[FrameMeasurement]:
    files = find_tiff_files(config.input_dir)
    background = build_background(files, config.background_frames)
    surface_y = (
        int(config.surface_y)
        if config.surface_y is not None
        else estimate_surface_y(background, config.surface_search_start_px, config.surface_drop_delta)
    )
    threshold = (
        float(config.threshold)
        if config.threshold is not None
        else estimate_threshold(files, background, surface_y, config.background_frames, config.min_foreground_delta)
    )
    structure = make_structure(config.morphology_radius_px)

    pending: list[tuple[int, Path, float, int, tuple[int, int, int, int] | None, bool]] = []
    impact_frame: int | None = None

    for frame_number, file_path in enumerate(files, start=1):
        image = read_image(file_path)
        mask = foreground_mask(image, background, surface_y, threshold, config, structure)
        diameter_px, area, bbox = component_measurement(mask, surface_y, config)
        is_touching = touches_surface(mask, surface_y, config)
        if impact_frame is None and is_touching and not math.isnan(diameter_px):
            impact_frame = frame_number
        pending.append((frame_number, file_path, diameter_px, area, bbox, is_touching))

        if config.debug_dir is not None and (
            frame_number == 1
            or frame_number % config.debug_every == 0
            or (is_touching and abs(frame_number - (impact_frame or frame_number)) <= 2)
        ):
            diameter_mm = diameter_px * config.pixel_size_mm if not math.isnan(diameter_px) else math.nan
            debug_path = config.debug_dir / f"{frame_number:06d}.png"
            write_debug_overlay(debug_path, image, mask, surface_y, bbox, frame_number, diameter_mm)

    measurements: list[FrameMeasurement] = []
    for frame_number, file_path, diameter_px, area, _bbox, _is_touching in pending:
        if impact_frame is None:
            frame_offset = frame_number - 1
        elif config.time_zero == "impact":
            frame_offset = frame_number - impact_frame
        else:
            frame_offset = frame_number - 1

        if not config.include_pre_impact and impact_frame is not None and frame_number < impact_frame:
            continue

        diameter_mm = diameter_px * config.pixel_size_mm if not math.isnan(diameter_px) else math.nan
        measurements.append(
            FrameMeasurement(
                frame_number=frame_number,
                filename=file_path.name,
                time_ms=frame_offset / config.fps * 1000.0,
                diameter_px=diameter_px,
                diameter_mm=diameter_mm,
                component_area_px=area,
                surface_y=surface_y,
                impact_frame=impact_frame,
            )
        )

    return measurements


def write_csv(path: Path, measurements: Iterable[FrameMeasurement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "filename",
                "time_ms",
                "diameter_px",
                "diameter_mm",
                "component_area_px",
                "surface_y_px",
                "impact_frame",
            ]
        )
        for row in measurements:
            writer.writerow(
                [
                    row.frame_number,
                    row.filename,
                    f"{row.time_ms:.6f}",
                    "" if math.isnan(row.diameter_px) else f"{row.diameter_px:.3f}",
                    "" if math.isnan(row.diameter_mm) else f"{row.diameter_mm:.9f}",
                    row.component_area_px,
                    row.surface_y,
                    "" if row.impact_frame is None else row.impact_frame,
                ]
            )


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure droplet spreading diameter from high-speed TIFF sequences.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path(".sourcedata/example"),
        help="Directory containing TIFF frames. Default: .sourcedata/example",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_csv",
        type=Path,
        default=Path("outputs/spreading_diameter.csv"),
        help="Output CSV path. Default: outputs/spreading_diameter.csv",
    )
    parser.add_argument("--fps", type=positive_float, default=DEFAULT_FPS, help="Camera frame rate in fps. Default: 8000")
    parser.add_argument(
        "--pixel-size-mm",
        type=positive_float,
        default=DEFAULT_PIXEL_SIZE_MM,
        help="Physical length per pixel in mm. Default: 0.00711883341",
    )
    parser.add_argument("--background-frames", type=int, default=8, help="Number of first frames used for median background.")
    parser.add_argument("--surface-y", type=nonnegative_int, default=None, help="Override detected surface y coordinate in pixels.")
    parser.add_argument("--threshold", type=positive_float, default=None, help="Override dark foreground threshold in gray levels.")
    parser.add_argument("--min-foreground-delta", type=positive_float, default=120.0, help="Lower bound for automatic threshold.")
    parser.add_argument("--min-area-px", type=positive_float, default=250, help="Minimum liquid component area in pixels.")
    parser.add_argument("--morphology-radius-px", type=nonnegative_int, default=3, help="Morphology radius for mask cleanup.")
    parser.add_argument("--measure-above-surface-px", type=nonnegative_int, default=20, help="Measurement window above detected surface.")
    parser.add_argument("--measure-below-surface-px", type=nonnegative_int, default=220, help="Measurement window below detected surface.")
    parser.add_argument("--touch-above-surface-px", type=nonnegative_int, default=20, help="Impact detection window above surface.")
    parser.add_argument("--touch-below-surface-px", type=nonnegative_int, default=170, help="Impact detection window below surface.")
    parser.add_argument("--min-touch-pixels", type=nonnegative_int, default=30, help="Minimum foreground pixels near surface to mark impact.")
    parser.add_argument("--include-pre-impact", action="store_true", help="Include frames before the detected impact frame in the CSV.")
    parser.add_argument(
        "--time-zero",
        choices=("impact", "first-frame"),
        default="impact",
        help="Set time origin. Default: impact",
    )
    parser.add_argument("--debug-dir", type=Path, default=None, help="Optional directory for overlay PNG diagnostics.")
    parser.add_argument("--debug-every", type=int, default=25, help="Write one debug overlay every N frames when --debug-dir is used.")
    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        fps=args.fps,
        pixel_size_mm=args.pixel_size_mm,
        background_frames=args.background_frames,
        surface_y=args.surface_y,
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
        debug_dir=args.debug_dir,
        debug_every=args.debug_every,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    measurements = analyze_sequence(config)
    write_csv(config.output_csv, measurements)

    impact_frame = measurements[0].impact_frame if measurements else None
    valid = [row for row in measurements if not math.isnan(row.diameter_mm)]
    if valid:
        max_row = max(valid, key=lambda row: row.diameter_mm)
        print(f"Wrote {len(measurements)} rows to {config.output_csv}")
        print(f"Surface y: {valid[0].surface_y}px")
        print(f"Impact frame: {impact_frame if impact_frame is not None else 'not detected'}")
        print(f"Max diameter: {max_row.diameter_mm:.6f} mm at {max_row.time_ms:.6f} ms")
    else:
        print(f"Wrote {len(measurements)} rows to {config.output_csv}")
        print("No valid droplet diameter measurements were found.")
