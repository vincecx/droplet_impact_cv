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
DEFAULT_SURFACE_ANGLE_DEG = 0.4
DEFAULT_MIN_FOREGROUND_DELTA = 700.0
DEBUG_MASK_ALPHA = 0.2
SURFACE_MASK_BELOW_TOLERANCE_PX = 4


@dataclass(frozen=True)
class AnalysisConfig:
    input_dir: Path
    output_csv: Path
    fps: float = DEFAULT_FPS
    pixel_size_mm: float = DEFAULT_PIXEL_SIZE_MM
    background_frames: int = 8
    surface_y: int | None = None
    surface_frame: int | None = None
    surface_angle_deg: float = DEFAULT_SURFACE_ANGLE_DEG
    threshold: float | None = None
    min_foreground_delta: float = DEFAULT_MIN_FOREGROUND_DELTA
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
    debug_dir: Path | None = Path("outputs/debug_overlays")
    debug_every: int = 25
    max_frame: int | None = None


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
    fps: float
    pixel_size_mm: float
    surface_frame: int | None


@dataclass(frozen=True)
class ComponentMeasurement:
    diameter_px: float
    area_px: int
    bbox: tuple[int, int, int, int] | None
    mask: np.ndarray


@dataclass(frozen=True)
class SurfaceLine:
    center_y: float
    angle_deg: float = DEFAULT_SURFACE_ANGLE_DEG

    @property
    def slope(self) -> float:
        # Image y increases downward. A visually counterclockwise line therefore has negative dy/dx.
        return -math.tan(math.radians(self.angle_deg))

    def y_at(self, x: np.ndarray | float, width: int) -> np.ndarray | float:
        center_x = (width - 1) / 2.0
        return self.center_y + self.slope * (x - center_x)

    def center_y_int(self) -> int:
        return int(round(self.center_y))


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


def clean_binary_mask(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, structure)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, structure)
    return ndi.binary_fill_holes(cleaned > 0).astype(np.uint8)


def foreground_mask(
    image: np.ndarray,
    background: np.ndarray,
    surface_line: SurfaceLine,
    threshold: float,
    config: AnalysisConfig,
    structure: np.ndarray,
) -> np.ndarray:
    yy, xx = np.indices(image.shape)
    distance_from_surface = yy - surface_line.y_at(xx, image.shape[1])
    roi = distance_from_surface <= config.measure_below_surface_px
    mask = np.zeros(image.shape, dtype=np.uint8)
    dark_change = ((background - image) >= threshold) & roi
    mask[dark_change] = 1

    return clean_binary_mask(mask, structure)


def full_foreground_mask(
    image: np.ndarray,
    background: np.ndarray,
    threshold: float,
    structure: np.ndarray,
) -> np.ndarray:
    dark_change = (background - image) >= threshold
    return clean_binary_mask(dark_change, structure)


def largest_component(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)

    best_label = 0
    best_area = 0
    for label in range(1, count + 1):
        area = int(np.sum(labels == label))
        if area >= min_area_px and area > best_area:
            best_label = label
            best_area = area

    if best_label == 0:
        return np.zeros(mask.shape, dtype=bool)
    return labels == best_label


def component_bbox(component: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def select_liquid_component(
    mask: np.ndarray,
    surface_line: SurfaceLine,
    config: AnalysisConfig,
) -> tuple[np.ndarray, int]:
    labels, count = ndi.label(mask)
    empty = np.zeros(mask.shape, dtype=bool)
    if count == 0:
        return empty, 0

    yy, xx = np.indices(mask.shape)
    distance_from_surface = yy - surface_line.y_at(xx, mask.shape[1])
    touch_region = (
        (distance_from_surface >= -config.touch_above_surface_px)
        & (distance_from_surface <= config.touch_below_surface_px)
    )
    above_touch_region = (
        (distance_from_surface >= -config.touch_above_surface_px)
        & (distance_from_surface <= 0)
    )

    best_label = 0
    best_score = -1
    best_area = 0

    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < config.min_area_px:
            continue
        above_touch_pixels = int((component & above_touch_region).sum())
        if above_touch_pixels < config.min_touch_pixels:
            continue
        touch_pixels = int((component & touch_region).sum())
        score = touch_pixels * 1_000_000 + area
        if score > best_score:
            best_label = label
            best_score = score
            best_area = area

    if best_label == 0:
        return empty, 0
    return labels == best_label, best_area


def longest_true_run(active: np.ndarray, close_size: int = 5) -> tuple[int, int] | None:
    active = np.asarray(active, dtype=bool)
    if close_size > 1:
        active = ndi.binary_closing(active, structure=np.ones(close_size, dtype=bool))
    if not active.any():
        return None

    labels, count = ndi.label(active)
    best_label = 0
    best_width = 0
    for label in range(1, count + 1):
        xs = np.flatnonzero(labels == label)
        width = int(xs[-1] - xs[0] + 1)
        if width > best_width:
            best_label = label
            best_width = width

    xs = np.flatnonzero(labels == best_label)
    return int(xs[0]), int(xs[-1])


def surface_band_mask(shape: tuple[int, int], surface_line: SurfaceLine, below_tolerance_px: int) -> np.ndarray:
    yy, xx = np.indices(shape)
    distance_from_surface = yy - surface_line.y_at(xx, shape[1])
    return distance_from_surface <= below_tolerance_px


def spreading_profile_on_surface(
    component: np.ndarray,
    surface_line: SurfaceLine,
    half_width_px: int = 1,
) -> np.ndarray:
    height, width = component.shape
    xs = np.arange(width)
    ys = np.rint(surface_line.y_at(xs, width)).astype(int)
    profile = np.zeros(width, dtype=bool)
    for offset in range(-half_width_px, half_width_px + 1):
        sample_y = ys + offset
        valid = (sample_y >= 0) & (sample_y < height)
        profile[valid] |= component[sample_y[valid], xs[valid]]
    return profile


def component_measurement(
    mask: np.ndarray,
    surface_line: SurfaceLine,
    config: AnalysisConfig,
) -> ComponentMeasurement:
    component, area = select_liquid_component(mask, surface_line, config)
    if area == 0:
        return ComponentMeasurement(math.nan, 0, None, component)

    # Keep the measured contour tied to foreground pixels. Polygon approximation
    # can cut across concave edges or extend beyond the actual liquid boundary.
    component &= surface_band_mask(component.shape, surface_line, SURFACE_MASK_BELOW_TOLERANCE_PX)
    clipped_component = largest_component(component, config.min_area_px)
    if clipped_component.any():
        component = clipped_component
    area = int(component.sum())
    profile = spreading_profile_on_surface(component, surface_line)
    bounds = longest_true_run(profile)
    if bounds is None:
        return ComponentMeasurement(math.nan, area, None, component)

    x_min, x_max = bounds
    y_min = int(round(surface_line.y_at(x_min, component.shape[1])))
    y_max = int(round(surface_line.y_at(x_max, component.shape[1])))
    diameter_px = float((x_max - x_min + 1) * math.sqrt(1.0 + surface_line.slope**2))
    return ComponentMeasurement(diameter_px, area, (x_min, y_min, x_max, y_max), component)


def touches_surface(mask: np.ndarray, surface_line: SurfaceLine, config: AnalysisConfig) -> bool:
    profile = spreading_profile_on_surface(mask, surface_line)
    bounds = longest_true_run(profile)
    if bounds is None:
        return False
    return (bounds[1] - bounds[0] + 1) >= config.min_touch_pixels


def select_calibration_component(
    mask: np.ndarray,
    coarse_surface_y: int,
    config: AnalysisConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    labels, count = ndi.label(mask)
    best_label = 0
    best_score = -1

    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < config.min_area_px:
            continue

        bbox = component_bbox(component)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        vertical_span = y_max - y_min + 1
        below_surface = int(component[coarse_surface_y:].sum())
        above_surface = int(component[:coarse_surface_y].sum())
        if below_surface < config.min_area_px or above_surface < config.min_area_px:
            continue
        if y_max < coarse_surface_y + 50:
            continue

        score = area + 20 * below_surface + 100 * vertical_span + (x_max - x_min + 1)
        if score > best_score:
            best_label = label
            best_score = score

    if best_label == 0:
        return None

    component = labels == best_label
    bbox = component_bbox(component)
    if bbox is None:
        return None
    return component, bbox


def row_run_width(component: np.ndarray, row_y: int) -> int:
    bounds = longest_true_run(component[row_y])
    if bounds is None:
        return 0
    x_min, x_max = bounds
    return int(x_max - x_min + 1)


def estimate_surface_y_from_waist(
    component: np.ndarray,
    bbox: tuple[int, int, int, int],
    coarse_surface_y: int,
    config: AnalysisConfig,
) -> int:
    _x_min, y_min, _x_max, y_max = bbox
    vertical_span = y_max - y_min + 1
    lower_margin = max(30, vertical_span // 10)
    search_top = max(coarse_surface_y + config.touch_above_surface_px, y_min + vertical_span // 3)
    search_bottom = min(y_max - lower_margin, coarse_surface_y + config.measure_below_surface_px)
    if search_bottom <= search_top:
        raise ValueError("Invalid waist search window for surface calibration frame")

    rows = np.arange(search_top, search_bottom + 1)
    widths = np.array([row_run_width(component, int(row)) for row in rows], dtype=np.float32)
    valid = widths > 0
    if not valid.any():
        raise ValueError("No foreground rows found in surface calibration waist search window")

    max_width = float(widths[valid].max())
    min_width = max(10.0, max_width * 0.08)
    candidate = valid & (widths >= min_width)
    if not candidate.any():
        candidate = valid

    smoothed = ndi.gaussian_filter1d(widths, sigma=3.0)
    smoothed[~candidate] = np.inf
    return int(rows[int(np.argmin(smoothed))])


def estimate_surface_y_from_symmetry_frame(
    files: list[Path],
    background: np.ndarray,
    frame_number: int,
    threshold: float,
    coarse_surface_y: int,
    config: AnalysisConfig,
    structure: np.ndarray,
) -> int:
    if frame_number < 1 or frame_number > len(files):
        raise ValueError(f"--surface-frame must be between 1 and {len(files)}, got {frame_number}")

    image = read_image(files[frame_number - 1])
    mask = full_foreground_mask(image, background, threshold, structure)
    selected = select_calibration_component(mask, coarse_surface_y, config)
    if selected is None:
        raise ValueError(
            f"Could not find a droplet/reflection component in surface calibration frame {frame_number}"
        )

    component, bbox = selected
    return estimate_surface_y_from_waist(component, bbox, coarse_surface_y, config)


def write_debug_overlay(
    path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    surface_line: SurfaceLine,
    bbox: tuple[int, int, int, int] | None,
    frame_number: int,
    diameter_mm: float,
) -> None:
    lo, hi = np.percentile(image, [0.5, 99.5])
    gray = np.clip((image - lo) / max(1.0, hi - lo) * 255.0, 0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    liquid_pixels = mask > 0
    overlay[liquid_pixels] = (
        overlay[liquid_pixels].astype(np.float32) * (1.0 - DEBUG_MASK_ALPHA)
        + np.array([0, 0, 255], dtype=np.float32) * DEBUG_MASK_ALPHA
    ).astype(np.uint8)
    width = overlay.shape[1]
    surface_start = (0, int(round(surface_line.y_at(0, width))))
    surface_end = (width - 1, int(round(surface_line.y_at(width - 1, width))))
    cv2.line(overlay, surface_start, surface_end, (0, 255, 255), 1)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.line(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
    text = f"frame={frame_number} diameter_mm={diameter_mm:.4f}"
    cv2.putText(
        overlay,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def analyze_sequence(config: AnalysisConfig) -> list[FrameMeasurement]:
    files = find_tiff_files(config.input_dir)
    if config.max_frame is not None:
        if config.max_frame < 1:
            raise ValueError("max_frame must be at least 1")
        files = files[: config.max_frame]
    background = build_background(files, config.background_frames)
    coarse_surface_y = estimate_surface_y(background, config.surface_search_start_px, config.surface_drop_delta)
    structure = make_structure(config.morphology_radius_px)
    preliminary_threshold = (
        float(config.threshold)
        if config.threshold is not None
        else estimate_threshold(files, background, coarse_surface_y, config.background_frames, config.min_foreground_delta)
    )
    if config.surface_y is not None:
        surface_y = int(config.surface_y)
    elif config.surface_frame is not None:
        surface_y = estimate_surface_y_from_symmetry_frame(
            files,
            background,
            config.surface_frame,
            preliminary_threshold,
            coarse_surface_y,
            config,
            structure,
        )
    else:
        surface_y = coarse_surface_y
    surface_line = SurfaceLine(float(surface_y), config.surface_angle_deg)

    threshold = (
        float(config.threshold)
        if config.threshold is not None
        else estimate_threshold(files, background, surface_y, config.background_frames, config.min_foreground_delta)
    )

    pending: list[tuple[int, Path, float, int, tuple[int, int, int, int] | None, bool]] = []
    impact_frame: int | None = None

    for frame_number, file_path in enumerate(files, start=1):
        image = read_image(file_path)
        mask = foreground_mask(image, background, surface_line, threshold, config, structure)
        measurement = component_measurement(mask, surface_line, config)
        is_touching = touches_surface(measurement.mask, surface_line, config)
        if impact_frame is None and is_touching and not math.isnan(measurement.diameter_px):
            impact_frame = frame_number
        pending.append(
            (
                frame_number,
                file_path,
                measurement.diameter_px,
                measurement.area_px,
                measurement.bbox,
                is_touching,
            )
        )

        if config.debug_dir is not None and (
            frame_number == 1
            or frame_number % config.debug_every == 0
            or (is_touching and abs(frame_number - (impact_frame or frame_number)) <= 2)
        ):
            diameter_mm = (
                measurement.diameter_px * config.pixel_size_mm
                if not math.isnan(measurement.diameter_px)
                else math.nan
            )
            debug_path = config.debug_dir / f"{frame_number:06d}.png"
            write_debug_overlay(
                debug_path,
                image,
                measurement.mask,
                surface_line,
                measurement.bbox,
                frame_number,
                diameter_mm,
            )

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
                surface_y=surface_line.center_y_int(),
                impact_frame=impact_frame,
                fps=config.fps,
                pixel_size_mm=config.pixel_size_mm,
                surface_frame=config.surface_frame,
            )
        )

    return measurements


def write_csv(path: Path, measurements: Iterable[FrameMeasurement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "filename",
                "frame",
                "time_ms",
                "diameter_px",
                "diameter_mm",
                "impact_frame",
                "surface_y_px",
                "component_area_px",
                "fps",
                "pixel-size-mm",
                "surface-frame",
            ]
        )
        for row in measurements:
            writer.writerow(
                [
                    row.filename,
                    row.frame_number,
                    f"{row.time_ms:.6f}",
                    "" if math.isnan(row.diameter_px) else f"{row.diameter_px:.3f}",
                    "" if math.isnan(row.diameter_mm) else f"{row.diameter_mm:.9f}",
                    "" if row.impact_frame is None else row.impact_frame,
                    row.surface_y,
                    row.component_area_px,
                    row.fps,
                    row.pixel_size_mm,
                    "" if row.surface_frame is None else row.surface_frame,
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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def default_output_dir(input_dir: Path) -> Path:
    return Path("outputs") / input_dir.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure droplet spreading diameter from high-speed TIFF sequences.")
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
    parser.add_argument("--fps", type=positive_float, default=DEFAULT_FPS, help="Camera frame rate in fps. Default: 8000")
    parser.add_argument(
        "--pixel-size-mm",
        type=positive_float,
        default=DEFAULT_PIXEL_SIZE_MM,
        help="Physical length per pixel in mm. Default: 0.00711883341",
    )
    parser.add_argument("--background-frames", type=int, default=8, help="Number of first frames used for median background.")
    parser.add_argument("--surface-y", type=nonnegative_int, default=None, help="Override detected surface y coordinate in pixels.")
    parser.add_argument("--surface-frame", type=nonnegative_int, default=None, help="Frame number used to calibrate surface y from the droplet/reflection waist.")
    parser.add_argument("--threshold", type=positive_float, default=None, help="Override dark foreground threshold in gray levels.")
    parser.add_argument(
        "--min-foreground-delta",
        type=positive_float,
        default=DEFAULT_MIN_FOREGROUND_DELTA,
        help="Lower bound for automatic threshold. Default: 700",
    )
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
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory for overlay PNG diagnostics. Default: outputs/<input-folder>/debug_overlays",
    )
    parser.add_argument("--debug-every", type=int, default=25, help="Write one debug overlay every N frames when --debug-dir is used.")
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
    if valid:
        max_row = max(valid, key=lambda row: row.diameter_mm)
        print(f"Wrote {len(measurements)} rows to {config.output_csv}")
        print(f"Surface y: {valid[0].surface_y}px")
        print(f"Impact frame: {impact_frame if impact_frame is not None else 'not detected'}")
        print(f"Max diameter: {max_row.diameter_mm:.6f} mm at {max_row.time_ms:.6f} ms")
    else:
        print(f"Wrote {len(measurements)} rows to {config.output_csv}")
        print("No valid droplet diameter measurements were found.")
