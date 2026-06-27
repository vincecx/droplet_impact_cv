from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi

from .models import AnalysisConfig, ComponentMeasurement, SurfaceLine


SURFACE_MASK_BELOW_TOLERANCE_PX = 4


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


def surface_band_mask(
    shape: tuple[int, int],
    surface_line: SurfaceLine,
    below_tolerance_px: int,
) -> np.ndarray:
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
    component &= surface_band_mask(
        component.shape,
        surface_line,
        SURFACE_MASK_BELOW_TOLERANCE_PX,
    )
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
    search_top = max(
        coarse_surface_y + config.touch_above_surface_px,
        y_min + vertical_span // 3,
    )
    search_bottom = min(
        y_max - lower_margin,
        coarse_surface_y + config.measure_below_surface_px,
    )
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
