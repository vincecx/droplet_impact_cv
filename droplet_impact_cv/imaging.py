from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from .models import AnalysisConfig, ComponentMeasurement, SurfaceLine


SURFACE_MASK_BELOW_TOLERANCE_PX = 0
FRAME_NUMBER_PATTERN = re.compile(r"(?<!\d)(\d{6})$")


def natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def frame_number_from_filename(path: Path) -> int:
    match = FRAME_NUMBER_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(
            f"TIFF filename must end with a six-digit frame number: {path.name}"
        )
    return int(match.group(1))


def find_tiff_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = [*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")]
    if not files:
        raise FileNotFoundError(f"No .tif or .tiff images found in: {input_dir}")

    numbered_files = sorted(
        ((frame_number_from_filename(path), path) for path in files),
        key=lambda item: (item[0], natural_sort_key(item[1])),
    )
    for (frame_number, first), (next_frame_number, second) in zip(
        numbered_files, numbered_files[1:]
    ):
        if frame_number == next_frame_number:
            raise ValueError(
                f"Duplicate frame number {frame_number:06d}: {first.name}, {second.name}"
            )
    return [path for _, path in numbered_files]


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
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, structure) > 0
    opened = cv2.morphologyEx(closed.astype(np.uint8), cv2.MORPH_OPEN, structure) > 0

    # Use the opened regions as reliable foreground markers, then reconstruct
    # their original closed shapes.  A conventional opening removes narrow
    # but real contour sections, such as a dark contact-line boundary beside
    # a bright reflection.  Reconstruction restores such sections when they
    # remain connected to a robust liquid region while still rejecting small,
    # disconnected foreground artifacts that the opening removed completely.
    reconstructed = ndi.binary_propagation(opened, mask=closed)
    return ndi.binary_fill_holes(reconstructed).astype(np.uint8)


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


def fill_droplet_interior(
    component: np.ndarray,
    surface_line: SurfaceLine,
    seal_half_width_px: int = 1,
) -> np.ndarray:
    """Seal the substrate contact span and fill the droplet interior.

    An internal bright region can leak into the background through the contact
    line and survive a conventional hole fill.  By default, the liquid contact
    span is continuous and the selected droplet contains no internal air gaps.
    Top- and side-open exterior concavities remain unchanged.
    """
    filled = ndi.binary_fill_holes(np.asarray(component, dtype=bool))
    profile = spreading_profile_on_surface(filled, surface_line, seal_half_width_px)
    contact_xs = np.flatnonzero(profile)
    if contact_xs.size == 0:
        return filled

    height, width = filled.shape
    xs = np.arange(int(contact_xs[0]), int(contact_xs[-1]) + 1)
    surface_ys = np.rint(surface_line.y_at(xs, width)).astype(int)
    for offset in range(-seal_half_width_px, seal_half_width_px + 1):
        ys = surface_ys + offset
        valid = (ys >= 0) & (ys < height)
        filled[ys[valid], xs[valid]] = True

    return ndi.binary_fill_holes(filled)


def component_measurement(
    mask: np.ndarray,
    surface_line: SurfaceLine,
    config: AnalysisConfig,
) -> ComponentMeasurement:
    component, area = select_liquid_component(mask, surface_line, config)
    if area == 0:
        return ComponentMeasurement(math.nan, 0, None, component)

    # Seal apparent holes that open through a few pixels near the substrate,
    # then treat the selected droplet's enclosed bright regions as liquid.
    component = fill_droplet_interior(component, surface_line)

    # Keep the outer measured contour tied to foreground pixels. Polygon
    # approximation can cut across concave edges or extend beyond the liquid.
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


def estimate_vertical_symmetry_y(
    component: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[int, float]:
    """Estimate a horizontal symmetry axis from component row widths.

    The surface calibration frame is expected to contain a droplet and its
    reflection.  Comparing the upper and lower silhouette avoids relying on a
    coarse background edge, which can identify a second substrate boundary
    below the actual surface.
    """
    _x_min, y_min, _x_max, y_max = bbox
    vertical_span = y_max - y_min + 1
    if vertical_span < 5:
        raise ValueError(
            "Droplet/reflection component is too short for symmetry calibration"
        )

    widths = np.array(
        [row_run_width(component, row_y) for row_y in range(y_min, y_max + 1)],
        dtype=np.float32,
    )
    max_width = float(widths.max())
    if max_width <= 0:
        raise ValueError("No foreground rows found in symmetry calibration component")

    quarter_span = max(1, vertical_span // 4)
    center_start = y_min + quarter_span
    center_stop = y_max - quarter_span
    best_y = center_start
    best_error = math.inf

    for center_y in range(center_start, center_stop + 1):
        center_index = center_y - y_min
        half_span = min(center_y - y_min, y_max - center_y)
        if half_span < 1:
            continue

        upper = widths[center_index - half_span : center_index]
        lower = widths[center_index + 1 : center_index + half_span + 1][::-1]
        silhouette_error = float(np.mean(np.abs(upper - lower))) / max_width
        extent_error = abs((center_y - y_min) - (y_max - center_y)) / vertical_span
        error = silhouette_error + extent_error
        if error < best_error:
            best_y = center_y
            best_error = error

    return best_y, best_error


def select_symmetric_calibration_component(
    mask: np.ndarray,
    coarse_surface_y: int,
    config: AnalysisConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Find a nearby component without requiring it to cross the coarse edge."""
    labels, count = ndi.label(mask)
    best: (
        tuple[tuple[float, float, int], np.ndarray, tuple[int, int, int, int]]
        | None
    ) = None
    max_edge_gap = max(
        config.touch_above_surface_px,
        4 * max(1, config.morphology_radius_px),
    )
    min_vertical_span = 2 * max(2, config.morphology_radius_px) + 1

    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < config.min_area_px:
            continue

        bbox = component_bbox(component)
        if bbox is None:
            continue
        _x_min, y_min, _x_max, y_max = bbox
        if y_max - y_min + 1 < min_vertical_span:
            continue

        if coarse_surface_y < y_min:
            edge_gap = y_min - coarse_surface_y
        elif coarse_surface_y > y_max:
            edge_gap = coarse_surface_y - y_max
        else:
            edge_gap = 0
        if edge_gap > max_edge_gap:
            continue

        symmetry_y, symmetry_error = estimate_vertical_symmetry_y(component, bbox)
        min_side_area = min(config.min_area_px, max(1, area // 10))
        above_area = int(component[:symmetry_y].sum())
        below_area = int(component[symmetry_y + 1 :].sum())
        if above_area < min_side_area or below_area < min_side_area:
            continue

        score = (float(edge_gap), symmetry_error, -area)
        if best is None or score < best[0]:
            best = (score, component, bbox)

    if best is None:
        return None
    return best[1], best[2]


def row_run_width(component: np.ndarray, row_y: int) -> int:
    bounds = longest_true_run(component[row_y])
    if bounds is None:
        return 0
    x_min, x_max = bounds
    return int(x_max - x_min + 1)


def _contact_vertex_index(profile: np.ndarray, points_outward: bool) -> int:
    """Find a contact tip, or the notch bracketed by two reflection lobes."""
    smoothed = ndi.gaussian_filter1d(profile.astype(np.float32), sigma=2.0)
    outward_signal = -smoothed if points_outward else smoothed
    inward_signal = -outward_signal
    prominence = max(2.0, 0.005 * float(np.ptp(profile)))
    distance = max(3, int(round(0.03 * len(profile))))
    outward, outward_properties = find_peaks(
        outward_signal,
        prominence=prominence,
        distance=distance,
    )
    inward, inward_properties = find_peaks(
        inward_signal,
        prominence=prominence,
        distance=distance,
    )

    bracketed_inward = [
        (index, inward_properties["prominences"][position])
        for position, index in enumerate(inward)
        if np.any(outward < index) and np.any(outward > index)
    ]
    if bracketed_inward:
        return int(max(bracketed_inward, key=lambda item: item[1])[0])
    if outward.size:
        strongest = int(np.argmax(outward_properties["prominences"]))
        return int(outward[strongest])
    return int(np.argmax(outward_signal))


def estimate_contact_line(component: np.ndarray) -> SurfaceLine:
    """Fit the surface through the two contact tips or concave vertices."""
    bbox = component_bbox(component)
    if bbox is None:
        raise ValueError("Cannot estimate a contact line from an empty component")
    x_min, y_min, x_max, y_max = bbox
    row_ys = np.arange(y_min, y_max + 1)
    left_profile = np.empty(len(row_ys), dtype=np.float32)
    right_profile = np.empty(len(row_ys), dtype=np.float32)
    for index, row_y in enumerate(row_ys):
        bounds = longest_true_run(component[row_y], close_size=1)
        if bounds is None:
            raise ValueError("Calibration component contains an empty interior row")
        left_profile[index], right_profile[index] = bounds

    left_index = _contact_vertex_index(left_profile, points_outward=True)
    right_index = _contact_vertex_index(right_profile, points_outward=False)
    left_x = float(left_profile[left_index])
    left_y = float(row_ys[left_index])
    right_x = float(right_profile[right_index])
    right_y = float(row_ys[right_index])
    if right_x <= left_x:
        raise ValueError("Detected contact vertices do not form a valid surface line")

    slope = (right_y - left_y) / (right_x - left_x)
    image_center_x = (component.shape[1] - 1) / 2.0
    center_y = left_y + slope * (image_center_x - left_x)
    return SurfaceLine(center_y, math.degrees(math.atan(slope)))


def estimate_surface_line_from_symmetry_frame(
    files: list[Path],
    background: np.ndarray,
    frame_number: int,
    threshold: float,
    coarse_surface_y: int,
    config: AnalysisConfig,
    structure: np.ndarray,
    angle_override_deg: float | None = None,
) -> SurfaceLine:
    matching_file = next(
        (path for path in files if frame_number_from_filename(path) == frame_number),
        None,
    )
    if matching_file is None:
        raise ValueError(f"--surface-frame {frame_number} does not exist in the input sequence")

    image = read_image(matching_file)
    mask = full_foreground_mask(image, background, threshold, structure)
    selected = select_calibration_component(mask, coarse_surface_y, config)
    if selected is None:
        selected = select_symmetric_calibration_component(mask, coarse_surface_y, config)
        if selected is None:
            raise ValueError(
                "Could not find a droplet/reflection component in surface "
                f"calibration frame {frame_number}"
            )
        component, _bbox = selected
    else:
        component, _bbox = selected

    detected_line = estimate_contact_line(component)
    if angle_override_deg is None:
        return detected_line
    return SurfaceLine(detected_line.center_y, float(angle_override_deg))


def estimate_surface_y_from_symmetry_frame(
    files: list[Path],
    background: np.ndarray,
    frame_number: int,
    threshold: float,
    coarse_surface_y: int,
    config: AnalysisConfig,
    structure: np.ndarray,
) -> int:
    line = estimate_surface_line_from_symmetry_frame(
        files,
        background,
        frame_number,
        threshold,
        coarse_surface_y,
        config,
        structure,
        angle_override_deg=0.0,
    )
    return line.center_y_int()
