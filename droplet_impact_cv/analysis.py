from __future__ import annotations

import math
from pathlib import Path

from .imaging import (
    build_background,
    component_measurement,
    estimate_surface_y,
    estimate_surface_line_from_symmetry_frame,
    estimate_threshold,
    find_tiff_files,
    frame_number_from_filename,
    foreground_mask,
    make_structure,
    read_image,
    touches_surface,
)
from .models import (
    DEFAULT_THRESHOLD_SAMPLE_FRAMES,
    DEFAULT_SURFACE_ANGLE_DEG,
    AnalysisConfig,
    FrameMeasurement,
    SurfaceLine,
)
from .visualization import write_debug_overlay


def analyze_sequence(config: AnalysisConfig) -> list[FrameMeasurement]:
    all_files = find_tiff_files(config.input_dir)
    if config.start_frame is not None and config.start_frame < 1:
        raise ValueError("start_frame must be at least 1")
    if config.end_frame is not None and config.end_frame < 1:
        raise ValueError("end_frame must be at least 1")
    if (
        config.start_frame is not None
        and config.end_frame is not None
        and config.start_frame > config.end_frame
    ):
        raise ValueError("start_frame must be less than or equal to end_frame")

    files = all_files
    if config.start_frame is not None or config.end_frame is not None:
        files_in_range = []
        for path in files:
            frame_number = frame_number_from_filename(path)
            after_start = config.start_frame is None or frame_number >= config.start_frame
            before_end = config.end_frame is None or frame_number <= config.end_frame
            if after_start and before_end:
                files_in_range.append(path)
        files = files_in_range
        if not files:
            bounds = []
            if config.start_frame is not None:
                bounds.append(f"at or after frame {config.start_frame}")
            if config.end_frame is not None:
                bounds.append(f"at or before frame {config.end_frame}")
            raise ValueError(f"No input frames found {' and '.join(bounds)}")
    first_frame_number = frame_number_from_filename(files[0])
    background = build_background(files)
    coarse_surface_y = estimate_surface_y(
        background,
        config.surface_search_start_px,
        config.surface_drop_delta,
    )
    structure = make_structure(config.morphology_radius_px)
    preliminary_threshold = (
        float(config.threshold)
        if config.threshold is not None
        else estimate_threshold(
            files,
            background,
            coarse_surface_y,
            DEFAULT_THRESHOLD_SAMPLE_FRAMES,
            config.min_foreground_delta,
        )
    )
    calibrated_surface_line: SurfaceLine | None = None
    if config.surface_frame is not None:
        calibrated_surface_line = estimate_surface_line_from_symmetry_frame(
            files,
            background,
            config.surface_frame,
            preliminary_threshold,
            coarse_surface_y,
            config,
            structure,
            angle_override_deg=config.surface_angle_deg,
        )

    if config.surface_y is not None:
        surface_y = int(config.surface_y)
    elif calibrated_surface_line is not None:
        surface_y = calibrated_surface_line.center_y_int()
    else:
        surface_y = coarse_surface_y

    if calibrated_surface_line is not None:
        surface_angle_deg = calibrated_surface_line.angle_deg
    elif config.surface_angle_deg is not None:
        surface_angle_deg = config.surface_angle_deg
    else:
        surface_angle_deg = DEFAULT_SURFACE_ANGLE_DEG
    surface_line = SurfaceLine(float(surface_y), surface_angle_deg)

    threshold = (
        float(config.threshold)
        if config.threshold is not None
        else estimate_threshold(
            files,
            background,
            surface_y,
            DEFAULT_THRESHOLD_SAMPLE_FRAMES,
            config.min_foreground_delta,
        )
    )

    pending: list[tuple[int, Path, float, int]] = []
    impact_frame: int | None = None

    for file_path in files:
        frame_number = frame_number_from_filename(file_path)
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
            )
        )

        if config.debug_dir is not None and (
            frame_number == first_frame_number
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
    for frame_number, file_path, diameter_px, area in pending:
        if impact_frame is None:
            frame_offset = frame_number - first_frame_number
        elif config.time_zero == "impact":
            frame_offset = frame_number - impact_frame
        else:
            frame_offset = frame_number - first_frame_number

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
