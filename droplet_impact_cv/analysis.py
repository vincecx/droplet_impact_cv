from __future__ import annotations

import math
from pathlib import Path

from .imaging import (
    build_background,
    component_measurement,
    estimate_surface_y,
    estimate_surface_y_from_symmetry_frame,
    estimate_threshold,
    find_tiff_files,
    foreground_mask,
    make_structure,
    read_image,
    touches_surface,
)
from .models import AnalysisConfig, FrameMeasurement, SurfaceLine
from .visualization import write_debug_overlay


def analyze_sequence(config: AnalysisConfig) -> list[FrameMeasurement]:
    files = find_tiff_files(config.input_dir)
    if config.max_frame is not None:
        if config.max_frame < 1:
            raise ValueError("max_frame must be at least 1")
        files = files[: config.max_frame]
    background = build_background(files, config.background_frames)
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
            config.background_frames,
            config.min_foreground_delta,
        )
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
        else estimate_threshold(
            files,
            background,
            surface_y,
            config.background_frames,
            config.min_foreground_delta,
        )
    )

    pending: list[tuple[int, Path, float, int]] = []
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
    for frame_number, file_path, diameter_px, area in pending:
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
