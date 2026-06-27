from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

from .models import FrameMeasurement


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
