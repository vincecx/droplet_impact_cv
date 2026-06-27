from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from droplet_impact_cv.imaging import component_measurement, foreground_mask, make_structure
from droplet_impact_cv.models import (
    DEFAULT_MIN_FOREGROUND_DELTA,
    AnalysisConfig,
    SurfaceLine,
)


class ForegroundMaskTests(unittest.TestCase):
    def test_default_threshold_excludes_weak_shadow_attached_to_droplet(self) -> None:
        background = np.full((60, 60), 4000, dtype=np.float32)
        image = background.copy()
        image[15:40, 20:40] = 2000  # Droplet: delta 2000.
        image[40:43, 20:40] = 3200  # Attached weak shadow: delta 800.
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            debug_dir=None,
        )

        mask = foreground_mask(
            image,
            background,
            SurfaceLine(45.0, angle_deg=0.0),
            DEFAULT_MIN_FOREGROUND_DELTA,
            config,
            make_structure(1),
        )

        self.assertEqual(mask[30, 30], 1)
        self.assertEqual(mask[41, 30], 0)

    def test_connected_thin_contact_boundary_survives_cleanup(self) -> None:
        background = np.full((80, 100), 4000, dtype=np.float32)
        dark_foreground = np.zeros(background.shape, dtype=np.uint8)
        dark_foreground[20:51, 40:81] = 1
        cv2.line(dark_foreground, (40, 20), (20, 50), 1, thickness=2)
        cv2.line(dark_foreground, (5, 10), (5, 40), 1, thickness=2)
        image = background.copy()
        image[dark_foreground > 0] = 2000
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            min_area_px=1,
            min_touch_pixels=1,
            debug_dir=None,
        )
        surface_line = SurfaceLine(50.0, angle_deg=0.0)

        mask = foreground_mask(
            image,
            background,
            surface_line,
            DEFAULT_MIN_FOREGROUND_DELTA,
            config,
            make_structure(3),
        )
        measurement = component_measurement(mask, surface_line, config)

        self.assertEqual(mask[50, 20], 1)
        self.assertEqual(mask[20, 5], 0)
        self.assertEqual(measurement.diameter_px, 62.0)


class ComponentMeasurementTests(unittest.TestCase):
    def test_bright_gap_open_at_surface_is_filled_as_droplet_interior(self) -> None:
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[30:55, 20:80] = 1
        mask[45:55, 45:55] = 0
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            min_area_px=1,
            min_touch_pixels=1,
            debug_dir=None,
        )

        measurement = component_measurement(mask, SurfaceLine(50.0, angle_deg=0.0), config)

        self.assertEqual(measurement.diameter_px, 60.0)
        self.assertTrue(measurement.mask[50, 50])


if __name__ == "__main__":
    unittest.main()
