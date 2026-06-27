from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile

from droplet_impact_cv.analysis import analyze_sequence
from droplet_impact_cv.imaging import component_measurement
from droplet_impact_cv.models import AnalysisConfig, SurfaceLine


class SequenceAnalysisTests(unittest.TestCase):
    def test_max_frame_limits_sequence_inclusively(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            image = np.full((100, 100), 4000, dtype=np.uint16)
            image[50:] = 2000
            for frame_number in range(1, 4):
                tifffile.imwrite(input_dir / f"frame{frame_number:06d}.tif", image)

            config = AnalysisConfig(
                input_dir=input_dir,
                output_csv=input_dir / "output.csv",
                surface_y=50,
                max_frame=2,
                debug_dir=None,
            )
            measurements = analyze_sequence(config)

        self.assertEqual([row.frame_number for row in measurements], [1, 2])

    def test_frame_numbers_come_from_six_digit_filename_suffixes(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            image = np.full((100, 100), 4000, dtype=np.uint16)
            image[50:] = 2000
            for frame_number in (5, 6, 8):
                tifffile.imwrite(input_dir / f"capture_{frame_number:06d}.tif", image)

            config = AnalysisConfig(
                input_dir=input_dir,
                output_csv=input_dir / "output.csv",
                surface_y=50,
                include_pre_impact=True,
                time_zero="first-frame",
                debug_dir=None,
            )
            measurements = analyze_sequence(config)

        self.assertEqual([row.frame_number for row in measurements], [5, 6, 8])
        self.assertEqual([row.time_ms for row in measurements], [0.0, 0.125, 0.375])

    def test_max_frame_uses_filename_frame_number(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            image = np.full((100, 100), 4000, dtype=np.uint16)
            image[50:] = 2000
            for frame_number in (5, 6, 7):
                tifffile.imwrite(input_dir / f"capture_{frame_number:06d}.tif", image)

            config = AnalysisConfig(
                input_dir=input_dir,
                output_csv=input_dir / "output.csv",
                surface_y=50,
                max_frame=6,
                debug_dir=None,
            )
            measurements = analyze_sequence(config)

        self.assertEqual([row.frame_number for row in measurements], [5, 6])


class ContourMeasurementTests(unittest.TestCase):
    def test_measurement_never_adds_pixels_outside_detected_contour(self) -> None:
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[30:55, 20:80] = 1
        mask[45:55, 20:30] = 0
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            min_area_px=1,
            min_touch_pixels=1,
            debug_dir=None,
        )

        measurement = component_measurement(mask, SurfaceLine(50.0, angle_deg=0.0), config)

        self.assertFalse(np.any(measurement.mask & ~mask.astype(bool)))
        self.assertEqual(measurement.diameter_px, 50.0)


if __name__ == "__main__":
    unittest.main()
