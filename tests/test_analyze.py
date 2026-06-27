from __future__ import annotations

import csv
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
import tifffile

from droplet_impact_cv.analyze import (
    DEFAULT_MIN_FOREGROUND_DELTA,
    AnalysisConfig,
    FrameMeasurement,
    SurfaceLine,
    analyze_sequence,
    build_parser,
    component_measurement,
    config_from_args,
    write_debug_overlay,
    write_csv,
)


class CliConfigTests(unittest.TestCase):
    def test_input_specific_default_output_paths(self) -> None:
        args = build_parser().parse_args(["sourcedata/example"])
        config = config_from_args(args)

        self.assertEqual(config.output_csv, Path("outputs/example/spreading_diameter.csv"))
        self.assertEqual(config.debug_dir, Path("outputs/example/debug_overlays"))

    def test_max_frame_is_optional_and_inclusive_limit_is_stored(self) -> None:
        parser = build_parser()
        self.assertIsNone(config_from_args(parser.parse_args([])).max_frame)
        self.assertEqual(config_from_args(parser.parse_args(["--max-frame", "62"])).max_frame, 62)

    def test_max_frame_must_be_positive(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--max-frame", "0"])

    def test_default_foreground_delta_rejects_weak_shadow(self) -> None:
        config = config_from_args(build_parser().parse_args([]))
        self.assertEqual(config.min_foreground_delta, DEFAULT_MIN_FOREGROUND_DELTA)
        self.assertEqual(config.min_foreground_delta, 700.0)

    def test_max_frame_limits_sequence_inclusively(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            image = np.full((100, 100), 4000, dtype=np.uint16)
            image[50:] = 2000
            for frame_number in range(1, 4):
                tifffile.imwrite(input_dir / f"frame{frame_number}.tif", image)

            config = AnalysisConfig(
                input_dir=input_dir,
                output_csv=input_dir / "output.csv",
                background_frames=1,
                surface_y=50,
                max_frame=2,
                debug_dir=None,
            )
            measurements = analyze_sequence(config)

        self.assertEqual([row.frame_number for row in measurements], [1, 2])


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


class OutputTests(unittest.TestCase):
    def test_debug_overlay_uses_antialiased_hershey_duplex(self) -> None:
        image = np.zeros((80, 100), dtype=np.float32)
        mask = np.zeros_like(image, dtype=bool)
        with TemporaryDirectory() as temporary_dir, patch(
            "droplet_impact_cv.analyze.cv2.putText", wraps=cv2.putText
        ) as put_text:
            output_path = Path(temporary_dir) / "overlay.png"
            write_debug_overlay(
                output_path,
                image,
                mask,
                SurfaceLine(50.0, angle_deg=0.0),
                None,
                1,
                1.0,
            )

        self.assertEqual(put_text.call_args.args[3], cv2.FONT_HERSHEY_DUPLEX)
        self.assertEqual(put_text.call_args.args[7], cv2.LINE_AA)

    def test_csv_columns_and_analysis_parameters(self) -> None:
        measurement = FrameMeasurement(
            frame_number=61,
            filename="frame0061.tif",
            time_ms=0.0,
            diameter_px=156.004,
            diameter_mm=1.110565076,
            component_area_px=72128,
            surface_y=798,
            impact_frame=61,
            fps=8000.0,
            pixel_size_mm=0.00711883341,
            surface_frame=61,
        )
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "result.csv"
            write_csv(output_path, [measurement])
            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(
            rows[0],
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
            ],
        )
        self.assertEqual(
            rows[1],
            [
                "frame0061.tif",
                "61",
                "0.000000",
                "156.004",
                "1.110565076",
                "61",
                "798",
                "72128",
                "8000.0",
                "0.00711883341",
                "61",
            ],
        )


if __name__ == "__main__":
    unittest.main()
