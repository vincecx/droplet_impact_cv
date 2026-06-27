from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from droplet_impact_cv.cli import build_parser, config_from_args, parse_args
from droplet_impact_cv.models import (
    DEFAULT_MIN_FOREGROUND_DELTA,
    DEFAULT_PIXEL_SIZE_MM,
)


class CliConfigTests(unittest.TestCase):
    def test_input_folder_config_overrides_code_defaults(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            (input_dir / "cv_config.txt").write_text(
                "# Experiment settings\n"
                "--fps 4000\n"
                "--pixel-size-mm 0.01682736321\n"
                "--surface-frame 9105\n"
                "--reflection-mode none\n",
                encoding="utf-8",
            )

            config = config_from_args(parse_args([str(input_dir)]))

        self.assertEqual(config.fps, 4000.0)
        self.assertEqual(config.pixel_size_mm, 0.01682736321)
        self.assertEqual(config.surface_frame, 9105)
        self.assertEqual(config.reflection_mode, "none")
        self.assertEqual(config.min_foreground_delta, DEFAULT_MIN_FOREGROUND_DELTA)

    def test_command_line_overrides_input_folder_config(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            (input_dir / "cv_config.txt").write_text(
                "--fps 4000\n"
                "--pixel-size-mm 0.01682736321\n"
                "--include-pre-impact\n",
                encoding="utf-8",
            )

            config = config_from_args(
                parse_args(
                    [
                        str(input_dir),
                        "--fps",
                        "2000",
                        "--no-include-pre-impact",
                    ]
                )
            )

        self.assertEqual(config.fps, 2000.0)
        self.assertEqual(config.pixel_size_mm, 0.01682736321)
        self.assertFalse(config.include_pre_impact)

    def test_missing_config_file_uses_code_defaults(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            config = config_from_args(parse_args([temporary_dir]))

        self.assertEqual(config.fps, 8000.0)
        self.assertEqual(config.pixel_size_mm, DEFAULT_PIXEL_SIZE_MM)
        self.assertFalse(config.include_pre_impact)

    def test_input_specific_default_output_paths(self) -> None:
        args = build_parser().parse_args(["sourcedata/example"])
        config = config_from_args(args)

        self.assertEqual(config.output_csv, Path("outputs/example/spreading_diameter.csv"))
        self.assertEqual(config.debug_dir, Path("outputs/example/debug_overlays"))

    def test_frame_range_is_optional_and_inclusive_limits_are_stored(self) -> None:
        parser = build_parser()
        default_config = config_from_args(parser.parse_args([]))
        bounded_config = config_from_args(
            parser.parse_args(["--start-frame", "12", "--end-frame", "62"])
        )

        self.assertIsNone(default_config.start_frame)
        self.assertIsNone(default_config.end_frame)
        self.assertEqual(bounded_config.start_frame, 12)
        self.assertEqual(bounded_config.end_frame, 62)

    def test_frame_range_limits_must_be_positive(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--start-frame", "0"])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--end-frame", "0"])

    def test_default_foreground_delta_rejects_weak_shadow(self) -> None:
        config = config_from_args(build_parser().parse_args([]))
        self.assertEqual(config.min_foreground_delta, DEFAULT_MIN_FOREGROUND_DELTA)
        self.assertEqual(config.min_foreground_delta, 1500.0)

    def test_surface_angle_is_automatic_unless_explicitly_overridden(self) -> None:
        parser = build_parser()
        automatic = config_from_args(parser.parse_args(["--surface-frame", "189"]))
        overridden = config_from_args(
            parser.parse_args(
                ["--surface-frame", "189", "--surface-angle-deg", "1.25"]
            )
        )

        self.assertIsNone(automatic.surface_angle_deg)
        self.assertEqual(overridden.surface_angle_deg, 1.25)

    def test_command_line_overrides_configured_reflection_mode(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            (input_dir / "cv_config.txt").write_text(
                "--reflection-mode mirror\n",
                encoding="utf-8",
            )

            config = config_from_args(
                parse_args([str(input_dir), "--reflection-mode", "none"])
            )

        self.assertEqual(config.reflection_mode, "none")


if __name__ == "__main__":
    unittest.main()
