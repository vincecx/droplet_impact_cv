from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from droplet_impact_cv.cli import build_parser, config_from_args
from droplet_impact_cv.models import DEFAULT_MIN_FOREGROUND_DELTA


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


if __name__ == "__main__":
    unittest.main()
