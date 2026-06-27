from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from droplet_impact_cv.imaging import foreground_mask, make_structure
from droplet_impact_cv.models import (
    DEFAULT_MIN_FOREGROUND_DELTA,
    AnalysisConfig,
    SurfaceLine,
)


class ForegroundMaskTests(unittest.TestCase):
    def test_default_threshold_excludes_weak_shadow_attached_to_droplet(self) -> None:
        background = np.full((60, 60), 4000, dtype=np.float32)
        image = background.copy()
        image[15:40, 20:40] = 2800  # Droplet: delta 1200.
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


if __name__ == "__main__":
    unittest.main()
