import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dmn.activities.image_riff import ImageRiffActivity
from dmn.activities.video_riff import VideoRiffActivity
from dmn.generators import video_runway


class MediaBackendSelectionTests(unittest.TestCase):
    def test_image_riff_prefers_only_nano_banana_by_default(self) -> None:
        activity = ImageRiffActivity()
        ctx = SimpleNamespace(dry_run=False)
        nano = SimpleNamespace(name="nano_banana")
        hf = SimpleNamespace(name="hf_flux")

        with patch("dmn.activities.image_riff.gens.available_for", return_value=[hf, nano]):
            self.assertEqual([g.name for g in activity._backends(ctx)], ["nano_banana"])

    def test_image_riff_can_enable_fallbacks_explicitly(self) -> None:
        activity = ImageRiffActivity()
        ctx = SimpleNamespace(dry_run=False)
        nano = SimpleNamespace(name="nano_banana")
        hf = SimpleNamespace(name="hf_flux")

        with patch.dict(os.environ, {"DMN_IMAGE_ALLOW_FALLBACKS": "1"}):
            with patch("dmn.activities.image_riff.gens.available_for", return_value=[hf, nano]):
                self.assertEqual(
                    [g.name for g in activity._backends(ctx)],
                    ["nano_banana", "hf_flux"],
                )

    def test_video_riff_prefers_runway_before_other_real_backends(self) -> None:
        activity = VideoRiffActivity()
        ctx = SimpleNamespace(dry_run=False)
        replicate = SimpleNamespace(name="replicate_video")
        runway = SimpleNamespace(name="runway_video")

        with patch("dmn.activities.video_riff.gens.available_for", return_value=[replicate, runway]):
            self.assertEqual(
                [g.name for g in activity._backends(ctx)],
                ["runway_video", "replicate_video"],
            )

    def test_video_riff_is_disabled_by_default(self) -> None:
        activity = VideoRiffActivity()
        ctx = SimpleNamespace(dry_run=False)
        runway = SimpleNamespace(name="runway_video")

        with patch.dict(os.environ, {}, clear=True):
            with patch("dmn.activities.video_riff.gens.available_for", return_value=[runway]):
                self.assertFalse(activity.available(ctx))

    def test_video_riff_can_be_enabled_explicitly(self) -> None:
        activity = VideoRiffActivity()
        ctx = SimpleNamespace(dry_run=False)
        runway = SimpleNamespace(name="runway_video")

        with patch.dict(os.environ, {"DMN_ENABLE_VIDEO_RIFFS": "1"}, clear=True):
            with patch("dmn.activities.video_riff.gens.available_for", return_value=[runway]):
                self.assertTrue(activity.available(ctx))

    def test_runway_uses_user_key_alias_and_ten_second_default(self) -> None:
        with patch.dict(os.environ, {"RUN_API_KEY": "run-key"}, clear=True):
            self.assertEqual(video_runway._api_key(), "run-key")
            self.assertEqual(video_runway._duration(None), 10)


if __name__ == "__main__":
    unittest.main()
