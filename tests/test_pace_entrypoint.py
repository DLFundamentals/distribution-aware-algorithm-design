from __future__ import annotations

import unittest
from unittest.mock import patch

from benchmarks import pace, pace_competitions


class PaceEntrypointTests(unittest.TestCase):
    def test_every_competition_is_reachable(self) -> None:
        """The unified registry must cover both Dominating Set tracks and every
        entry in the competitions registry, so no experiment loses its command."""
        self.assertIn("pace2025_ds_heuristic", pace.REGISTRY)
        self.assertIn("pace2025_ds_exact", pace.REGISTRY)
        for name in pace_competitions.COMPETITIONS:
            self.assertIn(name, pace.REGISTRY)

    def test_dominating_set_forwards_its_track(self) -> None:
        with patch("benchmarks.pace2025_dominating_set.main", return_value=0) as ds_main:
            self.assertEqual(
                pace.main(["--competition", "pace2025_ds_heuristic", "--test-source", "private"]), 0
            )
        ds_main.assert_called_once_with(["--track", "heuristic", "--test-source", "private"])

        with patch("benchmarks.pace2025_dominating_set.main", return_value=0) as ds_main:
            pace.main(["--competition", "pace2025_ds_exact", "--build-only"])
        ds_main.assert_called_once_with(["--track", "exact", "--build-only"])

    def test_other_competitions_forward_the_competition_flag(self) -> None:
        with patch("benchmarks.pace_competitions.main", return_value=0) as comp_main:
            pace.main(["--competition", "pace2025_hs", "--build-only", "--test-count", "5"])
        comp_main.assert_called_once_with(
            ["--competition", "pace2025_hs", "--build-only", "--test-count", "5"]
        )

    def test_unknown_flags_pass_through_untouched(self) -> None:
        """Pass-through keeps the wrapper from having to track every competition flag."""
        with patch("benchmarks.pace_competitions.main", return_value=0) as comp_main:
            pace.main(["--competition", "pace2024_ocm_exact", "--solvers", "a,b", "--force"])
        comp_main.assert_called_once_with(
            ["--competition", "pace2024_ocm_exact", "--solvers", "a,b", "--force"]
        )

    def test_track_flag_conflicts_with_the_dominating_set_competition(self) -> None:
        with patch("benchmarks.pace2025_dominating_set.main", return_value=0) as ds_main:
            self.assertEqual(
                pace.main(["--competition", "pace2025_ds_heuristic", "--track", "exact"]), 2
            )
        ds_main.assert_not_called()

    def test_missing_competition_is_an_error_not_a_run(self) -> None:
        with patch("benchmarks.pace2025_dominating_set.main") as ds_main, patch(
            "benchmarks.pace_competitions.main"
        ) as comp_main:
            self.assertEqual(pace.main([]), 2)
        ds_main.assert_not_called()
        comp_main.assert_not_called()

    def test_listing_runs_nothing(self) -> None:
        with patch("benchmarks.pace2025_dominating_set.main") as ds_main, patch(
            "benchmarks.pace_competitions.main"
        ) as comp_main:
            self.assertEqual(pace.main(["--list"]), 0)
        ds_main.assert_not_called()
        comp_main.assert_not_called()

    def test_exit_code_propagates_from_the_delegate(self) -> None:
        with patch("benchmarks.pace_competitions.main", return_value=3):
            self.assertEqual(pace.main(["--competition", "pace2025_hs"]), 3)


if __name__ == "__main__":
    unittest.main()
