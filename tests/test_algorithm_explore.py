import unittest

from dmn.activities.algorithm_explore import _ensure_numpy2_compat


class AlgorithmExploreTests(unittest.TestCase):
    def test_rewrites_removed_ndarray_ptp_method(self) -> None:
        code = "m = (M - M.min()) / (M.ptp() + 1e-9)"
        self.assertEqual(
            _ensure_numpy2_compat(code),
            "m = (M - M.min()) / (np.ptp(M) + 1e-9)",
        )


if __name__ == "__main__":
    unittest.main()
