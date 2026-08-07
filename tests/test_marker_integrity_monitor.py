import unittest

from marker_integrity_monitor import MarkerIntegrityMonitor


class MarkerIntegrityTopologyTests(unittest.TestCase):

    def test_all_three_fixed_stands_exist(self):
        self.assertEqual(
            set(MarkerIntegrityMonitor.STANDS),
            {
                "OV9281_STAND",
                "ELP1_STAND",
                "ELP2_STAND",
            },
        )

    def test_each_stand_has_two_observers(self):
        for observers in (
            MarkerIntegrityMonitor
            .OBSERVERS
            .values()
        ):
            self.assertEqual(
                len(observers),
                2,
            )

    def test_marker_pairs_are_correct(self):
        self.assertEqual(
            MarkerIntegrityMonitor
            .STANDS["OV9281_STAND"],
            (2, 20),
        )

        self.assertEqual(
            MarkerIntegrityMonitor
            .STANDS["ELP1_STAND"],
            (3, 30),
        )

        self.assertEqual(
            MarkerIntegrityMonitor
            .STANDS["ELP2_STAND"],
            (4, 40),
        )


if __name__ == "__main__":
    unittest.main()
