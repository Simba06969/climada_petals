"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test ImpfHeatWave class.
"""

import unittest
import numpy as np

from climada_petals.entity.impact_funcs.heat_wave import ImpfHeatWave


class TestImpfHeatWave(unittest.TestCase):
    """Impact function test for heat waves"""

    def test_init(self):
        """Test initialization"""
        impf = ImpfHeatWave()
        self.assertEqual(impf.haz_type, 'HW')
        self.assertEqual(impf.intensity_unit, '°C')

    def test_from_step_function_default(self):
        """Test step function with default parameters"""
        impf = ImpfHeatWave.from_step_function()

        self.assertEqual(impf.id, 1)
        self.assertEqual(impf.name, 'Step function heat wave')
        self.assertEqual(impf.haz_type, 'HW')
        self.assertEqual(impf.intensity_unit, '°C')

        # Check that intensity array exists and has correct shape
        self.assertEqual(len(impf.intensity), 4)
        self.assertEqual(len(impf.mdd), 4)
        self.assertEqual(len(impf.paa), 4)

        # Check step function behavior
        self.assertEqual(impf.mdd[0], 0.)  # Below threshold
        self.assertEqual(impf.mdd[1], 0.)  # Just below threshold
        self.assertEqual(impf.mdd[2], 1.)  # At threshold
        self.assertEqual(impf.mdd[3], 1.)  # Above threshold

        # Check paa is all ones
        self.assertTrue(np.all(impf.paa == 1.))

    def test_from_step_function_custom_threshold(self):
        """Test step function with custom threshold"""
        threshold = 28.0
        impf = ImpfHeatWave.from_step_function(
            impf_id=2,
            threshold=threshold,
            name='Custom step function'
        )

        self.assertEqual(impf.id, 2)
        self.assertEqual(impf.name, 'Custom step function')

        # Check threshold is applied correctly
        self.assertAlmostEqual(impf.intensity[2], threshold, places=5)

    def test_from_sigmoid_function_default(self):
        """Test sigmoid function with default parameters"""
        impf = ImpfHeatWave.from_sigmoid_function()

        self.assertEqual(impf.id, 1)
        self.assertEqual(impf.name, 'Sigmoid function heat wave')
        self.assertEqual(impf.haz_type, 'HW')

        # Check that intensity and mdd arrays have same length
        self.assertEqual(len(impf.intensity), len(impf.mdd))
        self.assertEqual(len(impf.intensity), len(impf.paa))

        # Check sigmoid behavior: mdd should increase monotonically
        for i in range(1, len(impf.mdd)):
            self.assertGreaterEqual(impf.mdd[i], impf.mdd[i-1])

        # Check that mdd is bounded between 0 and 1
        self.assertTrue(np.all(impf.mdd >= 0))
        self.assertTrue(np.all(impf.mdd <= 1))

    def test_from_sigmoid_function_custom_steepness(self):
        """Test sigmoid function with custom steepness parameter"""
        impf_steep = ImpfHeatWave.from_sigmoid_function(steepness=1.0)
        impf_gentle = ImpfHeatWave.from_sigmoid_function(steepness=0.2)

        # Steeper function should have larger gradient at half_point
        # Compare mdd values at points away from half_point
        half_point = 33.0
        idx_below = np.argmin(np.abs(impf_steep.intensity - (half_point - 5)))
        idx_above = np.argmin(np.abs(impf_steep.intensity - (half_point + 5)))

        # Steep function should be closer to 0/1 at these points
        self.assertLess(impf_steep.mdd[idx_below], impf_gentle.mdd[idx_below])
        self.assertGreater(impf_steep.mdd[idx_above], impf_gentle.mdd[idx_above])

    def test_from_sigmoid_function_custom_halfpoint(self):
        """Test sigmoid function with custom half_point"""
        half_point = 30.0
        impf = ImpfHeatWave.from_sigmoid_function(half_point=half_point)

        # Find the index closest to the half_point
        idx = np.argmin(np.abs(impf.intensity - half_point))

        # Check that mdd at half_point is approximately 0.5
        self.assertAlmostEqual(impf.mdd[idx], 0.5, places=1)

    def test_from_heat_mortality_default(self):
        """Test heat mortality function with default parameters"""
        impf = ImpfHeatWave.from_heat_mortality()

        self.assertEqual(impf.id, 1)
        self.assertEqual(impf.name, 'Heat mortality function')
        self.assertEqual(impf.haz_type, 'HW')

        # Check that arrays have correct length
        self.assertEqual(len(impf.intensity), 6)
        self.assertEqual(len(impf.mdd), 6)
        self.assertEqual(len(impf.paa), 6)

        # Check that mdd values are monotonically increasing
        for i in range(1, len(impf.mdd)):
            self.assertGreaterEqual(impf.mdd[i], impf.mdd[i-1])

        # Check boundary values
        self.assertEqual(impf.mdd[0], 0.)
        self.assertEqual(impf.mdd[-1], 1.)

    def test_from_heat_mortality_custom_threshold(self):
        """Test heat mortality function with custom threshold"""
        threshold = 25.0
        impf = ImpfHeatWave.from_heat_mortality(
            impf_id=3,
            threshold=threshold
        )

        self.assertEqual(impf.id, 3)
        # Check threshold is in intensity array
        self.assertIn(threshold, impf.intensity)

    def test_impf_check_passes(self):
        """Test that all impact functions pass the check method"""
        impf1 = ImpfHeatWave.from_step_function()
        impf2 = ImpfHeatWave.from_sigmoid_function()
        impf3 = ImpfHeatWave.from_heat_mortality()

        # check() raises an error if something is wrong
        # If no exception is raised, the test passes
        try:
            impf1.check()
            impf2.check()
            impf3.check()
        except Exception as e:
            self.fail(f"Impact function check failed: {e}")


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestImpfHeatWave)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
