import unittest

import numpy as np

from msbuddy import Adduct
from msbuddy._chemistry import chemistry_features, CHEMISTRY_NAMES


class ChemicalEvidenceTests(unittest.TestCase):
    def test_empty_and_nonfinite_peaks_remain_finite(self):
        ion = Adduct('[M-H]-', False)
        empty = chemistry_features([], [], 400., ion, 10.)
        invalid = chemistry_features([np.nan, np.inf, -1., 100.], [1., 1., 1., np.nan], 400., ion, 10.)
        self.assertEqual(empty.shape, (64,))
        np.testing.assert_array_equal(empty, invalid)
        self.assertTrue(np.isfinite(empty).all())

    def test_intensity_scale_and_peak_order_invariance(self):
        ion = Adduct('[M-H]-', False)
        mz = np.array([68.995758, 118.992564, 168.989370, 218.986176])
        intensity = np.array([7., 11., 31., 19.])
        a = chemistry_features(mz, intensity, 450., ion, 10.)
        b = chemistry_features(mz[::-1], intensity[::-1] * 100., 450., ion, 10.)
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-6)
        self.assertGreater(a[CHEMISTRY_NAMES.index('fluoro_fragment_weight')], .99)
        self.assertGreater(a[CHEMISTRY_NAMES.index('difference_CF2_count')], 0)

    def test_accurate_mass_separates_nominal_fifty_dalton_losses(self):
        ion = Adduct('[M+H]+', True)
        a = chemistry_features([100., 149.996806], [1., 1.], 400., ion, 5.)
        b = chemistry_features([100., 150.], [1., 1.], 400., ion, 5.)
        index = CHEMISTRY_NAMES.index('difference_CF2_count')
        self.assertGreater(a[index], 0.)
        self.assertEqual(b[index], 0.)

    def test_multimer_neutral_mass_is_not_precursor_mass(self):
        ion = Adduct('[2M+Na]+', True)
        mz = (2 * 300. + ion.net_formula.mass - .0005486)
        value = chemistry_features([], [], mz, ion, 20.)
        self.assertAlmostEqual(value[CHEMISTRY_NAMES.index('neutral_mass')], .2, places=7)

    def test_invalid_shapes_and_tolerances(self):
        ion = Adduct('[M+H]+', True)
        for mz, intensity, parent, tolerance in [([1.], [], 100., 10.), ([], [], np.nan, 10.), ([], [], 100., 0.)]:
            with self.assertRaises(ValueError):chemistry_features(mz, intensity, parent, ion, tolerance)
