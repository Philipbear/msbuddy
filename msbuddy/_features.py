"""Explicit elemental and ion metadata shared by candidate selection and ranking."""
import numpy as np

from msbuddy.utils import SUPPORTED_ELEMENTS


EXTRA_FEATURE_NAMES = (
    tuple('neutral_' + symbol for symbol in SUPPORTED_ELEMENTS)
    + tuple('ion_' + symbol for symbol in SUPPORTED_ELEMENTS)
    + ('neutral_mass', 'precursor_mz', 'precursor_multiplier', 'precursor_charge',
       'adduct_mass', 'adduct_H', 'adduct_C', 'adduct_N', 'adduct_O',
       'adduct_Na', 'adduct_K', 'adduct_Cl', 'precursor_database_match',
       'bottom_up_explained_intensity', 'raw_fragment_count',
       'processed_fragment_count', 'precursor_tolerance', 'fragment_tolerance')
)


def metadata_features(feature, candidates, ms1_tol, ms2_tol):
    has_ms2 = bool(feature.ms2_processed)
    ion = feature.adduct
    net = ion.net_formula.array
    neutral = np.asarray([c.formula.array for c in candidates], dtype=float).reshape(-1, 12)
    charged = np.asarray([(c.charged_formula.array if c.charged_formula is not None
                           else c.formula.array.astype(np.int64) * ion.m + net)
                          for c in candidates], dtype=float).reshape(-1, 12)
    return metadata_matrix(neutral, charged, np.asarray([c.formula.mass for c in candidates]),
                           feature.mz, ion, np.asarray([bool(c.db_existed) for c in candidates]),
                           np.asarray([c.exp_ms2_sum_int or 0. for c in candidates]) if has_ms2 else np.zeros(len(candidates)),
                           len(feature.ms2_raw) if has_ms2 and feature.ms2_raw else 0,
                           len(feature.ms2_processed) if has_ms2 else 0, ms1_tol, ms2_tol)


def metadata_matrix(neutral, charged, masses, precursor_mz, adduct, database,
                    explained, raw_count, processed_count, ms1_tol, ms2_tol):
    net = adduct.net_formula.array
    common = [precursor_mz, adduct.m, adduct.charge, adduct.net_formula.mass,
              net[1], net[0], net[7], net[9], net[8], net[6], net[3]]
    result = np.empty((len(neutral), len(EXTRA_FEATURE_NAMES)), dtype=float)
    result[:, :12] = neutral
    result[:, 12:24] = charged
    result[:, 24] = masses
    result[:, 25:36] = common
    result[:, 36] = database
    result[:, 37] = explained
    result[:, 38:] = [raw_count, processed_count, ms1_tol, ms2_tol]
    if not np.all(np.isfinite(result)):
        raise ValueError('Non-finite candidate metadata features.')
    return result


def select_evidence_features(matrix, group, version=1):
    """Select evidence columns while retaining appended versioned metadata."""
    base = matrix[:, :53]
    extra = matrix[:, 53:]
    if group == 1:
        base = base[:, :-24]
    elif group == 2:
        base = base[:, 1:]
    elif group == 3:
        base = base[:, 1:-24]
    if version >= 2:
        return np.concatenate((base, extra), axis=1)
    return base
