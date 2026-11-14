"""Compact spectral ranking with lazy, per-engine CPU inference."""
import hashlib
from functools import lru_cache

import numpy as np

from msbuddy._acquisition import collision_energy, instrument_family


ADDUCTS = ('[M+H]+', '[M-H]-', '[M+Na]+', '[M+K]+', '[M+NH4]+', '[M+Cl]-',
           '[M+CH2O2-H]-', '[M+C2H4O2-H]-', '[2M+H]+', '[2M-H]-', '[2M+Na]+')
INSTRUMENTS = ('orbitrap', 'qtof', 'timstof')


def validate(specification):
    from msbuddy._calibration import calibrated_probabilities, CONDITIONAL_CONTEXT_NAMES
    if specification.get('schema_version') != 1:
        raise ValueError('Unsupported compact spectral model schema.')
    encoded = specification['encoder_onnx']
    if (not isinstance(encoded, bytes)
            or hashlib.sha256(encoded).hexdigest() != specification['encoder_sha256']):
        raise ValueError('Compact spectral encoder checksum does not match.')
    for name in ('target_center', 'target_scale'):
        values = np.asarray(specification[name], dtype=float)
        if values.shape != (12,) or not np.all(np.isfinite(values)):
            raise ValueError('Invalid compact count normalization.')
        if name == 'target_scale' and np.any(values <= 0):
            raise ValueError('Invalid compact count normalization scale.')
    for group, dimension in ((0, 288), (2, 286)):
        mode = specification['modes'][group]
        if mode['model'].num_feature() != dimension:
            raise ValueError('Compact ranker feature count does not match its schema.')
        blend = mode['blend']
        values = [blend[k] for k in ('weight', 'baseline_scale', 'new_scale')]
        if not np.all(np.isfinite(values)) or not 0 <= values[0] <= 1 or min(values[1:]) <= 0:
            raise ValueError('Invalid compact ranker blend.')
        calibration = mode['calibration']
        if tuple(calibration.get('context_names', ())) != CONDITIONAL_CONTEXT_NAMES:
            raise ValueError('Invalid compact confidence context.')
        calibrated_probabilities(np.zeros(1), np.zeros(len(CONDITIONAL_CONTEXT_NAMES)), calibration)


@lru_cache(maxsize=1)
def _adduct_codes():
    from msbuddy.base import Adduct
    from msbuddy._candidate_selection import adduct_signature
    result = {}
    for index, text in enumerate(ADDUCTS):
        ion = Adduct(text, text.endswith('+'), report_invalid=True)
        result[adduct_signature(ion)] = index
    return result


def _adduct_index(adduct):
    from msbuddy._candidate_selection import adduct_signature
    return _adduct_codes().get(adduct_signature(adduct))


def in_scope(feature, config):
    """Use the established ranker outside the compact model's validated inputs."""
    if (not getattr(config, 'use_spectral', True) or not config.ppm
            or abs(feature.adduct.charge) != 1 or _adduct_index(feature.adduct) is None):
        return False
    instrument = instrument_family(getattr(feature, 'instrument', None))
    instrument = instrument or instrument_family(getattr(config, 'ms_instr', None))
    if instrument == 'fticr' or not 0 < feature.mz < 1500:
        return False
    if (config.ms1_tol, config.ms2_tol) not in ((5, 10), (10, 20)):
        return False
    expected = (5, 10) if instrument == 'orbitrap' else (10, 20) if instrument in ('qtof', 'timstof') else None
    if expected is not None and (config.ms1_tol, config.ms2_tol) != expected:
        return False
    has_isotope = bool(feature.ms1_processed) and len(feature.ms1_processed) > 1
    budget = 50 if has_isotope else 25
    if getattr(config, 'max_candidates', None) not in (None, budget):
        return False
    if (config.max_isotope_cnt != 4 or config.isotope_bin_mztol != .02
            or config.rel_int_denoise_cutoff != .01 or config.top_n_per_50_da != 6):
        return False
    return True


def encode(feature, default_instrument=None):
    ion = feature.adduct
    mzs = np.asarray(feature.ms2_raw.mz_array, dtype=np.float32)
    intensities = np.asarray(feature.ms2_raw.int_array, dtype=np.float32)
    # Match the sparse float16 representation used to fit the model.
    mzs = mzs.astype(np.float64)
    intensities = intensities.astype(np.float64)
    keep = (np.isfinite(mzs) & np.isfinite(intensities) & (mzs > 0)
            & (intensities > 0) & (mzs < feature.mz - 1.5) & (mzs < 1500))
    mzs, intensities = mzs[keep], intensities[keep]
    if len(mzs):
        keep = intensities >= intensities.max() * .001
        mzs, intensities = mzs[keep], intensities[keep]
        if len(mzs) > 512:
            keep = np.argsort(-intensities, kind='stable')[:512]
            mzs, intensities = mzs[keep], intensities[keep]
        intensities = np.sqrt(intensities / intensities.max())
    spectrum = np.zeros((2, 7520), dtype=np.float32)
    if len(mzs):
        np.maximum.at(spectrum[0], np.floor(mzs / .2).astype(int), intensities)
        losses = feature.mz - mzs
        keep = (losses >= 0) & (losses < 1500)
        np.maximum.at(spectrum[1], np.floor(losses[keep] / .2).astype(int), intensities[keep])
    metadata = np.zeros(24, dtype=np.float32)
    instrument = instrument_family(getattr(feature, 'instrument', None))
    instrument = instrument or instrument_family(default_instrument)
    if instrument in INSTRUMENTS:
        metadata[INSTRUMENTS.index(instrument)] = 1
    index = _adduct_index(ion)
    if index is None:
        raise ValueError('Adduct outside compact spectral model scope.')
    metadata[3 + index] = 1
    energy, unit = collision_energy(getattr(feature, 'collision_energy', None),
                                    getattr(feature, 'collision_energy_unit', None))
    unit_index = {'ev': 1, 'nce': 2}.get(unit, 0)
    metadata[14 + unit_index] = 1
    neutral = (feature.mz * abs(ion.charge) - ion.net_formula.mass + ion.charge * .0005486) / ion.m
    metadata[17:] = [feature.mz / 1500, neutral / 1000, ion.charge, ion.m / 2,
                     ion.net_formula.mass / 50, (energy or 0.) / 100, len(mzs) / 512]
    return spectrum.astype(np.float16), metadata


def _session(dependencies):
    cache = dependencies['model_spectral_cache']
    if 'session' not in cache:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError('Compact ranking requires onnxruntime. Install the package dependencies, '
                              'or set MsbuddyConfig(use_spectral=False).') from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        cache['session'] = ort.InferenceSession(dependencies['model_spectral']['encoder_onnx'],
                                               sess_options=options, providers=['CPUExecutionProvider'])
    return cache['session']


def rank(features, evidence, baseline, group, dependencies):
    spec = dependencies['model_spectral']
    mode = spec['modes'][group]
    center = np.asarray(spec['target_center'])
    scale = np.asarray(spec['target_scale'])
    session = _session(dependencies)
    matrices = []
    offset = 0
    for start in range(0, len(features), 32):
        batch = features[start:start + 32]
        encoded = [encode(f, dependencies.get('model_spectral_instrument')) for f in batch]
        metadata = np.asarray([x[1] for x in encoded], dtype=np.float32)
        masked = metadata.copy()
        masked[:, 17:19] = 0
        z, auxiliary = session.run(None, {'spectra': np.asarray([x[0] for x in encoded], dtype=np.float32),
                                        'metadata': masked})
        if z.shape != (len(batch), 128) or auxiliary.shape != (len(batch), 15):
            raise ValueError('Unexpected compact encoder output shape.')
        for i, feature in enumerate(batch):
            count = len(feature.candidate_formula_list)
            a, b = offset, offset + count
            neutral = np.asarray([c.formula.array for c in feature.candidate_formula_list], dtype=np.float32)
            difference = (auxiliary[i, :12] * scale + center - np.log1p(neutral)) / scale
            values = [evidence[a:b], baseline[a:b], np.repeat(metadata[i:i + 1], count, axis=0),
                      np.repeat(z[i:i + 1], count, axis=0), np.repeat(auxiliary[i:i + 1], count, axis=0),
                      difference, np.abs(difference)]
            if group == 0:
                values.append(np.full((count, 1), len(feature.ms1_processed) / 4.))
            matrices.append(np.column_stack(values))
            offset = b
    if offset != len(baseline):
        raise ValueError('Compact ranking candidate offsets do not match.')
    matrix = np.concatenate(matrices).astype(np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError('Non-finite compact model inputs.')
    predicted = mode['model'].predict(matrix, num_threads=1)
    blend = mode['blend']
    return ((1 - blend['weight']) * baseline / blend['baseline_scale']
            + blend['weight'] * predicted / blend['new_scale'])
