"""Set-valued probability calibration with an explicit absent-answer state."""
import numpy as np
from scipy.special import logsumexp


CONTEXT_NAMES = ('maximum_score', 'score_gap', 'log_candidate_count',
                 'top_isotope_similarity', 'top_explained_intensity',
                 'top_halogen_atoms', 'log_fragment_count', 'precursor_multiplier',
                 'precursor_mz')


def score_context(scores, isotope, explained, halogens, fragment_count, multiplier, precursor_mz):
    values = np.asarray(scores, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError('Calibration requires nonempty finite scores.')
    order = np.argsort(-values, kind='stable')
    top = order[0]
    gap = values[top] - values[order[1]] if len(values) > 1 else 0.
    return np.asarray([values[top], gap, np.log1p(len(values)), isotope[top],
                       explained[top], halogens[top], np.log1p(fragment_count), multiplier,
                       precursor_mz], dtype=float)


def feature_context(feature, scores):
    candidates = feature.candidate_formula_list
    has_iso = bool(feature.ms1_processed) and len(feature.ms1_processed) > 1
    has_ms2 = bool(feature.ms2_processed)
    # Match the float32 isotope input used when fitting the rankers and
    # calibrators, including its rounding before context construction.
    isotope = [float(np.float32(c.ms1_isotope_similarity or 0.)) if has_iso else 0. for c in candidates]
    explained = np.zeros(len(candidates))
    if has_ms2:
        denominator = np.sum(feature.ms2_raw.int_array[feature.ms2_processed.idx_array])
        if denominator > 0:
            for i, candidate in enumerate(candidates):
                if candidate.ms2_raw_explanation:
                    explained[i] = np.sum(feature.ms2_raw.int_array[candidate.ms2_raw_explanation.idx_array]) / denominator
    halogens = [np.sum(c.formula.array[2:6]) for c in candidates]
    return score_context(scores, isotope, explained, halogens,
                         len(feature.ms2_processed) if has_ms2 else 0, feature.adduct.m, feature.mz)


def calibrated_probabilities(scores, context, specification):
    if specification.get('method', 'null_softmax') != 'null_softmax':
        raise ValueError('Unsupported probability calibration method.')
    values = np.asarray(scores, dtype=float)
    if not len(values):
        return values, 1.
    center = np.asarray(specification['context_center'], dtype=float)
    scale = np.asarray(specification['context_scale'], dtype=float)
    coefficients = np.asarray(specification['coefficients'], dtype=float)
    inverse_temperature = float(specification['inverse_temperature'])
    context = np.asarray(context, dtype=float)
    if (context.shape != (len(CONTEXT_NAMES),) or center.shape != context.shape
            or scale.shape != context.shape or coefficients.shape != (len(context) + 1,)
            or np.any(scale <= 0) or not np.isfinite(inverse_temperature)
            or inverse_temperature <= 0 or not all(np.all(np.isfinite(x))
                                                  for x in [values, context, center, scale, coefficients])):
        raise ValueError('Invalid probability calibration specification or inputs.')
    null_logit = coefficients[0] + np.dot(coefficients[1:], (context - center) / scale)
    null_logit = np.clip(null_logit, -25., 25.)
    logits = (values - values.max()) * inverse_temperature
    normalizer = logsumexp(np.append(logits, null_logit))
    probabilities = np.exp(logits - normalizer)
    absent_probability = float(np.exp(null_logit - normalizer))
    return probabilities, absent_probability
