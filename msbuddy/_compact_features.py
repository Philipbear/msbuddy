"""Versioned candidate inputs shared by fitting and compact CPU inference."""
import numpy as np
from scipy.special import gammaln


FEATURE_COUNT = 435


def fluorine_log_probability(raw, counts):
    """Hurdle negative-binomial likelihood, with unbounded positive support."""
    raw = np.asarray(raw, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    mean = np.logaddexp(0., raw[..., 1]) + 1e-4
    shape = np.logaddexp(0., raw[..., 2]) + 1e-4
    log_zero = shape * (np.log(shape) - np.log(shape + mean))
    positive = (gammaln(counts + shape) - gammaln(shape) - gammaln(counts + 1)
                + log_zero + counts * (np.log(mean) - np.log(shape + mean))
                - np.log(np.maximum(-np.expm1(log_zero), 1e-8)))
    return np.where(counts > 0, -np.logaddexp(0., -raw[..., 0]) + positive,
                    -np.logaddexp(0., raw[..., 0]))


def candidate_matrix(evidence, counts, metadata, chemistry, embedding, auxiliary,
                     center, scale, group, isotope_count):
    """Build the same 435 inputs for either joint or MS2-only candidates."""
    physical = np.asarray(evidence, dtype=np.float32)
    n = len(physical)
    if group == 2:
        if physical.shape != (n, 94):
            raise ValueError('Unexpected MS2 evidence shape.')
        physical = np.column_stack([np.zeros(n, dtype=np.float32), physical])
    elif group != 0 or physical.shape != (n, 95):
        raise ValueError('Unexpected joint evidence shape.')
    counts = np.asarray(counts, dtype=np.float64)
    aux = np.asarray(auxiliary, dtype=np.float64)
    center, scale = np.asarray(center), np.asarray(scale)
    if (counts.shape != (n, 12) or aux.shape != (72,) or center.shape != (12,)
            or scale.shape != (12,) or np.any(counts < 0) or np.any(scale <= 0)):
        raise ValueError('Invalid compact count inputs.')
    difference = (aux[:12] * scale + center - np.log1p(counts)) / scale
    standardized = difference / np.exp(aux[15:27])
    log_likelihood = -.5 * standardized ** 2 - aux[15:27] - .5 * np.log(2 * np.pi)
    def repeat(values, dimension):
        values = np.asarray(values)
        if values.shape != (dimension,):
            raise ValueError('Invalid compact query input shape.')
        return np.broadcast_to(values, (n, dimension))
    result = np.column_stack([
        physical, repeat(metadata, 24), repeat(chemistry, 64), repeat(embedding, 128),
        repeat(aux, 72), difference, np.abs(difference), standardized, log_likelihood,
        fluorine_log_probability(aux[27:30], counts[:, 4]),
        np.full(n, float(group == 0)), np.ones(n), np.full(n, isotope_count / 4.)
    ]).astype(np.float32)
    if result.shape != (n, FEATURE_COUNT) or not np.all(np.isfinite(result)):
        raise ValueError('Non-finite or invalid compact candidate inputs.')
    return result
