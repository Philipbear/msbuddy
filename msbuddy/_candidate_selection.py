"""Cheap, versioned learned selection before fragment assignment."""
import numpy as np

from msbuddy._features import metadata_features


def adduct_signature(adduct):
    """Identify equivalent ion compositions independently of adduct spelling."""
    return (int(adduct.m), int(adduct.charge)) + tuple(map(int, adduct.net_formula.array))


def in_model_scope(feature, ppm, dependencies):
    allowed = dependencies.get('validated_adduct_signatures')
    return (ppm and abs(feature.adduct.charge) == 1
            and (allowed is None or adduct_signature(feature.adduct) in allowed))


def selector_features(feature, candidates, ppm, ms1_tol, ms2_tol):
    from msbuddy.ml import _gen_form_feature, _calc_log_p_norm_many
    if not candidates:
        return np.empty((0, 73), dtype=float)
    counts = np.asarray([c.formula.array for c in candidates], dtype=float)
    dbe = np.asarray([c.formula.dbe for c in candidates])
    masses = np.asarray([c.formula.mass for c in candidates])
    formula = _gen_form_feature(counts, dbe, masses)
    charged = np.asarray([c.charged_formula.mass for c in candidates], dtype=float)
    error = (feature.mz - charged) / charged * 1e6 if ppm else feature.mz - charged
    mass_score = _calc_log_p_norm_many(error, ms1_tol / 3)
    has_iso = bool(feature.ms1_processed) and len(feature.ms1_processed) > 1
    has_ms2 = bool(feature.ms2_processed)
    isotope = [float(c.ms1_isotope_similarity or 0.) if has_iso else 0. for c in candidates]
    extra = metadata_features(feature, candidates, ms1_tol, ms2_tol)
    matrix = np.column_stack((isotope, mass_score, np.full(len(candidates), feature.adduct.pos_mode),
                              formula, extra, np.full(len(candidates), has_iso),
                              np.full(len(candidates), has_ms2)))
    if matrix.shape[1] != 73 or not np.all(np.isfinite(matrix)):
        raise ValueError('Invalid learned candidate-selector features.')
    return matrix.astype(np.float32)


def select_candidates(feature, candidates, ppm, ms1_tol, ms2_tol, dependencies,
                      limit=None):
    from msbuddy.cand import _retain_top_cand_form
    has_iso = bool(feature.ms1_processed) and len(feature.ms1_processed) > 1
    has_ms2 = bool(feature.ms2_processed)
    group = 0 if has_iso and has_ms2 else 1 if has_iso else 2 if has_ms2 else 3
    selector = dependencies.get('candidate_selector')
    applicable = selector is not None and in_model_scope(feature, ppm, dependencies)
    cap = limit if limit is not None else dependencies.get('candidate_limits', (350,) * 4)[group] if applicable else 350
    if isinstance(cap, bool) or not isinstance(cap, (int, np.integer)) or not 1 <= cap <= 10000:
        raise ValueError('Candidate limit must be an integer between 1 and 10000.')
    if not candidates or (not applicable and len(candidates) <= cap):
        return candidates
    if not applicable:
        return _retain_top_cand_form(candidates, ms1_tol, has_iso, has_ms2, cap)
    matrix = selector_features(feature, candidates, ppm, ms1_tol, ms2_tol)
    scores = np.asarray(selector.predict(matrix, num_threads=1), dtype=float)
    if scores.shape != (len(candidates),) or not np.all(np.isfinite(scores)):
        raise ValueError('Candidate selector returned invalid scores.')
    order = np.argsort(-scores, kind='stable')[:cap]
    return [candidates[i] for i in order]
