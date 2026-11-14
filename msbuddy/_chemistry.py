"""Small, continuous chemical-evidence features from accurate fragment masses."""
import numpy as np


_STATISTICS = ('count', 'weighted_mz', 'mean_defect', 'std_defect', 'absolute_defect',
               'positive_defect_fraction', 'maximum_mz', 'minimum_mz')
_DIFFERENCES = (('CF2', 49.996806), ('C2F4', 99.993612), ('CF2O', 65.991721),
                ('HF', 20.006228), ('H2O', 18.010565), ('CO', 27.994915),
                ('CO2', 43.989830), ('NH3', 17.026549), ('HCl', 35.976678),
                ('HBr', 79.926161), ('SO2', 63.961901), ('SO3', 79.956816))
_DIAGNOSTICS = (('F', 18.998403), ('I', 126.904473), ('Cl', 34.968853),
                ('Br', 78.918336), ('SO3', 79.956816), ('SO2', 63.961901),
                ('PO3', 78.958507), ('CO2', 43.989830))
CHEMISTRY_NAMES = (
    tuple(prefix + '_' + name for prefix in ('fragment', 'loss') for name in _STATISTICS)
    + tuple('difference_' + name + '_' + kind for name, _ in _DIFFERENCES for kind in ('count', 'weight'))
    + ('fluoro_fragment_fraction', 'fluoro_fragment_weight', 'fluoro_maximum_C',
       'fluoro_maximum_F', 'fluoro_longest_series')
    + tuple('diagnostic_' + name for name, _ in _DIAGNOSTICS)
    + ('chlorine_pair_count', 'chlorine_pair_ratio_support', 'bromine_pair_count', 'bromine_pair_ratio_support')
    + ('precursor_mz', 'neutral_mass', 'precursor_multiplier', 'precursor_charge')
    + ('intensity_entropy', 'top_peak_fraction', 'fragment_tolerance')
)
assert len(CHEMISTRY_NAMES) == 64


def _statistics(masses, weights):
    if not len(masses):
        return [0.] * 8
    defect = masses - np.rint(masses)
    average = float(np.sum(defect * weights))
    return [len(masses) / 512., float(np.sum(masses * weights)) / 1500., average,
            float(np.sqrt(np.sum((defect - average) ** 2 * weights))),
            float(np.sum(np.abs(defect) * weights)), float(np.sum(weights[defect > 0])),
            float(masses.max()) / 1500., float(masses.min()) / 1500.]


def chemistry_features(mzs, intensities, precursor_mz, adduct, ms2_tolerance):
    """Return 64 soft features; no class or elemental decision removes a formula."""
    masses = np.asarray(mzs, dtype=np.float32).astype(np.float64)
    intensity = np.asarray(intensities, dtype=np.float32).astype(np.float64)
    if (masses.ndim != 1 or masses.shape != intensity.shape
            or not np.isfinite(precursor_mz) or precursor_mz <= 0
            or not np.isfinite(ms2_tolerance) or ms2_tolerance <= 0):
        raise ValueError('Invalid chemical evidence spectrum or tolerance.')
    keep = (np.isfinite(masses) & np.isfinite(intensity) & (masses > 0) & (intensity > 0)
            & (masses < precursor_mz - 1.5) & (masses < 1500))
    masses, intensity = masses[keep], intensity[keep]
    if len(masses):
        keep = intensity >= intensity.max() * .001
        masses, intensity = masses[keep], intensity[keep]
        if len(masses) > 512:
            indices = np.argsort(-intensity, kind='stable')[:512]
            masses, intensity = masses[indices], intensity[indices]
        intensity = intensity / intensity.max()
        weights = intensity / intensity.sum()
    else:
        weights = np.empty(0)
    result = _statistics(masses, weights) + _statistics(precursor_mz - masses, weights)
    strongest = np.argsort(-intensity, kind='stable')[:64]
    m, v = masses[strongest], intensity[strongest]
    order = np.argsort(m, kind='stable')
    m, v = m[order], v[order]
    pair_i, pair_j = np.triu_indices(len(m), 1)
    differences = m[pair_j] - m[pair_i]
    tolerance = np.maximum(.0003, np.hypot(m[pair_i], m[pair_j]) * ms2_tolerance * 1e-6)
    pair_weight = np.sqrt(v[pair_i] * v[pair_j])
    denominator = float(pair_weight.sum()) or 1.
    count_scale = np.log1p(max(1, len(differences)))
    for _, delta in _DIFFERENCES:
        matched = np.abs(differences - delta) <= tolerance
        result += [float(np.log1p(matched.sum()) / count_scale), float(pair_weight[matched].sum() / denominator)]
    targets = []
    carbon = []
    fluorine = []
    family = []
    charge = 1 if adduct.charge > 0 else -1
    for kind, adjustment in enumerate((1, -1)):
        for oxygen in range(4):
            for n in range(1, 31):
                f = 2 * n + adjustment
                targets.append(12 * n + 18.998403 * f + 15.994915 * oxygen - charge * .0005486)
                carbon.append(n)
                fluorine.append(f)
                family.append(kind * 4 + oxygen)
    if len(m):
        hits = np.abs(m[:, None] - np.asarray(targets)[None, :]) <= np.maximum(.0002, m[:, None] * ms2_tolerance * 1e-6)
        peak_hits = hits.any(axis=1)
        target_hits = hits.any(axis=0)
        carbons, fluorines, families = np.asarray(carbon), np.asarray(fluorine), np.asarray(family)
        longest = 0
        for group in range(8):
            present = sorted(set(carbons[target_hits & (families == group)].tolist()))
            run = 0
            previous = -2
            for n in present:
                run = run + 1 if n == previous + 1 else 1
                longest = max(longest, run)
                previous = n
        result += [float(peak_hits.mean()), float(v[peak_hits].sum() / v.sum()),
                   float(carbons[target_hits].max(initial=0)) / 40.,
                   float(fluorines[target_hits].max(initial=0)) / 80., longest / 40.]
    else:
        result += [0.] * 5
    for _, neutral_mass in _DIAGNOSTICS:
        target = neutral_mass - charge * .0005486
        hits = np.abs(m - target) <= np.maximum(.0002, m * ms2_tolerance * 1e-6)
        result.append(float(v[hits].sum() / v.sum()) if len(v) else 0.)
    for delta, ratio in [(1.997050, .3196), (1.997952, .9728)]:
        hits = np.abs(differences - delta) <= tolerance
        weight = pair_weight[hits]
        support = np.exp(-.5 * ((np.log(np.maximum(v[pair_j[hits]], 1e-12) / np.maximum(v[pair_i[hits]], 1e-12)) - np.log(ratio)) / .4) ** 2)
        result += [float(np.log1p(hits.sum()) / count_scale), float(np.sum(weight * support) / denominator)]
    neutral_mass = (precursor_mz * abs(adduct.charge) - adduct.net_formula.mass + adduct.charge * .0005486) / adduct.m
    result += [precursor_mz / 1500., neutral_mass / 1500., adduct.m / 2., adduct.charge]
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-12)))) / np.log(len(weights)) if len(weights) > 1 else 0.
    result += [entropy, float(weights.max()) if len(weights) else 0., ms2_tolerance / 20.]
    values = np.asarray(result, dtype=np.float32)
    if values.shape != (64,) or not np.isfinite(values).all():
        raise ValueError('Invalid chemical evidence features.')
    return values
