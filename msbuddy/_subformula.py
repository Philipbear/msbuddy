"""Bounded, mass-targeted matching with the legacy numerical conventions."""

import numpy as np
from numba import njit


_MASSES = np.array([12.000000, 1.007825, 78.918336, 34.968853,
                    18.998403, 126.904473, 38.963707, 14.003074,
                    22.989769, 15.994915, 30.973762, 31.972071], dtype=np.float32)


@njit(cache=True)
def _valid(fragment, precursor):
    atoms = 0
    loss_atoms = 0
    valences = 0
    for i in range(12):
        atoms += fragment[i]
        loss_atoms += precursor[i] - fragment[i]
    if (atoms == fragment[0] or atoms == fragment[7] or atoms == fragment[10]
            or loss_atoms == precursor[0] - fragment[0]
            or loss_atoms == precursor[7] - fragment[7]
            or loss_atoms == precursor[10] - fragment[10]):
        return False
    if fragment[10] > 0 and fragment[9] < 2 * fragment[10]:
        return False
    if precursor[10] > fragment[10] and precursor[9] - fragment[9] < 2 * (precursor[10] - fragment[10]):
        return False
    dbe = fragment[0] + 1 - (fragment[1] + fragment[4] + fragment[3] + fragment[2] + fragment[5] + fragment[8] + fragment[6]) / 2 + (fragment[7] + fragment[10]) / 2
    if dbe < 0:
        return False
    valences = (6 * fragment[11] + 5 * fragment[10] + 4 * fragment[0]
                + 3 * fragment[7] + 2 * fragment[9] + fragment[1]
                + fragment[4] + fragment[3] + fragment[2] + fragment[5]
                + fragment[8] + fragment[6])
    return valences >= 2 * (atoms - 1)


@njit(cache=True)
def _earlier(fragment, previous):
    # The exhaustive enumerator increments carbon fastest. Preserve its tie
    # ordering without calculating a possibly overflowing mixed-radix index.
    for j in range(11, -1, -1):
        if fragment[j] != previous[j]:
            return fragment[j] < previous[j]
    return False


@njit(cache=True)
def closest_subformulas(precursor, targets, tolerances, raw_targets, charge):
    """Return the closest valid subformula per target, without a Cartesian array.

    Conservative double-precision bounds locate possible compositions. The
    final inclusion, mass, distance and tie checks reproduce the float32
    arithmetic and chemical filters used by the exhaustive implementation.
    """
    count = len(targets)
    found = np.zeros(count, dtype=np.bool_)
    formulas = np.zeros((count, 12), dtype=np.int16)
    masses = np.zeros(count, dtype=np.float32)
    order = np.empty(11, dtype=np.int64)
    depth_count = 0
    for i in np.argsort(-_MASSES):
        if i != 1 and precursor[i] > 0:
            order[depth_count] = i
            depth_count += 1
    remaining = np.zeros(depth_count + 1, dtype=np.float64)
    for d in range(depth_count - 1, -1, -1):
        i = order[d]
        remaining[d] = remaining[d + 1] + float(_MASSES[i]) * int(precursor[i])
    hydrogen_max = int(precursor[1])
    hydrogen_mass = float(_MASSES[1])
    electron = np.float32(charge * 0.0005486)
    # The standard floating-point accumulation bound is gamma_n * sum(abs(x)).
    # A larger bound also covers per-element products and the final subtraction.
    maximum_mass = remaining[0] + hydrogen_max * hydrogen_mass
    rounding_guard = max(0.00001, maximum_mass * 32 * np.finfo(np.float32).eps)
    for peak in range(count):
        target = np.float32(targets[peak])
        tolerance = np.float32(tolerances[peak])
        lower = float(target) + float(electron) - float(tolerance) - rounding_guard
        upper = float(target) + float(electron) + float(tolerance) + rounding_guard
        partial = np.zeros(depth_count + 1, dtype=np.float64)
        values = np.zeros(depth_count, dtype=np.int64)
        limits = np.zeros(depth_count, dtype=np.int64)
        fresh = np.ones(depth_count + 1, dtype=np.bool_)
        fragment = np.zeros(12, dtype=np.int16)
        best_error = np.inf
        depth = 0
        while depth >= 0:
            if depth == depth_count:
                h_min = max(0, int(np.ceil((lower - partial[depth]) / hydrogen_mass)))
                h_max = min(hydrogen_max, int(np.floor((upper - partial[depth]) / hydrogen_mass)))
                for h in range(h_min, h_max + 1):
                    fragment[1] = h
                    mass = np.float32(0)
                    for j in range(12):
                        mass += fragment[j] * _MASSES[j]
                    mass = np.float32(mass - electron)
                    if abs(np.float32(target - mass)) > tolerance or not _valid(fragment, precursor):
                        continue
                    error = abs(np.float32(np.float32(raw_targets[peak]) - mass))
                    if error < best_error or (error == best_error and _earlier(fragment, formulas[peak])):
                        best_error = error
                        found[peak] = True
                        formulas[peak] = fragment
                        masses[peak] = mass
                depth -= 1
                if depth >= 0:
                    values[depth] += 1
                continue
            element = order[depth]
            weight = float(_MASSES[element])
            if fresh[depth]:
                rest = remaining[depth + 1] + hydrogen_max * hydrogen_mass
                values[depth] = max(0, int(np.ceil((lower - partial[depth] - rest) / weight)))
                limits[depth] = min(int(precursor[element]), int(np.floor((upper - partial[depth]) / weight)))
                fresh[depth] = False
            if values[depth] > limits[depth]:
                fragment[element] = 0
                fresh[depth] = True
                depth -= 1
                if depth >= 0:
                    values[depth] += 1
                continue
            fragment[element] = values[depth]
            partial[depth + 1] = partial[depth] + values[depth] * weight
            depth += 1
            fresh[depth] = True
    return found, formulas, masses
