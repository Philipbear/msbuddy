"""Signed elemental differences for directed molecular-network edges.

This module enumerates a bounded space; it does not infer a reaction mechanism
from a mass difference.  No formula database or learned model is loaded.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations_with_replacement, islice
import math
from numbers import Integral, Real
import re
from types import MappingProxyType
from typing import Optional, Tuple

import numpy as np

from .utils import (SUPPORTED_ELEMENTS, UnsupportedElementError,
                    _validate_composition, formula_to_string, read_formula)


# Atomic (not proton/ion) monoisotopic masses in Da.  Keep double precision:
# cancellation in a signed difference is unusually sensitive to early rounding.
MONOISOTOPIC_MASSES = MappingProxyType(dict(zip(SUPPORTED_ELEMENTS, (
    12.0, 1.00782503223, 78.9183376, 34.968852682, 18.99840316273,
    126.9044719, 38.9637064864, 14.00307400443, 22.9897692820,
    15.99491461957, 30.97376199842, 31.9720711744,
))))
ELECTRON_MASS = 0.000548579909065
CARBON_ISOTOPE_DIFFERENCE = 1.00335483507
_MASSES = np.array([MONOISOTOPIC_MASSES[e] for e in SUPPORTED_ELEMENTS])
_MAX_COUNT = 32767
_DEFAULT_RANGES = {
    'C': (-6, 6), 'H': (-18, 18), 'Br': (-1, 1), 'Cl': (-1, 1),
    'F': (-12, 12), 'I': (-1, 1), 'K': (-1, 1), 'N': (-3, 3),
    'Na': (-1, 1), 'O': (-6, 6), 'P': (-1, 1), 'S': (-2, 2),
}
DEFAULT_DELTA_RANGES = MappingProxyType(_DEFAULT_RANGES)
BROAD_DELTA_RANGES = MappingProxyType({
    'C': (-12, 12), 'H': (-30, 30), 'Br': (-2, 2), 'Cl': (-2, 2),
    'F': (-16, 16), 'I': (-1, 1), 'K': (-1, 1), 'N': (-6, 6),
    'Na': (-1, 1), 'O': (-10, 10), 'P': (-2, 2), 'S': (-3, 3),
})
_PRESETS = {'compact': (DEFAULT_DELTA_RANGES, 24, 4), 'broad': (BROAD_DELTA_RANGES, 40, 6)}


def _integer(value, name, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(name + ' must be an integer.')
    value = int(value)
    if ((minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError('%s must be between %s and %s.' % (name, minimum, maximum))
    return value


def _real(value, name, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + ' must be a finite real number.')
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ValueError(name + ' must be a finite real number.') from None
    if not math.isfinite(value):
        raise ValueError(name + ' must be a finite real number.')
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(name + (' must be positive.' if positive else ' must be nonnegative.'))
    return value


@dataclass(frozen=True, init=False)
class DeltaFormula:
    """An immutable signed composition, always **target minus source**.

    Unlike a molecular :class:`msbuddy.Formula`, individual counts may be
    negative.  ``DeltaFormula({'H': -1, 'Na': 1})`` is the H/Na exchange.
    ``mass`` is an atomic mass difference, with no implicit charge correction.
    """

    counts: Tuple[int, ...]

    def __init__(self, composition):
        if not isinstance(composition, Mapping):
            raise TypeError('Delta composition must be an element-count mapping.')
        for element in composition:
            if element not in MONOISOTOPIC_MASSES:
                raise UnsupportedElementError('Unsupported element %r.' % element)
        counts = tuple(_integer(composition.get(e, 0), 'Count for ' + e,
                                -_MAX_COUNT, _MAX_COUNT) for e in SUPPORTED_ELEMENTS)
        object.__setattr__(self, 'counts', counts)

    @classmethod
    def from_string(cls, text):
        """Parse e.g. ``'-H+Na'``, ``'+O-H2'``, ``'CH2'``, or ``'0'``.

        Signs apply to entire formula terms: ``-H2O`` removes H2 and O.
        Repeated terms are combined. Isotope and charge notation is not accepted.
        """
        if not isinstance(text, str):
            raise TypeError('Delta formula must be a string.')
        text = text.strip()
        if text == '0':
            return cls({})
        if not text:
            raise ValueError('Delta formula cannot be empty.')
        if text[0] not in '+-':
            text = '+' + text
        composition = {}
        position = 0
        for match in re.finditer(r'([+-])([^+-]+)', text):
            if match.start() != position:
                raise ValueError('Invalid signed formula syntax: %r.' % text)
            sign, formula = match.groups()
            multiplier = 1 if sign == '+' else -1
            for element, count in read_formula(formula).items():
                composition[element] = composition.get(element, 0) + multiplier * count
            position = match.end()
        if position != len(text):
            raise ValueError('Invalid signed formula syntax: %r.' % text)
        return cls(composition)

    @property
    def composition(self):
        """A new mapping containing nonzero signed counts."""
        return {e: n for e, n in zip(SUPPORTED_ELEMENTS, self.counts) if n}

    @property
    def mass(self):
        return math.fsum(n * MONOISOTOPIC_MASSES[e]
                         for e, n in zip(SUPPORTED_ELEMENTS, self.counts))

    @property
    def changed_atoms(self):
        """L1 size of the net change, not the number of atoms in a pathway."""
        return sum(abs(n) for n in self.counts)

    def apply_to(self, source_formula):
        """Return target counts; raise if any resulting atom count is invalid.

        Nonnegative counts are necessary, but do not establish that the target
        is a chemically valid or experimentally present molecule.
        """
        source = _neutral_composition(source_formula)
        target = {e: source.get(e, 0) + n for e, n in zip(SUPPORTED_ELEMENTS, self.counts)}
        if any(n < 0 for n in target.values()):
            raise ValueError('Delta removes more atoms than the source formula contains.')
        return _validate_composition(target)

    def __bool__(self):
        return any(self.counts)

    def __str__(self):
        return ''.join(('+' if n > 0 else '-') + e + (str(abs(n)) if abs(n) != 1 else '')
                       for e, n in zip(SUPPORTED_ELEMENTS, self.counts) if n) or '0'

    def __neg__(self):
        return DeltaFormula({e: -n for e, n in self.composition.items()})

    def __add__(self, other):
        if not isinstance(other, DeltaFormula):
            return NotImplemented
        return DeltaFormula({e: a + b for e, a, b in zip(SUPPORTED_ELEMENTS, self.counts, other.counts)})

    def __sub__(self, other):
        if not isinstance(other, DeltaFormula):
            return NotImplemented
        return self + (-other)


def _neutral_composition(value):
    if isinstance(value, str):
        return read_formula(value)
    if not isinstance(value, Mapping) and hasattr(value, 'composition'):
        if getattr(value, 'charge', 0) != 0:
            raise ValueError('source_formula must be a neutral formula.')
        value = value.composition
    return _validate_composition(value)


@dataclass(frozen=True)
class DeltaTransformation:
    """A named net change, supplied as a hypothesis rather than a mechanism.

    ``kind`` is ``'chemical'`` or ``'ion'``. Include both directed rules when
    both directions are desired. A custom list replaces the built-in rules.
    """

    name: str
    delta: DeltaFormula
    kind: str = 'chemical'

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('A transformation needs a nonempty name.')
        if not isinstance(self.delta, DeltaFormula) or not self.delta:
            raise ValueError('A transformation needs a nonzero DeltaFormula.')
        if self.kind not in ('chemical', 'ion'):
            raise ValueError("Transformation kind must be 'chemical' or 'ion'.")


def _rules():
    pairs = (
        ('O', 'oxygen addition / hydroxylation', 'oxygen removal', 'chemical'),
        ('H2', 'hydrogenation', 'dehydrogenation', 'chemical'),
        ('CH2', 'methylation / methylene addition', 'demethylation / methylene removal', 'chemical'),
        ('C2H2O', 'acetylation', 'deacetylation', 'chemical'),
        ('C6H8O6', 'glucuronidation', 'deglucuronidation', 'chemical'),
        ('SO3', 'sulfation', 'desulfation', 'chemical'),
        ('HPO3', 'phosphorylation', 'dephosphorylation', 'chemical'),
        ('H2O', 'hydration', 'dehydration / water loss', 'chemical'),
        ('CO2', 'carboxylation', 'decarboxylation / carbon dioxide loss', 'chemical'),
        ('C6H10O5', 'hexosylation', 'dehexosylation', 'chemical'),
        ('C5H8', 'prenylation / isoprenyl addition', 'isoprenyl removal', 'chemical'),
        ('F-H', 'H/F substitution', 'F/H substitution', 'chemical'),
        ('Cl-H', 'H/Cl substitution', 'Cl/H substitution', 'chemical'),
        ('Br-H', 'H/Br substitution', 'Br/H substitution', 'chemical'),
        ('I-H', 'H/I substitution', 'I/H substitution', 'chemical'),
        ('Na-H', 'proton to sodium adduct exchange', 'sodium to proton adduct exchange', 'ion'),
        ('K-H', 'proton to potassium adduct exchange', 'potassium to proton adduct exchange', 'ion'),
        ('K-Na', 'sodium to potassium adduct exchange', 'potassium to sodium adduct exchange', 'ion'),
        ('NH3', 'proton to ammonium adduct exchange', 'ammonium to proton adduct exchange', 'ion'),
    )
    result = []
    for text, forward, reverse, kind in pairs:
        delta = DeltaFormula.from_string(text)
        result += [DeltaTransformation(forward, delta, kind), DeltaTransformation(reverse, -delta, kind)]
    return tuple(result)


DEFAULT_TRANSFORMATIONS = _rules()


@dataclass(frozen=True)
class TransformationHypothesis:
    """One shortest additive explanation; step order and intermediates untested."""

    steps: Tuple[str, ...]
    kinds: Tuple[str, ...]


@dataclass(frozen=True)
class DeltaPhenomenon:
    """A possible isotope effect, separate from elemental formula candidates."""

    name: str
    theoretical_mass: float
    mass_error: float


@dataclass(frozen=True)
class DeltaCandidate:
    delta: DeltaFormula
    mass_error: float
    mass_error_ppm: Optional[float]
    hypotheses: Tuple[TransformationHypothesis, ...]
    target_composition: Optional[dict] = None
    ion_delta: Optional[DeltaFormula] = None
    target_mass_error: Optional[float] = None
    heuristic_score: Optional[float] = None

    @property
    def formula(self):
        """Canonical signed formula text, e.g. ``'-H+Na'``."""
        return str(self.delta)

    @property
    def target_formula(self):
        return (formula_to_string(self.target_composition)
                if self.target_composition is not None else None)

    @property
    def mass(self):
        return self.delta.mass


@dataclass(frozen=True)
class DeltaMassResult(Sequence):
    """All matches inside the declared search domain, in heuristic order.

    Index/iterate this object for candidates. ``complete`` means exhaustive
    within ``element_ranges`` and ``max_changed_atoms``, never unrestricted.
    Rank is not a calibrated probability, FDR estimate, or reaction evidence.
    """

    candidates: Tuple[DeltaCandidate, ...]
    delta_mass: float
    tolerance_da: float
    element_ranges: dict
    max_changed_atoms: int
    max_changed_elements: int
    phenomena: Tuple[DeltaPhenomenon, ...] = ()
    mass_space: str = 'atomic'
    source_neutral_mass: Optional[float] = None
    target_neutral_mass: Optional[float] = None
    observed_mz_difference: Optional[float] = None
    adduct_delta: Optional[DeltaFormula] = None
    notes: Tuple[str, ...] = ()
    complete: bool = True
    source_mass_error: Optional[float] = None
    source_adduct: Optional[str] = None
    target_adduct: Optional[str] = None
    source_charge: Optional[int] = None
    target_charge: Optional[int] = None
    source_multiplier: Optional[int] = None
    target_multiplier: Optional[int] = None
    source_tolerance_da: Optional[float] = None
    target_tolerance_da: Optional[float] = None
    max_steps: int = 2
    max_half_states: int = 2000000
    max_candidates: int = 10000
    transformations: Tuple[DeltaTransformation, ...] = ()
    preset: str = 'compact'
    source_composition: Optional[dict] = None
    source_mz: Optional[float] = None
    target_mz: Optional[float] = None
    ranking_method: str = 'rules'

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, index):
        return self.candidates[index]

    def to_dict(self):
        """Return a detached JSON-safe record, including all search constraints.

        No probability or FDR is fabricated from a mass-error/complexity rank.
        ``truncated`` is always False: resource limits raise instead of emitting
        an incomplete list. Slice ``result.candidates`` explicitly for display.
        """
        return {
            'schema': 'msbuddy.signed_delta.v1',
            'delta_mass': self.delta_mass,
            'mass_space': self.mass_space,
            'tolerance_da': self.tolerance_da,
            'complete_within_bounds': self.complete,
            'truncated': False,
            'candidate_count': len(self),
            'source_composition': dict(self.source_composition) if self.source_composition is not None else None,
            'ranking': self.ranking_method,
            'ranking_description': (
                'heuristic: 0.8*log1p(changed_atoms)+0.2*changed_elements-1.5*named_prior'
                '+0.5*min(10,max(0,-target_DBE)); lower is preferred; not a probability or FDR'
                if self.ranking_method == 'balanced' else
                'heuristic: zero change, shortest named change, atom count, absolute mass error'),
            'search': {
                'preset': self.preset,
                'element_ranges': {e: list(bounds) for e, bounds in self.element_ranges.items()},
                'max_changed_atoms': self.max_changed_atoms,
                'max_changed_elements': self.max_changed_elements,
                'max_steps': self.max_steps,
                'max_half_states': self.max_half_states,
                'max_candidates': self.max_candidates,
                'transformations': [{'name': rule.name, 'delta': rule.delta.composition,
                                     'kind': rule.kind} for rule in self.transformations],
            },
            'network': {
                'source_mz': self.source_mz, 'target_mz': self.target_mz,
                'source_neutral_mass': self.source_neutral_mass,
                'target_neutral_mass': self.target_neutral_mass,
                'source_mass_error': self.source_mass_error,
                'observed_mz_difference': self.observed_mz_difference,
                'source_adduct': self.source_adduct, 'target_adduct': self.target_adduct,
                'source_charge': self.source_charge, 'target_charge': self.target_charge,
                'source_multiplier': self.source_multiplier, 'target_multiplier': self.target_multiplier,
                'source_tolerance_da': self.source_tolerance_da,
                'target_tolerance_da': self.target_tolerance_da,
                'adduct_delta': self.adduct_delta.composition if self.adduct_delta is not None else None,
            },
            'candidates': [
                {'formula': row.formula, 'delta': row.delta.composition, 'mass': row.mass,
                 'mass_error': row.mass_error, 'mass_error_ppm': row.mass_error_ppm,
                 'target_formula': row.target_formula,
                 'target_composition': dict(row.target_composition) if row.target_composition is not None else None,
                 'target_mass_error': row.target_mass_error,
                 'heuristic_score': row.heuristic_score,
                 'ion_delta': row.ion_delta.composition if row.ion_delta is not None else None,
                 'hypotheses': [{'steps': list(item.steps), 'kinds': list(item.kinds)} for item in row.hypotheses]}
                for row in self
            ],
            'phenomena': [{'name': item.name, 'theoretical_mass': item.theoretical_mass,
                           'mass_error': item.mass_error} for item in self.phenomena],
            'notes': list(self.notes),
        }


class DeltaSearchSpaceError(ValueError):
    """Search exceeds an explicit limit; no partial answer is returned."""


def _normalize_ranges(element_ranges, source, max_changed_atoms):
    if element_ranges is None:
        element_ranges = DEFAULT_DELTA_RANGES
    elif not isinstance(element_ranges, Mapping):
        raise TypeError('element_ranges must be an element to (lower, upper) mapping.')
    for e in element_ranges:
        if e not in MONOISOTOPIC_MASSES:
            raise UnsupportedElementError('Unsupported element %r.' % e)
    result = []
    for e in SUPPORTED_ELEMENTS:
        bounds = element_ranges.get(e, (0, 0))
        if isinstance(bounds, (str, bytes)) or not hasattr(bounds, '__len__') or len(bounds) != 2:
            raise ValueError('Each element range must contain lower and upper integer bounds.')
        lower = _integer(bounds[0], e + ' lower bound', -_MAX_COUNT, _MAX_COUNT)
        upper = _integer(bounds[1], e + ' upper bound', -_MAX_COUNT, _MAX_COUNT)
        if lower > upper:
            raise ValueError('Element lower bound exceeds upper bound: ' + e)
        lower, upper = max(lower, -max_changed_atoms), min(upper, max_changed_atoms)
        if source is not None:
            lower = max(lower, -source.get(e, 0))
            upper = min(upper, _MAX_COUNT - source.get(e, 0))
        result.append((lower, upper))
    return tuple(result)


def _partition_ranges(ranges):
    groups, sizes = [[], []], [1, 1]
    for i in sorted(range(len(ranges)), key=lambda i: (-(ranges[i][1] - ranges[i][0] + 1), i)):
        group = 0 if sizes[0] <= sizes[1] else 1
        groups[group].append(i)
        sizes[group] *= ranges[i][1] - ranges[i][0] + 1
    return tuple(tuple(sorted(g)) for g in groups), tuple(sizes)


@lru_cache(maxsize=4)
def _half_space(indices, ranges, max_changed_atoms, max_changed_elements):
    """Build one bounded half; caller checks cardinality before allocation."""
    counts = np.empty((1, 0), dtype=np.int16)
    costs = np.zeros(1, dtype=np.int32)
    element_costs = np.zeros(1, dtype=np.int8)
    for index in indices:
        values = np.arange(ranges[index][0], ranges[index][1] + 1, dtype=np.int16)
        count = len(values)
        new_costs = np.repeat(costs, count) + np.tile(np.abs(values.astype(np.int32)), len(costs))
        new_element_costs = np.repeat(element_costs, count) + np.tile(values != 0, len(costs))
        next_counts = np.column_stack((np.repeat(counts, count, axis=0), np.tile(values, len(counts))))
        keep = (new_costs <= max_changed_atoms) & (new_element_costs <= max_changed_elements)
        counts, costs, element_costs = next_counts[keep], new_costs[keep], new_element_costs[keep]
    masses = counts.astype(np.float64) @ _MASSES[list(indices)]
    order = np.argsort(masses, kind='stable')
    masses, counts, costs, element_costs = masses[order], counts[order], costs[order], element_costs[order]
    for array in (masses, counts, costs, element_costs):
        array.flags.writeable = False
    return masses, counts, costs, element_costs


@lru_cache(maxsize=8)
def _hypothesis_map(transformations, max_steps):
    # This deliberately describes shortest unordered additive explanations,
    # not enumerated reaction paths or chemically checked intermediates.
    result, seen = {}, {}
    for length in range(1, max_steps + 1):
        for selection in combinations_with_replacement(range(len(transformations)), length):
            rules = tuple(transformations[i] for i in selection)
            counts = tuple(sum(rule.delta.counts[i] for rule in rules) for i in range(12))
            if not any(counts):
                continue
            if counts in result and len(result[counts][0].steps) < length:
                continue
            explanation = TransformationHypothesis(tuple(rule.name for rule in rules),
                                                    tuple(rule.kind for rule in rules))
            known = seen.setdefault(counts, set())
            if explanation not in known:
                known.add(explanation)
                result.setdefault(counts, []).append(explanation)
    return {key: tuple(value) for key, value in result.items()}


def _transformation_rules(transformations, include_ion, max_steps):
    if transformations is None:
        transformations = DEFAULT_TRANSFORMATIONS
    try:
        transformations = tuple(islice(transformations, 100001))
    except TypeError:
        raise TypeError('transformations must be an iterable of DeltaTransformation objects.') from None
    if len(transformations) > 100000:
        raise DeltaSearchSpaceError('Too many transformation rules; supply at most 100000 finite rules.')
    if not all(isinstance(rule, DeltaTransformation) for rule in transformations):
        raise TypeError('transformations must contain DeltaTransformation objects.')
    transformations = tuple(dict.fromkeys(rule for rule in transformations if include_ion or rule.kind != 'ion'))
    if sum(math.comb(len(transformations) + n - 1, n) for n in range(1, max_steps + 1)
           if transformations) > 100000:
        raise DeltaSearchSpaceError('Too many transformation combinations; reduce rules or max_steps.')
    return transformations


def _isotope_hypotheses(delta_mass, tolerance, scales=(1.,)):
    result = []
    for scale in sorted(set(scales)):
        for count in (-2, -1, 1, 2):
            mass = count * CARBON_ISOTOPE_DIFFERENCE * scale
            if abs(mass - delta_mass) <= tolerance:
                result.append(DeltaPhenomenon('%+d 13C-like isotope offset (mass scale %g)' % (count, scale),
                                             mass, mass - delta_mass))
    return tuple(result)


def annotate_delta_mass(delta_mass, *, tolerance=0.005, ppm=False, reference_mass=None,
                        preset='compact', element_ranges=None, max_changed_atoms=None,
                        max_changed_elements=None, source_formula=None,
                        transformations=None, max_steps=2, include_ion=True,
                        max_half_states=2000000, max_candidates=10000):
    """Enumerate signed formulas for a directed atomic mass difference in Da.

    ``delta_mass`` is target minus source, **not automatically delta m/z**.
    Default tolerance is absolute Da. For ppm, an explicit positive
    ``reference_mass`` is mandatory; dividing by a tiny delta mass is unsafe.
    Prefer :func:`annotate_network_delta` when both precursor m/z are known.

    A supplied ``element_ranges`` replaces the defaults: omitted elements are
    fixed at zero. ``max_changed_atoms`` bounds sum(abs(net atom counts)).
    ``max_changed_elements`` bounds the number of nonzero element changes.
    ``preset='compact'`` uses 24 atoms and 4 element types; ``'broad'`` uses
    wider ranges with 40 atoms and 6 types. Explicit limits override the preset.
    Source constraints only require nonnegative, representable target counts.
    Search is exhaustive in this domain. Exceeding ``max_half_states`` or
    ``max_candidates`` raises :class:`DeltaSearchSpaceError`, without truncation.

    Ordering prefers zero delta, named shortest net-change hypotheses, then
    fewer changed atoms and lower absolute mass error. It is heuristic only.
    ``max_steps`` (0 to 3) controls hypothesis labels, not formula enumeration.
    """
    delta_mass = _real(delta_mass, 'delta_mass')
    tolerance = _real(tolerance, 'tolerance', nonnegative=True)
    if not isinstance(ppm, (bool, np.bool_)) or not isinstance(include_ion, (bool, np.bool_)):
        raise ValueError('ppm and include_ion must be boolean.')
    if reference_mass is not None:
        reference_mass = _real(reference_mass, 'reference_mass', positive=True)
    if ppm and reference_mass is None:
        raise ValueError('ppm needs an explicit reference_mass; do not scale tolerance by delta mass.')
    tolerance_da = tolerance * reference_mass * 1e-6 if ppm else tolerance
    if not math.isfinite(tolerance_da):
        raise ValueError('Converted tolerance is not finite.')
    if not isinstance(preset, str) or preset not in _PRESETS:
        raise ValueError("preset must be 'compact' or 'broad'.")
    preset_ranges, preset_atoms, preset_elements = _PRESETS[preset]
    if element_ranges is None:
        element_ranges = preset_ranges
    if max_changed_atoms is None:
        max_changed_atoms = preset_atoms
    if max_changed_elements is None:
        max_changed_elements = preset_elements
    max_changed_atoms = _integer(max_changed_atoms, 'max_changed_atoms', 0, _MAX_COUNT)
    max_changed_elements = _integer(max_changed_elements, 'max_changed_elements', 0, 12)
    max_steps = _integer(max_steps, 'max_steps', 0, 3)
    max_half_states = _integer(max_half_states, 'max_half_states', 1)
    max_candidates = _integer(max_candidates, 'max_candidates', 1)
    source = _neutral_composition(source_formula) if source_formula is not None else None
    ranges = _normalize_ranges(element_ranges, source, max_changed_atoms)
    rules = _transformation_rules(transformations, include_ion, max_steps)
    hypotheses = _hypothesis_map(rules, max_steps)
    result = []
    if all(lower <= upper for lower, upper in ranges):
        groups, sizes = _partition_ranges(ranges)
        if any(size > max_half_states for size in sizes):
            raise DeltaSearchSpaceError('Half-space cardinalities %s exceed max_half_states=%d; '
                                       'tighten element_ranges or raise the limit explicitly.' % (sizes, max_half_states))
        left_mass, left_counts, left_cost, left_elements = _half_space(groups[0], ranges, max_changed_atoms, max_changed_elements)
        right_mass, right_counts, right_cost, right_elements = _half_space(groups[1], ranges, max_changed_atoms, max_changed_elements)
        # Pad only the index lookup for float64 summation roundoff; exact fsum
        # candidate masses are checked below against the requested tolerance.
        # Half-space masses are sorted, so their endpoints give the largest
        # magnitude without a Python scan of potentially millions of states.
        scale = max(1., abs(delta_mass),
                    max(abs(left_mass[0]), abs(left_mass[-1])) if len(left_mass) else 0.,
                    max(abs(right_mass[0]), abs(right_mass[-1])) if len(right_mass) else 0.)
        pad = np.finfo(np.float64).eps * scale * 32
        lower = np.searchsorted(right_mass, delta_mass - tolerance_da - left_mass - pad, side='left')
        upper = np.searchsorted(right_mass, delta_mass + tolerance_da - left_mass + pad, side='right')
        for i in np.flatnonzero(upper > lower):
            js = np.arange(lower[i], upper[i])
            js = js[(right_cost[js] + left_cost[i] <= max_changed_atoms)
                    & (right_elements[js] + left_elements[i] <= max_changed_elements)]
            for j in js:
                counts = [0] * 12
                for index, count in zip(groups[0], left_counts[i]):
                    counts[index] = int(count)
                for index, count in zip(groups[1], right_counts[j]):
                    counts[index] = int(count)
                delta = DeltaFormula(dict(zip(SUPPORTED_ELEMENTS, counts)))
                error = delta.mass - delta_mass
                if abs(error) > tolerance_da:
                    continue
                target = delta.apply_to(source) if source is not None else None
                result.append(DeltaCandidate(delta, error,
                    error / reference_mass * 1e6 if reference_mass is not None else None,
                    hypotheses.get(delta.counts, ()), target))
                if len(result) > max_candidates:
                    raise DeltaSearchSpaceError('More than max_candidates=%d matches; tighten the '
                                               'search domain/tolerance or raise the limit explicitly.' % max_candidates)
    result.sort(key=lambda item: (0 if not item.delta else 1 if item.hypotheses else 2,
                                 len(item.hypotheses[0].steps) if item.hypotheses else 0,
                                 item.delta.changed_atoms, abs(item.mass_error), item.delta.counts))
    return DeltaMassResult(candidates=tuple(result), delta_mass=delta_mass, tolerance_da=tolerance_da,
                           element_ranges=dict(zip(SUPPORTED_ELEMENTS, ranges)),
                           max_changed_atoms=max_changed_atoms, max_changed_elements=max_changed_elements,
                           phenomena=_isotope_hypotheses(delta_mass, tolerance_da),
                           notes=('Complete only within the reported signed bounds, net atom-change limit, and changed-element limit.',
                                  'Ranks and transformation labels are hypotheses, not probabilities or validated pathways.'),
                           max_steps=max_steps, max_half_states=max_half_states, max_candidates=max_candidates,
                           transformations=rules, preset=preset,
                           source_composition=dict(source) if source is not None else None)


def _adduct(value):
    # Reparse even supplied Adduct objects, rather than trusting mutable state.
    from .base import Adduct, _expand_adduct_aliases
    if isinstance(value, Adduct):
        value = value.string
    if not isinstance(value, str):
        raise TypeError('Each adduct must be an explicit string or Adduct object.')
    value = value.replace(' ', '')
    expanded = _expand_adduct_aliases(value)
    # Adduct historically rejects bare [M]+; those molecular/radical ions have
    # a well-defined charge here and need no arbitrary H default.
    match = re.fullmatch(r'\[(\d*)M((?:[+-][^\[\]]+)?)\](\d*)([+-])', expanded)
    if match is None:
        raise ValueError('Invalid adduct: %r.' % value)
    multiplier, expression, charge, sign = match.groups()
    multiplier = int(multiplier) if multiplier else 1
    charge = int(charge) if charge else 1
    if multiplier <= 0 or charge <= 0 or multiplier > _MAX_COUNT or charge > _MAX_COUNT:
        raise ValueError('Invalid adduct multiplier or charge.')
    net = DeltaFormula.from_string(expression) if expression else DeltaFormula({})
    return multiplier, charge if sign == '+' else -charge, net


def _balanced_candidates(candidates):
    """Frozen transparent optional scorer; never filters a candidate.

    Coefficients were selected on a separate development edge panel. The zero
    mass coefficient was retained explicitly to match that frozen calculation.
    Soft target DBE is a heuristic; DBE is never applied to signed deltas.
    """
    if not candidates:
        return ()
    features = []
    for candidate in candidates:
        delta, target = candidate.delta, candidate.target_composition
        dbe = (target.get('C', 0) + 1 + (target.get('N', 0) + target.get('P', 0)) / 2.
               - sum(target.get(e, 0) for e in ('H', 'F', 'Cl', 'Br', 'I', 'Na', 'K')) / 2.)
        prior = (1. if not delta else 1. / len(candidate.hypotheses[0].steps)
                 if candidate.hypotheses else 0.)
        features.append((0., math.log1p(delta.changed_atoms), len(delta.composition),
                         prior, min(10., max(0., -dbe))))
    scores = np.asarray(features, dtype=np.float64) @ np.asarray((0., .8, .2, -1.5, .5))
    # The established rules order is retained exactly for equal scores.
    order = np.argsort(scores, kind='stable')
    return tuple(replace(candidates[i], heuristic_score=float(scores[i])) for i in order)


def annotate_network_delta(source_mz, target_mz, source_adduct, target_adduct, *,
                           mz_tolerance=5., ppm=True, source_formula=None,
                           ranking='rules',
                           **search_options):
    """Annotate the neutral formula difference between two explicit ion nodes.

    For adduct ``[nM+A]z``, neutral mass is
    ``(|z| * mz - mass(A) + z * electron_mass) / n``.
    Both m/z errors are propagated conservatively by summing their absolute
    neutral-mass bounds. ``mz_tolerance`` is either one value for both nodes or
    a two-value sequence ``(source_tolerance, target_tolerance)``.

    The returned ``delta`` is the neutral difference. ``adduct_delta`` is the
    signed adduct exchange. ``ion_delta`` on each candidate is the atomic ion
    composition difference; with unequal multimers this requires a source
    formula and is otherwise None. Electron corrections are not atoms.
    When a source formula is given, both that formula and each propagated
    target must independently match their own precursor mass tolerance.
    ``ranking='rules'`` is the default. ``'balanced'`` is an optional transparent
    complexity/named-change/soft-target-DBE score, requiring a known source
    formula. It does not consistently improve every chemistry subgroup and is
    neither a probability nor an FDR estimate. No candidate is removed by it.
    No adduct guessing is performed. Modified cosine alone does not establish
    that an edge is a reaction, an ion-identity relationship, or even genuine.
    """
    source_mz = _real(source_mz, 'source_mz', positive=True)
    target_mz = _real(target_mz, 'target_mz', positive=True)
    if not isinstance(ppm, (bool, np.bool_)):
        raise ValueError('ppm must be boolean.')
    if not isinstance(ranking, str) or ranking not in ('rules', 'balanced'):
        raise ValueError("ranking must be 'rules' or 'balanced'.")
    if ranking == 'balanced' and source_formula is None:
        raise ValueError("ranking='balanced' requires a known source_formula.")
    if isinstance(mz_tolerance, Real):
        tolerances = (mz_tolerance, mz_tolerance)
    else:
        if isinstance(mz_tolerance, (str, bytes)):
            raise ValueError('mz_tolerance must be a number or two-value sequence.')
        try:
            tolerances = tuple(islice(mz_tolerance, 3))
        except TypeError:
            raise ValueError('mz_tolerance must be a number or two-value sequence.') from None
        if len(tolerances) != 2:
            raise ValueError('mz_tolerance must contain source and target values.')
    tolerances = tuple(_real(value, 'mz_tolerance', nonnegative=True) for value in tolerances)
    src_n, src_z, src_a = _adduct(source_adduct)
    dst_n, dst_z, dst_a = _adduct(target_adduct)
    source_mass = (abs(src_z) * source_mz - src_a.mass + src_z * ELECTRON_MASS) / src_n
    target_mass = (abs(dst_z) * target_mz - dst_a.mass + dst_z * ELECTRON_MASS) / dst_n
    if not math.isfinite(source_mass + target_mass) or min(source_mass, target_mass) <= 0:
        raise ValueError('Adduct correction produces invalid neutral precursor masses.')
    source_tol, target_tol = (tol * (mz * 1e-6 if ppm else 1.) * abs(charge) / mult
                             for tol, mz, charge, mult in
                             zip(tolerances, (source_mz, target_mz), (src_z, dst_z), (src_n, dst_n)))
    source = _neutral_composition(source_formula) if source_formula is not None else None
    if source is not None:
        exact_source_mass = DeltaFormula(source).mass
        if abs(exact_source_mass - source_mass) > source_tol + np.spacing(max(1., abs(source_mass))) * 16:
            raise ValueError('source_formula does not match source m/z and adduct within its tolerance.')
        if any(src_n * source.get(e, 0) + n < 0 for e, n in src_a.composition.items()):
            raise ValueError('source_adduct removes more atoms than source_formula contains.')
    reserved = {'tolerance', 'ppm', 'reference_mass', 'include_ion', 'source_formula'}
    if reserved.intersection(search_options):
        raise TypeError('Use mz_tolerance/ppm/source_formula for network mass handling; '
                        'do not pass ' + ', '.join(sorted(reserved.intersection(search_options))))
    result = annotate_delta_mass(target_mass - source_mass, tolerance=source_tol + target_tol,
                                 source_formula=source, include_ion=False, **search_options)
    exchange = dst_a - src_a
    candidates = []
    for candidate in result:
        target_mass_error = None
        if source is not None:
            target_mass_error = DeltaFormula(candidate.target_composition).mass - target_mass
            if abs(target_mass_error) > target_tol + np.spacing(max(1., abs(target_mass))) * 16:
                continue
            # Impossible adduct subtraction is rejected even when the neutral
            # formula itself remains nonnegative.
            if any(dst_n * candidate.target_composition.get(e, 0) + n < 0
                   for e, n in dst_a.composition.items()):
                continue
            ion_counts = {e: dst_n * candidate.target_composition.get(e, 0) - src_n * source.get(e, 0)
                            + exchange.composition.get(e, 0) for e in SUPPORTED_ELEMENTS}
            ion_delta = DeltaFormula(ion_counts)
        elif src_n == dst_n:
            ion_delta = DeltaFormula({e: dst_n * n + exchange.composition.get(e, 0)
                                      for e, n in zip(SUPPORTED_ELEMENTS, candidate.delta.counts)})
        else:
            ion_delta = None
        candidates.append(DeltaCandidate(candidate.delta, candidate.mass_error, None,
                                         candidate.hypotheses, candidate.target_composition, ion_delta,
                                         target_mass_error))
    notes = result.notes + (
        'Neutral mass tolerance is the sum of both propagated precursor error bounds.',
        'Adduct assignments are inputs, not inferred from mass or modified cosine.',
        'Zero neutral change includes isomers and cannot prove that two nodes are the same molecule.',
        'Ion mass difference includes an additional -(target_charge-source_charge)*electron_mass.',
    )
    if src_n != dst_n and source is None:
        notes += ('Ion elemental difference is unavailable for unequal multimers without a source formula.',)
    if source is not None:
        notes += ('Propagated targets independently satisfy the target precursor mass tolerance.',)
    if ranking == 'balanced':
        candidates = _balanced_candidates(candidates)
        notes += ('Balanced ranking trades off subgroup performance; its soft target-valence score is not a hard validity rule.',)
    return DeltaMassResult(candidates=tuple(candidates), delta_mass=result.delta_mass,
                           tolerance_da=result.tolerance_da, element_ranges=result.element_ranges,
                           max_changed_atoms=result.max_changed_atoms,
                           max_changed_elements=result.max_changed_elements,
                           phenomena=_isotope_hypotheses(result.delta_mass, result.tolerance_da,
                                                         (1. / src_n, 1. / dst_n)),
                           mass_space='neutral', source_neutral_mass=source_mass,
                           target_neutral_mass=target_mass, observed_mz_difference=target_mz - source_mz,
                           adduct_delta=exchange, notes=notes,
                           source_mass_error=exact_source_mass - source_mass if source is not None else None,
                           source_adduct=source_adduct if isinstance(source_adduct, str) else source_adduct.string,
                           target_adduct=target_adduct if isinstance(target_adduct, str) else target_adduct.string,
                           source_charge=src_z, target_charge=dst_z,
                           source_multiplier=src_n, target_multiplier=dst_n,
                           source_tolerance_da=source_tol, target_tolerance_da=target_tol,
                           max_steps=result.max_steps, max_half_states=result.max_half_states,
                           max_candidates=result.max_candidates, transformations=result.transformations,
                           preset=result.preset,
                           source_composition=dict(source) if source is not None else None,
                           source_mz=source_mz, target_mz=target_mz, ranking_method=ranking)
