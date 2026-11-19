"""Optional MassDesk fragment/loss search for controlled recall experiments.

This module is not used by the public annotation pipeline. Bounds depend only
on configuration and adduct, never on a reference formula. Search tolerances
are measured on the original fragment m/z or neutral-loss mass. Result caps
are explicitly reported: a capped search cannot establish exhaustive recall.
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np

from msbuddy.base import Formula, mass_e
from msbuddy.query import query_fragnl_mass
from msbuddy.utils import SUPPORTED_ELEMENTS


def load_massdesk(source_path=None):
    if source_path:
        source = Path(source_path).resolve()
        if not (source / 'massdesk' / 'core' / 'decompose.py').is_file():
            raise ValueError('--massdesk-path must contain massdesk/core/decompose.py')
        sys.path.insert(0, str(source))
    try:
        decompose = importlib.import_module('massdesk.core.decompose').decompose_mass
        element_range = importlib.import_module('massdesk.core.elements').ElementRange
    except ImportError as exc:
        raise ImportError('Install MassDesk and its dependencies, or pass --massdesk-path to its source checkout.') from exc
    return decompose, element_range


class MassDeskSearch:
    def __init__(self, config, *, source_path=None, max_results=1000, mode='de-novo',
                 peak_budget=3, decompose=None, element_range=None):
        if max_results < 1 or peak_budget < 1:
            raise ValueError('Search result and peak limits must be positive.')
        if mode not in ('de-novo', 'fallback', 'union'):
            raise ValueError('Unknown search mode: {}'.format(mode))
        if decompose is None or element_range is None:
            decompose, element_range = load_massdesk(source_path)
        self.decompose = decompose
        self.element_range = element_range
        self.config = config
        self.max_results = max_results
        self.mode = mode
        self.peak_budget = peak_budget
        self.cached_decomposition = lru_cache(maxsize=128)(self._decompose)
        self.upper = None
        self.allowed_masses = set()
        self.reset_stats()

    def reset_stats(self):
        self.stats = {'query_calls': 0, 'de_novo_calls': 0, 'truncated_queries': 0,
                      'database_queries': 0, 'de_novo_seconds': 0.0}

    def configure(self, feature):
        # Lower bounds on precursor atoms do not apply to fragment/loss atoms.
        self.upper = tuple(int(value) for value in
                           self.config.ele_upper.astype(np.int64) * feature.adduct.m +
                           np.maximum(feature.adduct.net_formula.array, 0))
        spectrum = feature.ms2_processed
        self.allowed_masses = set()
        if spectrum:
            indices = np.argsort(-spectrum.int_array, kind='stable')[:self.peak_budget]
            for index in indices:
                mz = float(spectrum.mz_array[index])
                self.allowed_masses.update((mz, feature.mz - mz))
        self.reset_stats()

    def _decompose(self, measured_mass, fragment, positive, tolerance_da, upper):
        charge = (1 if positive else -1) if fragment else 0
        target = measured_mass + charge * mass_e
        if target <= 0:
            return (), False
        ranges = [self.element_range(symbol, float(mass), 0, maximum)
                  for symbol, mass, maximum in zip(SUPPORTED_ELEMENTS, Formula.mass_arr, upper)
                  if maximum > 0]
        batch = self.decompose(target, ranges, tolerance_ppm=None,
                               tolerance_da=tolerance_da, max_results=self.max_results,
                               dbe_min=0 if fragment else None)
        formulas = tuple(Formula(item.composition, charge=charge,
                                 mass=item.mass - charge * mass_e) for item in batch.results)
        return formulas, batch.truncated

    def __call__(self, mass, fragment, positive, sodium, potassium, tolerance, ppm, db_mode, gd):
        mass = float(mass)
        self.stats['query_calls'] += 1
        database = []
        if self.mode != 'de-novo':
            self.stats['database_queries'] += 1
            database = query_fragnl_mass(mass, fragment, positive, sodium, potassium,
                                         tolerance, ppm, db_mode, gd)
            if self.mode == 'fallback' and database:
                return database
            if self.mode == 'union' and mass not in self.allowed_masses:
                return database
        tolerance_da = abs(mass) * tolerance * 1e-6 if ppm else tolerance
        started = perf_counter()
        generated, truncated = self.cached_decomposition(mass, fragment, positive,
                                                         tolerance_da, self.upper)
        self.stats['de_novo_seconds'] += perf_counter() - started
        self.stats['de_novo_calls'] += 1
        self.stats['truncated_queries'] += int(truncated)
        if not database:
            return list(generated)
        # Prefer the existing mass representation for duplicated compositions.
        seen = {tuple(formula.array) for formula in database}
        return database + [formula for formula in generated if tuple(formula.array) not in seen]
