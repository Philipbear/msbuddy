"""Posterior error estimates for one top-ranked annotation per spectrum."""
from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BatchFDRResult:
    """Batch estimates aligned with the input queries, including empty queries.

    ``q_values`` are the mean posterior error among annotations at least as
    confident as each query. Equal probabilities receive the same value and
    are selected together. These are model-based estimates, not a standalone
    frequentist FDR guarantee; their validity depends on the input calibration.
    """

    posterior_error_probabilities: Tuple[Optional[float], ...]
    q_values: Tuple[Optional[float], ...]
    selected: Tuple[bool, ...]
    fdr_threshold: float
    eligible_count: int
    selected_count: int
    expected_false_discoveries: float
    estimated_fdr: float

    def to_dict(self):
        """Return JSON-safe values; a missing annotation is represented by null."""
        return {
            'posterior_error_probabilities': list(self.posterior_error_probabilities),
            'q_values': list(self.q_values), 'selected': list(self.selected),
            'fdr_threshold': self.fdr_threshold, 'total_queries': len(self.selected),
            'eligible_count': self.eligible_count, 'selected_count': self.selected_count,
            'expected_false_discoveries': self.expected_false_discoveries,
            'estimated_fdr': self.estimated_fdr,
            'method': 'posterior_expected_fdp',
            'scope': 'One top-ranked formula per spectrum; calibrated posterior probabilities required.',
        }


def estimate_batch_fdr(probabilities: Iterable[Optional[float]],
                       fdr_threshold: float = 0.05) -> BatchFDRResult:
    """Estimate error across a complete batch of top-ranked annotations.

    Pass one calibrated top-formula probability per spectrum, or ``None`` for
    no annotation. Do not pass ranker scores, p-values, or probabilities
    renormalized to discard the absent-formula state. The input is not changed.

    The selected set is the largest complete confidence prefix whose mean
    posterior error is at most ``fdr_threshold``. Ties are never split. Compute
    this once across the full collection, not separately in processing chunks.
    An empty selected set has zero expected false discoveries and zero FDP.
    """
    if (isinstance(fdr_threshold, (bool, np.bool_)) or not isinstance(fdr_threshold, Real)
            or not math.isfinite(fdr_threshold) or not 0 <= fdr_threshold <= 1):
        raise ValueError('fdr_threshold must be a finite number between zero and one.')
    if isinstance(probabilities, (str, bytes)):
        raise ValueError('Provide one calibrated probability or None per spectrum.')
    values = list(probabilities)
    errors = [None] * len(values)
    eligible = []
    for index, probability in enumerate(values):
        if probability is None:
            continue
        if (isinstance(probability, (bool, np.bool_)) or not isinstance(probability, Real)
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError('Each probability must be a finite number in [0, 1], or None.')
        errors[index] = 1. - float(probability)
        eligible.append(index)
    q_values = [None] * len(values)
    selected = [False] * len(values)
    if eligible:
        indices = np.asarray(eligible, dtype=np.int64)
        error_array = np.asarray([errors[i] for i in indices], dtype=float)
        order = np.argsort(error_array, kind='stable')
        indices, error_array = indices[order], error_array[order]
        prefix_error = np.cumsum(error_array, dtype=float) / np.arange(1, len(indices) + 1)
        # A tied block gets the error of its complete prefix, independently of
        # input order. This prevents partially accepting tied probabilities.
        endpoints = np.r_[np.flatnonzero(error_array[1:] != error_array[:-1]) + 1, len(indices)]
        start = 0
        for end in endpoints:
            risk = float(np.clip(prefix_error[end - 1], 0., 1.))
            # Compare mean confidence at the cutoff to avoid rejecting an exact
            # decimal boundary such as p=.95 solely because 1-p rounds upward.
            accepted = ((1. - risk) >= (1. - float(fdr_threshold)))
            if fdr_threshold == 0:
                accepted = bool(error_array[end - 1] == 0)
            for index in indices[start:end]:
                q_values[index] = risk
                selected[index] = bool(accepted)
            start = int(end)
    count = sum(selected)
    expected = math.fsum(error for error, accepted in zip(errors, selected) if accepted)
    return BatchFDRResult(tuple(errors), tuple(q_values), tuple(selected), float(fdr_threshold),
                          len(eligible), count, expected, expected / count if count else 0.)
