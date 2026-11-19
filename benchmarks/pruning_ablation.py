"""Replay the old and corrected pruning scores on identical exported pools.

Only the MS2 evidence discarded on precursor-database duplicates is changed.
This is a deterministic candidate-retention ablation, not a ranking benchmark.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from msbuddy.ml import _calc_log_p_norm_many


def retained_truth(row, ms1_tol, *, preserve_evidence=True, cap=350):
    names = row['proposal_candidates']
    if row['formula'] not in names:
        return False
    if len(names) <= cap:
        return True
    features = np.asarray(row['proposal_features'], dtype=float)
    mass_score = _calc_log_p_norm_many(features[:, 12], ms1_tol / 3)
    if row.get('has_ms1_isotopes', False):
        scores = mass_score * .05 + features[:, 14]
    elif row.get('processed_peaks', 0):
        intensity = features[:, 13]
        if not preserve_evidence:
            intensity = intensity * (1 - features[:, 15])
        scores = mass_score + 4 * intensity
    else:
        scores = -np.abs(features[:, 12])
    kept = np.argsort(-scores, kind='stable')[:cap]
    return bool(names.index(row['formula']) in kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.audit.read_text().splitlines() if line.strip()]
    settings = json.loads(args.summary.read_text())['settings']
    groups = defaultdict(lambda: {'completed': 0, 'legacy_retained': 0, 'fixed_retained': 0,
                                  'recovered': [], 'regressed': []})
    for row in rows:
        if row['status'] != 'ok':
            continue
        tolerance = settings.get('ms1_tol') or (5 if row['instrument'] == 'orbitrap' else 10)
        legacy = retained_truth(row, tolerance, preserve_evidence=False)
        fixed = retained_truth(row, tolerance)
        if fixed != row['stages']['retained']['truth_present']:
            raise ValueError('Replay does not match recorded selection: {}'.format(row['identifier']))
        group = groups[row['backend']]
        group['completed'] += 1
        group['legacy_retained'] += int(legacy)
        group['fixed_retained'] += int(fixed)
        if fixed and not legacy:
            group['recovered'].append(row['identifier'])
        if legacy and not fixed:
            group['regressed'].append(row['identifier'])
    result = {'method': 'Paired replay on identical pre-pruning candidate pools; old merge zeros MS2 evidence on precursor database hits.',
              'audit_path': str(args.audit.resolve()), 'results': dict(groups)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
