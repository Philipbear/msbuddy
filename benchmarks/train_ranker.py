"""Compare cheap tree and spectral neural rankers on formula-disjoint splits.

Research feasibility experiment, not a replacement for released models. Uses
pre-pruning candidate exports from candidate_audit. Inference features contain
no reference atom counts. Training labels are used only for supervision and
sampling negatives. Test formulas never enter training, normalization, or
checkpoint selection. All pools share the same saved split manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from msbuddy import Adduct, Formula
from msbuddy._acquisition import instrument_family
from msbuddy.utils import read_formula


def formula_split(formula, seed=20261113):
    canonical = str(Formula(read_formula(formula)))
    digest = hashlib.blake2b('{}:{}'.format(seed, canonical).encode(), digest_size=8).digest()
    value = int.from_bytes(digest, 'big') / 2**64
    return 'train' if value < .6 else 'validation' if value < .8 else 'test'


def encode_spectrum(record, resolution=.2, maximum=1500):
    bins = int(maximum / resolution)
    result = np.zeros((2, bins), dtype=np.float32)
    mzs = np.asarray(record['mzs'], dtype=float)
    intensities = np.asarray(record['intensities'], dtype=float)
    valid = np.isfinite(mzs) & np.isfinite(intensities) & (intensities > 0)
    mzs, intensities = mzs[valid], intensities[valid]
    if len(intensities):
        intensities = np.sqrt(intensities / intensities.max())
    for channel, masses in enumerate((mzs, float(record['precursor_mz']) - mzs)):
        indices = np.floor(masses / resolution).astype(np.int64)
        keep = (indices >= 0) & (indices < bins)
        np.maximum.at(result[channel], indices[keep], intensities[keep])
    return result


def encode_metadata(record):
    positive = str(record.get('polarity', 'positive')).lower() not in ('negative', 'neg', 'n', '-')
    adduct = Adduct(record['adduct'], positive, report_invalid=True)
    energy = record.get('collision_energy_ev')
    if energy is None:
        energy = []
    if isinstance(energy, (int, float)):
        energy = [energy]
    energy = [float(value) for value in energy if value is not None and np.isfinite(value)]
    instrument = instrument_family(record.get('instrument'))
    return np.array([float(record['precursor_mz']) / 1500, adduct.charge / 2, adduct.m / 2,
                     np.mean(energy) / 100 if energy else 0, bool(energy),
                     instrument == 'orbitrap', instrument == 'tof'],
                    dtype=np.float32)


def rank_metrics(rows, predictions):
    attempted = len(rows)
    correct = {1: 0, 5: 0, 350: 0}
    available = completed = 0
    for row, scores in zip(rows, predictions):
        if scores is None:
            continue
        completed += 1
        candidates = row.get('proposal_candidates', [])
        if len(scores) != len(candidates):
            raise ValueError('Candidate count differs from score count.')
        order = np.argsort(-np.asarray(scores), kind='stable')
        truth_index = next((i for i, formula in enumerate(candidates) if formula == row['formula']), None)
        available += truth_index is not None
        if truth_index is not None:
            rank = int(np.where(order == truth_index)[0][0]) + 1
            for k in correct:
                correct[k] += rank <= k
    return {'attempted': attempted, 'completed': completed, 'candidate_available': available,
            'top1_correct': correct[1], 'top5_correct': correct[5], 'retained350_correct': correct[350],
            'top1_all_attempts': correct[1] / attempted if attempted else None,
            'top5_all_attempts': correct[5] / attempted if attempted else None,
            'top1_given_candidate': correct[1] / available if available else None}


def build_network(include_spectra=True, feature_dim=16):
    import torch
    from torch import nn

    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(2, 8, 11, stride=4, padding=5), nn.ReLU(),
                nn.Conv1d(8, 16, 7, stride=4, padding=3), nn.ReLU(),
                nn.Conv1d(16, 32, 5, stride=4, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Linear(512, 64), nn.ReLU(),
            ) if include_spectra else None
            self.scorer = nn.Sequential(nn.Linear(64 + feature_dim + 7, 64), nn.ReLU(),
                                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            self.none_scorer = nn.Linear(64 + 7, 1)
            self.formula_decoder = nn.Linear(64 + 7, 12)

        def forward(self, spectra, metadata, features, groups):
            latent = self.encoder(spectra) if self.encoder else torch.zeros((len(spectra), 64))
            shared = torch.cat((latent, metadata), dim=1)
            scores = self.scorer(torch.cat((shared[groups], features), dim=1)).squeeze(1)
            return scores, self.none_scorer(shared).squeeze(1), self.formula_decoder(shared)

    return Ranker()


def training_indices(row, rng, limit):
    count = len(row['proposal_candidates'])
    truth = next((i for i, formula in enumerate(row['proposal_candidates']) if formula == row['formula']), None)
    if count <= limit:
        return np.arange(count)
    negatives = np.array([i for i in range(count) if i != truth])
    # Include difficult candidates near the precursor mass as well as random
    # distractors. This affects supervision, never the evaluated candidate pool.
    near = negatives if row.get('feature_kind') == 'retained' else negatives[
        np.argsort(np.abs(np.asarray(row['proposal_features'])[negatives, 12]), kind='stable')]
    hard = near[:limit // 2]
    remainder = np.setdiff1d(negatives, hard, assume_unique=True)
    random = rng.choice(remainder, limit - len(hard) - int(truth is not None), replace=False)
    return np.concatenate((hard, random, [truth] if truth is not None else [])).astype(int)


def train_neural(train_rows, validation_rows, records, center, scale, *, include_spectra,
                 epochs, seed, candidate_limit):
    import torch
    from torch.nn import functional as F

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    feature_dim = len(center)
    network = build_network(include_spectra, feature_dim)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.001)
    spectra = {key: encode_spectrum(row) for key, row in records.items()}
    metadata = {}
    for key, row in records.items():
        try:
            metadata[key] = encode_metadata(row)
        except ValueError:
            # Invalid adducts are reported by the audit and never scored.
            continue
    atom_scale = np.array([80, 150, 10, 15, 20, 10, 1, 20, 1, 30, 10, 15], dtype=np.float32)

    def tensors(rows, indices):
        sizes = [len(index) for index in indices]
        matrices = [(np.asarray(row['proposal_features'], dtype=np.float32).reshape(-1, feature_dim)[index] - center) / scale
                    for row, index in zip(rows, indices)]
        x = torch.from_numpy(np.concatenate(matrices))
        groups = torch.from_numpy(np.repeat(np.arange(len(rows)), sizes))
        spec = torch.from_numpy(np.stack([spectra[row['identifier']] for row in rows]))
        env = torch.from_numpy(np.stack([metadata[row['identifier']] for row in rows]))
        return spec, env, x, groups, sizes

    def loss_for(rows, indices):
        spec, env, x, groups, sizes = tensors(rows, indices)
        scores, none, decoded = network(spec, env, x, groups)
        losses = []
        offset = 0
        for i, (row, index, size) in enumerate(zip(rows, indices, sizes)):
            candidates = [row['proposal_candidates'][j] for j in index]
            target = candidates.index(row['formula']) if row['formula'] in candidates else size
            logits = torch.cat((scores[offset:offset + size], none[i:i + 1]))
            losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target])))
            offset += size
        loss = torch.stack(losses).mean()
        if include_spectra:
            labels = np.stack([Formula(read_formula(row['formula'])).array for row in rows]).astype(np.float32)
            loss = loss + .05 * F.mse_loss(decoded, torch.from_numpy(labels / atom_scale))
        return loss

    best, best_loss, history = None, float('inf'), []
    for epoch in range(epochs):
        network.train()
        shuffled = [train_rows[i] for i in rng.permutation(len(train_rows))]
        training_loss = []
        for start in range(0, len(shuffled), 8):
            batch = shuffled[start:start + 8]
            indices = [training_indices(row, rng, candidate_limit) for row in batch]
            optimizer.zero_grad()
            loss = loss_for(batch, indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
            optimizer.step()
            training_loss.append(float(loss.detach()))
        network.eval()
        with torch.no_grad():
            values = [float(loss_for([row], [np.arange(len(row['proposal_candidates']))]))
                      for row in validation_rows]
        validation_loss = float(np.mean(values))
        history.append({'epoch': epoch + 1, 'training_loss': float(np.mean(training_loss)),
                        'validation_loss': validation_loss})
        if validation_loss < best_loss:
            best, best_loss = copy.deepcopy(network.state_dict()), validation_loss
    network.load_state_dict(best)
    network.eval()

    def predict(rows):
        predictions = []
        with torch.no_grad():
            for row in rows:
                if row['status'] != 'ok':
                    predictions.append(None)
                    continue
                indices = [np.arange(len(row['proposal_candidates']))]
                spec, env, x, groups, _ = tensors([row], indices)
                scores, _, _ = network(spec, env, x, groups)
                predictions.append(scores.numpy())
        return predictions

    return network, history, predict


def _tree_worker(connection, train_x, train_y, groups, inputs, model_path, seed):
    """Keep LightGBM and PyTorch's native thread runtimes in separate processes."""
    import lightgbm as lgb

    try:
        tree = lgb.LGBMRanker(n_estimators=100, num_leaves=15, min_child_samples=10,
                              learning_rate=.05, n_jobs=1, verbosity=-1, random_state=seed)
        tree.fit(train_x, train_y, group=groups)
        tree.booster_.save_model(model_path)
        predictions = {key: [None if matrix is None else tree.predict(matrix) if len(matrix) else np.array([])
                             for matrix in matrices] for key, matrices in inputs.items()}
        connection.send({'predictions': predictions})
    except Exception:
        connection.send({'error': traceback.format_exc()})
    finally:
        connection.close()


def fit_tree(train_x, train_y, groups, inputs, model_path, seed):
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=_tree_worker,
                               args=(child, train_x, train_y, groups, inputs, str(model_path), seed))
    process.start()
    child.close()
    try:
        if not parent.poll(120):
            raise RuntimeError('Tree training exceeded 120 seconds.')
        try:
            response = parent.recv()
        except EOFError as exc:
            raise RuntimeError('Tree training process failed.') from exc
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response['predictions']
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        parent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--audit', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-backend', default='database')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--candidate-limit', type=int, default=256)
    parser.add_argument('--candidate-pool', choices=['proposal', 'retained'], default='proposal',
                        help='proposal uses 16 cheap features before pruning; retained uses all 53 BUDDY features')
    parser.add_argument('--seed', type=int, default=20261113)
    args = parser.parse_args()
    if args.epochs < 1 or args.candidate_limit < 2:
        parser.error('Epochs must be positive and candidate limit at least two.')
    records = {}
    excluded_labels = []
    for row in map(json.loads, args.records.read_text().splitlines()):
        identifier = str(row['identifier'])
        if identifier in records:
            parser.error('Duplicate input spectrum identifier: {}'.format(identifier))
        try:
            Formula(read_formula(row['formula']))
        except ValueError as exc:
            excluded_labels.append({'identifier': identifier, 'reason': str(exc)})
            continue
        records[identifier] = row
    rows = [json.loads(line) for path in args.audit for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row['identifier'] in records]
    feature_dim = 16 if args.candidate_pool == 'proposal' else 53
    for row in rows:
        row['feature_kind'] = args.candidate_pool
        if row['status'] == 'ok' and args.candidate_pool == 'retained':
            if row['candidates'] and 'model_features' not in row:
                parser.error('Retained pool requires an audit with ranking and --export-features.')
            row['proposal_candidates'] = row['candidates']
            row['proposal_features'] = row.get('model_features', [])
    keys = [(row['identifier'], row['backend']) for row in rows]
    if len(set(keys)) != len(keys):
        parser.error('Duplicate spectrum/backend trials in audit inputs.')
    splits = {identifier: formula_split(record['formula'], args.seed) for identifier, record in records.items()}
    eligible = [row for row in rows if row['status'] == 'ok' and row['truth_in_mass_window']
                and not row['truth_constraint_failures'] and 'proposal_features' in row]
    train_rows = [row for row in eligible if row['backend'] == args.training_backend
                  and splits[row['identifier']] == 'train']
    validation_rows = [row for row in eligible if row['backend'] == args.training_backend
                       and splits[row['identifier']] == 'validation']
    if len(train_rows) < 8 or not validation_rows:
        parser.error('Need at least eight eligible training spectra and a nonempty validation split.')
    test_rows = [row for row in rows if splits[row['identifier']] == 'test']
    if not test_rows:
        parser.error('No held-out test spectra.')
    training_formulas = {row['formula'] for row in train_rows}
    assert not training_formulas.intersection(row['formula'] for row in validation_rows)
    assert not training_formulas.intersection(str(Formula(read_formula(records[row['identifier']]['formula'])))
                                              for row in test_rows)
    rng = np.random.default_rng(args.seed)
    samples = [(row, training_indices(row, rng, args.candidate_limit)) for row in train_rows]
    matrices = [np.asarray(row['proposal_features'], dtype=np.float32).reshape(-1, feature_dim)[index]
                for row, index in samples if len(index)]
    if not matrices:
        parser.error('No generated candidate features in the training split.')
    matrix = np.concatenate(matrices)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-4] = 1
    tree_x, tree_y, groups = [], [], []
    for row, index in samples:
        if row['formula'] not in row['proposal_candidates'] or not len(index):
            continue
        env = encode_metadata(records[row['identifier']])
        features = np.asarray(row['proposal_features'], dtype=np.float32)[index]
        tree_x.append(np.column_stack((features, np.tile(env, (len(index), 1)))))
        tree_y.extend(int(row['proposal_candidates'][i] == row['formula']) for i in index)
        groups.append(len(index))
    if not groups:
        parser.error('No positive candidates in training data.')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'seed': args.seed, 'training_backend': args.training_backend,
              'training_spectra': len(train_rows), 'training_formulas': len(training_formulas),
              'validation_spectra': len(validation_rows), 'split': 'canonical formula 60/20/20 hash split',
              'excluded_labels': excluded_labels,
              'candidate_pool': args.candidate_pool, 'feature_dim': feature_dim,
              'audit_paths': [str(path.resolve()) for path in args.audit],
              'feature_mean': center.tolist(), 'feature_scale': scale.tolist(),
              'results': {}, 'neural_history': {},
              'limitation': 'Exploratory pilot. Scores are uncalibrated and are not FDR estimates. Existing production models were not replaced.'}
    by_backend = defaultdict(list)
    for row in test_rows:
        by_backend[row['backend']].append(row)
    tree_inputs = {}
    for backend, subset in by_backend.items():
        inputs = []
        for row in subset:
            if row['status'] != 'ok':
                inputs.append(None)
                continue
            features = np.asarray(row['proposal_features'], dtype=np.float32).reshape(-1, feature_dim)
            env = encode_metadata(records[row['identifier']])
            inputs.append(np.column_stack((features, np.tile(env, (len(features), 1)))))
        tree_inputs[backend] = inputs
    tree_name = 'retrained_tree' if args.candidate_pool == 'retained' else 'cheap_tree'
    tree_predictions = fit_tree(np.concatenate(tree_x), np.asarray(tree_y), groups, tree_inputs,
                                 args.output / (tree_name + '.txt'), args.seed)
    for backend, subset in by_backend.items():
        report['results'].setdefault(backend, {})[tree_name] = rank_metrics(subset, tree_predictions[backend])
        if args.candidate_pool == 'retained':
            report['results'][backend]['released_ranker'] = rank_metrics(
                subset, [row['scores'] if row['status'] == 'ok' else None for row in subset])
    import torch
    for name, include_spectra in (('neural_features', False), ('neural_spectrum', True)):
        network, history, predict = train_neural(train_rows, validation_rows, records, center, scale,
                                                 include_spectra=include_spectra, epochs=args.epochs,
                                                 seed=args.seed, candidate_limit=args.candidate_limit)
        torch.save({'state_dict': network.state_dict(), 'feature_mean': center.tolist(),
                    'feature_scale': scale.tolist(), 'include_spectra': include_spectra,
                    'feature_dim': feature_dim, 'candidate_pool': args.candidate_pool,
                    'resolution': .2, 'seed': args.seed}, args.output / (name + '.pt'))
        report['neural_history'][name] = history
        for backend, subset in by_backend.items():
            report['results'][backend][name] = rank_metrics(subset, predict(subset))
    manifest = [{'identifier': identifier, 'formula': str(Formula(read_formula(records[identifier]['formula']))),
                 'split': split} for identifier, split in sorted(splits.items())]
    (args.output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'neural_history'}, indent=2))


if __name__ == '__main__':
    main()
