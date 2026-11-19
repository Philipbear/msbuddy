"""Audit candidate recall and ranking on labelled JSONL spectra.

Each backend sees the same spectra and configuration. Labels are available
only to the observer and evaluation code. The production backend stays the
default; MassDesk is an optional research dependency. A persistent isolated
worker enforces per-spectrum deadlines, including decomposition and ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import platform
import sys
import traceback
from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np

from msbuddy import Adduct, Formula, MetaFeature, MsbuddyConfig, Spectrum
from msbuddy.base import mass_e
from msbuddy.cand import (gen_candidate_formula, _adduct_loss_check, _dbe_check,
                          _element_check, _o_p_check, _senior_rules)
from msbuddy.load import init_db, init_ml_models
from msbuddy.main import _gen_subformula
from msbuddy.ml import gen_ml_feature, predict_formula_probability
from msbuddy.utils import read_formula


class CandidateTrace:
    def __init__(self, formula, feature, export_proposals=False):
        self.truth = Formula(read_formula(formula)).array
        self.charged_truth = self.truth * feature.adduct.m + feature.adduct.net_formula.array
        self.stages = {}
        self.queries = []
        self.export_proposals = export_proposals
        self.proposal_candidates = []
        self.proposal_features = []

    def __call__(self, event, arrays=None, **details):
        if arrays is not None:
            count = 0
            found = False
            for array in arrays:
                count += 1
                found = found or np.array_equal(array, self.truth)
            self.stages[event] = {'count': count, 'truth_present': bool(found)}
        if event == 'pre_pruning' and self.export_proposals:
            for candidate in details['candidates']:
                self.proposal_candidates.append(str(candidate.formula))
                self.proposal_features.append([
                    *map(float, candidate.formula.array), float(candidate.mz_error),
                    float(candidate.exp_ms2_sum_int or 0), float(candidate.ms1_isotope_similarity or 0),
                    float(bool(candidate.db_existed)),
                ])
        if event == 'fragment_query':
            fragments, losses = details['fragments'], details['losses']
            # The smaller side is queried first. None/None means that lookup
            # failed and the larger side was deliberately never searched.
            loss_first = details['loss_mass'] < details['fragment_mz']
            fragment_count = None if fragments is None and loss_first else len(fragments or [])
            loss_count = None if losses is None and not loss_first else len(losses or [])
            loss_arrays = {tuple(formula.array) for formula in losses or []}
            compatible = sum(tuple(self.charged_truth - formula.array) in loss_arrays
                             for formula in fragments or [])
            self.queries.append({
                'index': details['index'], 'fragment_mz': details['fragment_mz'],
                'loss_mass': details['loss_mass'], 'fragment_matches': fragment_count,
                'loss_matches': loss_count, 'truth_compatible_pairs': compatible,
                'loss_tolerance_da': details.get('loss_tolerance_da'),
            })


def make_config(record, options):
    instrument = record.get('instrument', 'unknown')
    ms1, ms2 = (5, 10) if instrument == 'orbitrap' else (10, 20)
    return MsbuddyConfig(ppm=True, halogen=options.get('halogen', True),
                         ms1_tol=options.get('ms1_tol') or ms1,
                         ms2_tol=options.get('ms2_tol') or ms2,
                         element_ranges=options.get('element_ranges'),
                         data_dir=options.get('data_dir'))


def make_feature(record):
    polarity = str(record.get('polarity', 'positive')).lower()
    positive = polarity not in ('negative', 'neg', 'n', '-')
    adduct = Adduct(record['adduct'], positive, report_invalid=True)
    if adduct.pos_mode != positive:
        raise ValueError('Adduct and reported polarity disagree.')
    mzs, intensities = record['mzs'], record['intensities']
    if len(mzs) != len(intensities):
        raise ValueError('Peak m/z and intensity arrays must have equal lengths.')
    if not 1 < float(record['precursor_mz']) < 1500:
        raise ValueError('Precursor m/z must be in (1, 1500).')
    ms1 = record.get('ms1')
    return MetaFeature(str(record['identifier']), float(record['precursor_mz']), adduct.charge,
                       adduct=adduct.string, ms2=Spectrum(mzs, intensities),
                       ms1=Spectrum(ms1['mzs'], ms1['intensities']) if ms1 else None)


def constraint_failures(truth, feature, config):
    checks = (
        ('element_ranges', _element_check(truth, config.ele_lower, config.ele_upper)),
        ('senior', _senior_rules(truth)), ('oxygen_phosphorus', _o_p_check(truth)),
        ('dbe', _dbe_check(truth)),
        ('adduct_loss', _adduct_loss_check(truth, feature.adduct.loss_formula)),
    )
    return [name for name, passes in checks if not passes]


def first_loss(stages, *, in_mass_window, has_ms2):
    present = lambda name: stages.get(name, {}).get('truth_present', False)
    if present('retained'):
        return None
    if not in_mass_window:
        return 'precursor_mass_window'
    if present('valid'):
        return 'candidate_pruning'
    if present('merged'):
        return 'physical_validation'
    if present('ms2_stitched') or present('precursor_database'):
        return 'chemical_filters'
    return 'not_generated' if has_ms2 else 'precursor_database'


def run_record(record, backend, options, dependencies, searches):
    from benchmarks.formula_search import MassDeskSearch

    started = perf_counter()
    config = make_config(record, options)
    feature = make_feature(record)
    trace = CandidateTrace(record['formula'], feature, options.get('export_features', False))
    if np.any(trace.charged_truth < 0):
        raise ValueError('Reference formula is incompatible with the reported adduct.')
    feature.data_preprocess(config.ppm, config.ms1_tol, config.ms2_tol,
                            config.isotope_bin_mztol, config.max_isotope_cnt,
                            config.rel_int_denoise_cutoff, config.top_n_per_50_da)
    preprocessing = perf_counter() - started
    search = None
    if backend != 'database':
        key = (backend, tuple(config.ele_upper), config.ms1_tol, config.ms2_tol)
        if key not in searches:
            searches[key] = MassDeskSearch(config, source_path=options.get('massdesk_path'),
                                           mode=backend, max_results=options.get('max_results', 1000),
                                           peak_budget=options.get('peak_budget', 3))
        search = searches[key]
        search.configure(feature)
    generation_start = perf_counter()
    gen_candidate_formula(feature, config.ppm, config.ms1_tol, config.ms2_tol,
                           config.db_mode, config.ele_lower, config.ele_upper,
                           config.max_isotope_cnt, dependencies, trace=trace,
                           fragment_query=search,
                           propagate_loss_tolerance=options.get('propagate_loss_tolerance', False))
    generation_seconds = perf_counter() - generation_start
    charged_mass = float(np.dot(trace.charged_truth, Formula.mass_arr.astype(float)))
    truth_mz = (charged_mass - feature.adduct.charge * mass_e) / abs(feature.adduct.charge)
    error = feature.mz - truth_mz
    in_window = abs(error) <= config.ms1_tol * feature.mz * 1e-6
    annotation_seconds = ranking_seconds = 0.0
    if options.get('rank', True):
        step = perf_counter()
        feature = _gen_subformula(feature, config)
        annotation_seconds = perf_counter() - step
        trace('annotated', arrays=(candidate.formula.array for candidate in feature.candidate_formula_list))
        step = perf_counter()
        predict_formula_probability([feature], 0, 1, config, dependencies)
        feature.candidate_formula_list.sort(key=lambda candidate: candidate.estimated_prob, reverse=True)
        ranking_seconds = perf_counter() - step
    candidates = feature.candidate_formula_list or []
    truth_rank = next((index + 1 for index, candidate in enumerate(candidates)
                       if np.array_equal(candidate.formula.array, trace.truth)), None)
    result = {
        'identifier': str(record['identifier']), 'backend': backend, 'status': 'ok',
        'formula': str(Formula(read_formula(record['formula']))),
        'source_library': record.get('source_library'), 'instrument': record.get('instrument'),
        'polarity': record.get('polarity'), 'adduct': feature.adduct.string,
        'precursor_mz': feature.mz, 'truth_mass_error_ppm': error / truth_mz * 1e6,
        'truth_in_mass_window': bool(in_window),
        'truth_constraint_failures': constraint_failures(trace.truth, feature, config),
        'raw_peaks': len(feature.ms2_raw) if feature.ms2_raw else 0,
        'processed_peaks': len(feature.ms2_processed) if feature.ms2_processed else 0,
        'has_ms1_isotopes': bool(feature.ms1_processed) and len(feature.ms1_processed) > 1,
        'bottom_up_supported': abs(feature.adduct.charge) == 1,
        'stages': trace.stages, 'fragment_queries': trace.queries,
        'candidate_loss': first_loss(trace.stages, in_mass_window=in_window,
                                     has_ms2=bool(feature.ms2_processed)),
        'truth_rank': truth_rank if options.get('rank', True) else None,
        'candidates': [str(candidate.formula) for candidate in candidates],
        'scores': [float(candidate.estimated_prob) for candidate in candidates]
                  if options.get('rank', True) else None,
        'timing_seconds': {'preprocessing': preprocessing, 'generation': generation_seconds,
                           'subformula_assignment': annotation_seconds, 'ranking': ranking_seconds,
                           'total': perf_counter() - started},
        'search': dict(search.stats) if search else {},
    }
    if search:
        result['search']['cache'] = search.cached_decomposition.cache_info()._asdict()
    if options.get('export_features'):
        result['proposal_candidates'] = trace.proposal_candidates
        result['proposal_features'] = trace.proposal_features
    if options.get('export_features') and candidates and options.get('rank', True):
        features = gen_ml_feature([feature], config.ppm, config.ms1_tol, config.ms2_tol, dependencies)
        if not np.all(np.isfinite(features)):
            raise ValueError('Non-finite model features; refusing to export a training example.')
        result['model_features'] = features.tolist()
    return result


def _worker(connection, options):
    try:
        dependencies = init_db(data_dir=options.get('data_dir'))
        if options.get('rank', True):
            init_ml_models(dependencies, data_dir=options.get('data_dir'))
        searches = {}
        # Compile candidate filters and core matching before timed requests.
        warm = {'identifier': 'warmup', 'formula': 'C6H12O6', 'adduct': '[M+H]+',
                'precursor_mz': 181.070665, 'mzs': [61.0284, 91.03897],
                'intensities': [100, 50], 'instrument': 'orbitrap'}
        run_record(warm, 'database', {**options, 'export_features': False}, dependencies, searches)
        connection.send({'ready': True})
        while True:
            request = connection.recv()
            if request is None:
                break
            record, backend = request
            try:
                result = run_record(record, backend, options, dependencies, searches)
            except (ValueError, TypeError) as exc:
                result = {'identifier': str(record.get('identifier')), 'backend': backend,
                          'status': 'invalid_input', 'error': str(exc)}
            except Exception as exc:
                result = {'identifier': str(record.get('identifier')), 'backend': backend,
                          'status': 'error', 'error': '{}: {}'.format(type(exc).__name__, exc),
                          'traceback': traceback.format_exc()}
            connection.send(result)
    except Exception:
        connection.send({'startup_error': traceback.format_exc()})
    finally:
        connection.close()


class IsolatedRunner:
    def __init__(self, options, timeout=30):
        self.options, self.timeout = options, timeout
        self.process = self.connection = None

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = None

    def _start(self):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, self.options))
        self.process.start()
        child.close()
        if not self.connection.poll(120):
            self.close()
            raise RuntimeError('Worker initialization exceeded 120 seconds.')
        message = self.connection.recv()
        if not message.get('ready'):
            self.close()
            raise RuntimeError(message.get('startup_error', str(message)))

    def run(self, record, backend):
        if self.process is None:
            self._start()
        started = perf_counter()
        self.connection.send((record, backend))
        if not self.connection.poll(self.timeout):
            self.close()
            return {'identifier': str(record['identifier']), 'backend': backend,
                    'status': 'timeout', 'timeout_seconds': self.timeout}
        try:
            result = self.connection.recv()
        except EOFError:
            code = self.process.exitcode
            self.close()
            return {'identifier': str(record['identifier']), 'backend': backend,
                    'status': 'worker_failed', 'exit_code': code}
        result['request_seconds'] = perf_counter() - started
        return result


def summarize(results):
    groups = defaultdict(list)
    for result in results:
        groups[result['backend']].append(result)
    output = {}
    for backend, rows in sorted(groups.items()):
        complete = [row for row in rows if row['status'] == 'ok']
        ranked = [row for row in complete if row.get('scores') is not None]
        stages = sorted({name for row in complete for name in row['stages']})
        recall = {}
        for stage in stages:
            applicable = [row for row in complete if stage in row['stages']]
            recall[stage] = {'correct': sum(row['stages'][stage]['truth_present'] for row in applicable),
                             'denominator': len(applicable)}
        total_seconds = [row['timing_seconds']['total'] for row in complete]
        output[backend] = {
            'attempted': len(rows), 'completed': len(complete),
            'statuses': dict(Counter(row['status'] for row in rows)),
            'stage_recall_completed_only': recall,
            'top1_correct': sum(row['truth_rank'] == 1 for row in ranked),
            'top5_correct': sum(row['truth_rank'] is not None and row['truth_rank'] <= 5 for row in ranked),
            'ranked_denominator': len(ranked),
            'candidate_losses': dict(Counter(row['candidate_loss'] for row in complete
                                              if row['candidate_loss'])),
            'truncated_spectra': sum(row['search'].get('truncated_queries', 0) > 0 for row in complete),
            'median_completed_seconds': float(np.median(total_seconds)) if total_seconds else None,
            'p95_completed_seconds': float(np.percentile(total_seconds, 95)) if total_seconds else None,
            'note': 'Stage recall excludes timeouts/errors; operational accuracy must also use all attempted spectra. Truncated searches are not exhaustive.',
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backends', nargs='+', default=['database'],
                        choices=['database', 'fallback', 'union', 'de-novo'])
    parser.add_argument('--massdesk-path', type=Path)
    parser.add_argument('--max-results', type=int, default=1000)
    parser.add_argument('--peak-budget', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--candidate-only', action='store_true')
    parser.add_argument('--export-features', action='store_true')
    parser.add_argument('--propagate-loss-tolerance', action='store_true',
                        help='experimental loss window = precursor absolute tolerance + fragment absolute tolerance')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--element-ranges', type=Path, help='JSON mapping of element to [minimum, maximum]')
    parser.add_argument('--ms1-tol', type=float)
    parser.add_argument('--ms2-tol', type=float)
    args = parser.parse_args()
    if not isfinite(args.timeout) or args.timeout <= 0 or args.max_results < 1 or args.peak_budget < 1:
        parser.error('Timeout, result limit and peak budget must be positive.')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    if any(value is not None and (not isfinite(value) or value <= 0)
           for value in (args.ms1_tol, args.ms2_tol)):
        parser.error('Mass tolerances must be finite and positive.')
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit:
        records = records[:args.limit]
    identifiers = [str(record['identifier']) for record in records]
    if len(set(identifiers)) != len(identifiers):
        parser.error('Input spectrum identifiers must be unique.')
    options = {'rank': not args.candidate_only, 'export_features': args.export_features,
               'massdesk_path': str(args.massdesk_path) if args.massdesk_path else None,
               'max_results': args.max_results, 'peak_budget': args.peak_budget,
               'data_dir': str(args.data_dir) if args.data_dir else None,
               'ms1_tol': args.ms1_tol, 'ms2_tol': args.ms2_tol}
    options['propagate_loss_tolerance'] = args.propagate_loss_tolerance
    if args.element_ranges:
        options['element_ranges'] = json.loads(args.element_ranges.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    runner = IsolatedRunner(options, args.timeout)
    try:
        with (args.output / 'audit.jsonl').open('w') as stream:
            for index, record in enumerate(records):
                for backend in args.backends:
                    result = runner.run(record, backend)
                    results.append(result)
                    stream.write(json.dumps(result, allow_nan=False) + '\n')
                    stream.flush()
                    print('{}/{} {} {} {}'.format(index + 1, len(records), record['identifier'],
                                                   backend, result['status']), flush=True)
    finally:
        runner.close()
    report = {'input': str(args.input.resolve()), 'settings': options,
              'timeout_seconds': args.timeout, 'python': sys.version,
              'platform': platform.platform(), 'results': summarize(results)}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output / 'summary.tsv').open('w') as stream:
        columns = ['backend', 'attempted', 'completed', 'top1_correct', 'top5_correct',
                   'ranked_denominator', 'truncated_spectra', 'median_completed_seconds',
                   'p95_completed_seconds']
        writer = csv.DictWriter(stream, columns, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        for backend, row in report['results'].items():
            writer.writerow({'backend': backend, **row})
    print(json.dumps(report['results'], indent=2))


if __name__ == '__main__':
    main()
