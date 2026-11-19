"""Sample labelled spectra from the NAS CASMI 2026 Parquet without copying it.

Run ``python -m benchmarks.library_sample --help``. Sampling uses metadata
first; only selected row groups' peak arrays are materialized. Originals are
opened read-only. The result is an internal research dataset, not a new library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from msbuddy._acquisition import instrument_family as _instrument_family


def instrument_family(value):
    text = str(value or '').lower().replace('-', '').replace(' ', '')
    family = _instrument_family(value)
    if family in ('tof', 'orbitrap'):
        return family
    if any(token in text for token in ('hcd', 'ftms')):
        return 'orbitrap'
    return 'unknown'


def _priority(seed, *parts):
    value = json.dumps([seed, *parts], ensure_ascii=True, separators=(',', ':'))
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), 'big')


def sample_parquet(path, per_stratum=2, seed=20261113, libraries=None):
    import pyarrow.parquet as pq

    path = Path(path)
    parquet = pq.ParquetFile(path)
    columns = ['ingest_lib', 'instrument_type', 'ionization_mode', 'precursor_mz',
               'molecular_formula', 'inchikey14', 'normalized_smiles', 'adduct']
    selected = defaultdict(dict)
    populations = Counter()
    offset = 0
    for batch in parquet.iter_batches(batch_size=32768, columns=columns):
        for local_index, row in enumerate(batch.to_pylist()):
            library = row['ingest_lib']
            if libraries and library not in libraries:
                continue
            mz = row['precursor_mz']
            if mz is None or not 1 < mz < 1500 or not row['molecular_formula']:
                continue
            family = instrument_family(row['instrument_type'])
            if family == 'unknown':
                continue
            mass_bin = '<400' if mz < 400 else '400-800' if mz <= 800 else '>800'
            stratum = (library, family, row['ionization_mode'], mass_bin)
            populations[stratum] += 1
            group = row['inchikey14'] or row['normalized_smiles'] or row['molecular_formula']
            index = offset + local_index
            priority = _priority(seed, *stratum, group)
            tie_break = _priority(seed, index)
            bucket = selected[stratum]
            previous = bucket.get(group)
            if previous is not None:
                if tie_break < previous[1]:
                    bucket[group] = (priority, tie_break, index)
            elif len(bucket) < per_stratum:
                bucket[group] = (priority, tie_break, index)
            else:
                worst = max(bucket, key=lambda key: bucket[key][0])
                if priority < bucket[worst][0]:
                    del bucket[worst]
                    bucket[group] = (priority, tie_break, index)
        offset += batch.num_rows

    wanted = {}
    for stratum, bucket in selected.items():
        for _, _, index in bucket.values():
            wanted[index] = stratum
    records = []
    group_offset = 0
    for group_index in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group_index).num_rows
        indices = sorted(index for index in wanted if group_offset <= index < group_offset + count)
        if indices:
            table = parquet.read_row_group(group_index)
            rows = table.take([index - group_offset for index in indices]).to_pylist()
            for index, row in zip(indices, rows):
                records.append({
                    'identifier': 'casmi2026:{}'.format(index),
                    'source_path': str(path.resolve()), 'source_row': index,
                    'source_library': row['ingest_lib'],
                    'instrument': instrument_family(row['instrument_type']),
                    'instrument_original': row['instrument_type'],
                    'polarity': row['ionization_mode'],
                    'formula': row['molecular_formula'],
                    'structure_group': row['inchikey14'],
                    'smiles': row['normalized_smiles'],
                    'adduct': row['adduct'], 'precursor_mz': row['precursor_mz'],
                    'precursor_error_ppm_reported': row['precursor_error_ppm'],
                    'collision_energy_ev': row['collision_energy_ev'],
                    'mzs': row['ms2_mzs'], 'intensities': row['ms2_normalized_intensities'],
                    'sampling_stratum': list(wanted[index]),
                })
        group_offset += count
    report = {
        'source_path': str(path.resolve()), 'source_bytes': path.stat().st_size,
        'source_rows': parquet.metadata.num_rows,
        'seed': seed, 'per_stratum': per_stratum,
        'selected_spectra': len(records),
        'selection': 'lowest deterministic hash per structure within source/instrument/polarity/precursor-mz stratum',
        'excluded': 'unknown instruments, missing formulas, precursor m/z outside (1, 1500)',
        'strata': [{'stratum': list(key), 'eligible_spectra': count,
                    'selected_structures': len(selected[key])}
                   for key, count in sorted(populations.items())],
    }
    return records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-stratum', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--libraries', nargs='+')
    args = parser.parse_args()
    if args.per_stratum < 1:
        parser.error('--per-stratum must be positive')
    records, report = sample_parquet(args.parquet, args.per_stratum, args.seed,
                                     set(args.libraries) if args.libraries else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    args.output.with_suffix('.sampling.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'selected_spectra': len(records), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
