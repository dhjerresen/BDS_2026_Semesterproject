#!/usr/bin/env python3
"""Build the pooled DaLA zero-shot evaluation files from prepare_dataset.py outputs.

No manual reviews, train/validation/test selection, model calls or prompt development.
The original split is kept ONLY for provenance. Uses Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

DATASET = 'danish-foundation-models/dala'
REVISION = '0cc96091dc265d14fb5d507114690fd85857b6da'
EXPECTED_SPLITS = {'train': 512, 'val': 128, 'test': 1024}
OUTPUT_FIELDS = ('item_id', 'pair_id', 'source_split', 'error_type', 'is_control', 'input_text', 'reference_text')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def file_hash(path: Path) -> str:
    require(path.is_file(), f'Mangler fil: {path}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    require(path.is_file(), f'Mangler CSV: {path}')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and set(expected) <= set(reader.fieldnames),
                f'Manglende kolonner i {path}; forventede {expected}')
        rows = list(reader)
    return rows


def csv_bytes(rows: list[dict[str, str]]) -> bytes:
    target = io.StringIO(newline='')
    writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode('utf-8-sig')


def build(root: Path) -> dict[str, bytes]:
    source = root / 'data' / 'processed'
    pair_path = source / 'dala_gec_pairs.csv'
    control_path = source / 'dala_controls.csv'
    report_path = source / 'dataset_report.json'
    require(report_path.is_file(), f'Mangler datasetrapport: {report_path}')
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    require(report.get('source_dataset') == DATASET and report.get('source_revision') == REVISION
            and report.get('shuffle_seed') == 4242,
            'DaLA-versionen er ikke fastlåst korrekt. Kør: python prepare_dataset.py --revision ' + REVISION)
    require(report.get('number_of_pairs') == 1664 and report.get('number_of_control_inputs') == 1664
            and report.get('pairs_by_split') == EXPECTED_SPLITS and report.get('number_of_categories') == 16,
            'Datasetrapporten har uventede antal eller kategorier')

    pairs = read_csv(pair_path, ('pair_id', 'split', 'error_type', 'incorrect_text', 'reference_text'))
    controls = read_csv(control_path, ('pair_id', 'split', 'error_type', 'is_control', 'input_text', 'reference_text'))
    require(len(pairs) == len(controls) == 1664, 'Forventede 1.664 fejlpar og 1.664 kontroller')
    by_id = {}
    category_counts, split_counts = Counter(), Counter()
    for row in pairs:
        pid, split = row['pair_id'], row['split']
        require(pid and pid not in by_id and split in EXPECTED_SPLITS and pid.startswith(split + '-'),
                f'Duplikat eller ugyldigt pair_id/split: {pid}')
        require(row['error_type'].strip() and row['incorrect_text'].strip() and row['reference_text'].strip()
                and row['incorrect_text'] != row['reference_text'], f'Ugyldigt fejlpar: {pid}')
        by_id[pid] = row
        split_counts[split] += 1
        category_counts[row['error_type']] += 1
    require(dict(split_counts) == EXPECTED_SPLITS, f'Uventede splitantal: {dict(split_counts)}')
    require(len(category_counts) == 16 and dict(category_counts) == report.get('pairs_by_category'),
            'Fejlkategorier matcher ikke datasetrapporten')

    control_by_id = {}
    for row in controls:
        pid = row['pair_id']
        require(pid in by_id and pid not in control_by_id, f'Ugyldigt kontrol-ID: {pid}')
        corresponding = by_id[pid]
        require(row['split'] == corresponding['split'] and row['error_type'] == 'none'
                and row['is_control'].strip().lower() == 'true'
                and row['input_text'] == row['reference_text'] == corresponding['reference_text'],
                f'Kontrollen matcher ikke referenceparret: {pid}')
        control_by_id[pid] = row
    require(set(by_id) == set(control_by_id), 'Manglende kontrolsætninger')

    errors, correct, combined = [], [], []
    for pid in sorted(by_id):
        row = by_id[pid]
        base = {'pair_id': pid, 'source_split': row['split'], 'reference_text': row['reference_text']}
        error = {'item_id': 'err:' + pid, **base, 'error_type': row['error_type'],
                 'is_control': 'false', 'input_text': row['incorrect_text']}
        control = {'item_id': 'ctrl:' + pid, **base, 'error_type': 'none',
                   'is_control': 'true', 'input_text': row['reference_text']}
        errors.append(error)
        correct.append(control)
        combined.extend((error, control))

    files = {'all_errors.csv': csv_bytes(errors), 'all_controls.csv': csv_bytes(correct),
             'evaluation_inputs.csv': csv_bytes(combined)}
    manifest = {
        'protocol_version': 'pooled-simple-v3',
        'source_dataset': DATASET,
        'source_revision': REVISION,
        'reconstruction_shuffle_seed': 4242,
        'design': 'pooled zero-shot model comparison; no fine-tuning or prompt development',
        'source_splits': 'Original train/val/test retained as metadata only, not experimental partitions',
        'counts': {'error_inputs': 1664, 'control_inputs': 1664, 'total_inputs': 3328,
                   'error_categories': 16},
        'error_category_counts': dict(sorted(category_counts.items())),
        'source_split_counts': dict(sorted(split_counts.items())),
        'quality_caveat': 'References are not fully linguistically validated; Exact Match is reference agreement, not necessarily grammatical correctness.',
        'review_policy': 'Manual reference review files not required or generated; optional later qualitative analysis.',
        'prompt_policy': 'Select and document one identical zero-shot prompt before running any models.',
        'input_sha256': {p.name: file_hash(p) for p in (pair_path, control_path, report_path)},
        'output_sha256': {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()},
    }
    files['pooled_manifest.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description='Saml alle DaLA-par uden review- eller split-afhængigheder')
    parser.add_argument('--root', type=Path, default=Path('.'), help='Projektrod (standard: nuværende mappe)')
    parser.add_argument('--force', action='store_true', help='Overskriv ændrede eksisterende pooled-output bevidst')
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        files = build(root)
        destination = root / 'data' / 'pooled'
        # Validate all target conflicts before writing anything.
        for name, data in files.items():
            target = destination / name
            require(args.force or not target.exists() or target.read_bytes() == data,
                    f'Eksisterende output er anderledes: {target}. Brug --force for bevidst overskrivning.')
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (destination / name).write_bytes(data)
        print('OK: Pooled DaLA-evalueringsdatasæt genereret.')
        print('  1.664 fejl + 1.664 kontroller = 3.328 inputs')
        print('  Ingen afhængighed af reviews/, review_data/ eller data/final/')
        print('  Gemte filer: ' + ', '.join(files))
        print('  Placering: ' + str(destination))
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f'FEJL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
