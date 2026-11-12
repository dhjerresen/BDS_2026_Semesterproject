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

# Pin the dataset identity and exact source revision so every run uses the same underlying data.
# Expected split counts are checked but the split labels will not divide the evaluation.
DATASET = 'danish-foundation-models/dala'
REVISION = '0cc96091dc265d14fb5d507114690fd85857b6da'
EXPECTED_SPLITS = {'train': 512, 'val': 128, 'test': 1024}
OUTPUT_FIELDS = ('item_id', 'pair_id', 'source_split', 'error_type', 'is_control', 'input_text', 'reference_text')


# Centralize validation: stop immediately when an assumption about the input data is false.
def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


# Compute a SHA-256 fingerprint of the file contents to make input versions auditable.
def file_hash(path: Path) -> str:
    require(path.is_file(), f'Mangler fil: {path}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


# Load a CSV after confirming it exists and contains the columns required downstream.
def read_csv(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    require(path.is_file(), f'Mangler CSV: {path}')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and set(expected) <= set(reader.fieldnames),
                f'Manglende kolonner i {path}; forventede {expected}')
        rows = list(reader)
    return rows


# Serialize rows with a fixed column order and newline convention for reproducible outputs.
# UTF-8 with a BOM also helps spreadsheet software recognize Danish characters.
def csv_bytes(rows: list[dict[str, str]]) -> bytes:
    target = io.StringIO(newline='')
    writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode('utf-8-sig')


# Assemble and validate the complete pooled dataset in memory before any files are written.
def build(root: Path) -> dict[str, bytes]:
    # Read the reconstructed error pairs, their matching controls, and the dataset report.
    source = root / 'data' / 'processed'
    pair_path = source / 'dala_gec_pairs.csv'
    control_path = source / 'dala_controls.csv'
    report_path = source / 'dataset_report.json'
    require(report_path.is_file(), f'Mangler datasetrapport: {report_path}')
    # Check the dataset ID, pinned revision and reconstruction seed against the expected protocol.
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    require(report.get('source_dataset') == DATASET and report.get('source_revision') == REVISION
            and report.get('shuffle_seed') == 4242,
            'DaLA-versionen er ikke fastlåst korrekt. Kør: python prepare_dataset.py --revision ' + REVISION)
    # Confirm the expected 1,664 sentence pairs, 1,664 controls and 16 error categories.
    require(report.get('number_of_pairs') == 1664 and report.get('number_of_control_inputs') == 1664
            and report.get('pairs_by_split') == EXPECTED_SPLITS and report.get('number_of_categories') == 16,
            'Datasetrapporten har uventede antal eller kategorier')

    # Read the two source tables; neither annotations nor manual review outputs are needed.
    pairs = read_csv(pair_path, ('pair_id', 'split', 'error_type', 'incorrect_text', 'reference_text'))
    controls = read_csv(control_path, ('pair_id', 'split', 'error_type', 'is_control', 'input_text', 'reference_text'))
    require(len(pairs) == len(controls) == 1664, 'Forventede 1.664 fejlpar og 1.664 kontroller')
    # Index erroneous pairs by unique ID and count their split and error-type distributions.
    by_id = {}
    category_counts, split_counts = Counter(), Counter()
    # Reject missing/duplicate IDs, invalid splits, empty text and unchanged error sentences.
    for row in pairs:
        pid, split = row['pair_id'], row['split']
        require(pid and pid not in by_id and split in EXPECTED_SPLITS and pid.startswith(split + '-'),
                f'Duplikat eller ugyldigt pair_id/split: {pid}')
        require(row['error_type'].strip() and row['incorrect_text'].strip() and row['reference_text'].strip()
                and row['incorrect_text'] != row['reference_text'], f'Ugyldigt fejlpar: {pid}')
        by_id[pid] = row
        split_counts[split] += 1
        category_counts[row['error_type']] += 1
    # Compare observed counts with the pinned dataset report to detect accidental data changes.
    require(dict(split_counts) == EXPECTED_SPLITS, f'Uventede splitantal: {dict(split_counts)}')
    require(len(category_counts) == 16 and dict(category_counts) == report.get('pairs_by_category'),
            'Fejlkategorier matcher ikke datasetrapporten')

    # Verify that every error pair has one correct control sentence with the same reference.
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

    # Construct separate error and control records plus one interleaved evaluation dataset.
    errors, correct, combined = [], [], []
    # Stable sorting makes output ordering independent of the source CSV row order.
    for pid in sorted(by_id):
        row = by_id[pid]
        base = {'pair_id': pid, 'source_split': row['split'], 'reference_text': row['reference_text']}
        # Prefix IDs with err:/ctrl: to identify the observation type without losing pair identity.
        error = {'item_id': 'err:' + pid, **base, 'error_type': row['error_type'],
                 'is_control': 'false', 'input_text': row['incorrect_text']}
        control = {'item_id': 'ctrl:' + pid, **base, 'error_type': 'none',
                   'is_control': 'true', 'input_text': row['reference_text']}
        # Keep separate lists for analysis and combine both conditions for model inference.
        errors.append(error)
        correct.append(control)
        combined.extend((error, control))

    # Prepare three CSV outputs entirely in memory before writing anything to disk.
    files = {'all_errors.csv': csv_bytes(errors), 'all_controls.csv': csv_bytes(correct),
             'evaluation_inputs.csv': csv_bytes(combined)}
    # Document dataset provenance, experimental design, exact observation counts and caveats.
    # Hashes of input and output files allow later checks that the data has not changed.
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
    # Add the JSON manifest as a fourth output; return all output bytes together.
    files['pooled_manifest.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    return files


# Command-line entry point: set the project folder and control overwrite behavior.
def main() -> int:
    parser = argparse.ArgumentParser(description='Saml alle DaLA-par uden review- eller split-afhængigheder')
    parser.add_argument('--root', type=Path, default=Path('.'), help='Projektrod (standard: nuværende mappe)')
    parser.add_argument('--force', action='store_true', help='Overskriv ændrede eksisterende pooled-output bevidst')
    args = parser.parse_args()
    try:
        # Resolve the project root and build all output content before touching existing files.
        root = args.root.resolve()
        files = build(root)
        destination = root / 'data' / 'pooled'
        # Validate all target conflicts before writing anything.
        # Refuse to overwrite changed outputs unless the user explicitly passes --force.
        for name, data in files.items():
            target = destination / name
            require(args.force or not target.exists() or target.read_bytes() == data,
                    f'Eksisterende output er anderledes: {target}. Brug --force for bevidst overskrivning.')
        # Once all conflicts are cleared, create the output directory and write every file.
        destination.mkdir(parents=True, exist_ok=True)
        # Refuse to overwrite changed outputs unless the user explicitly passes --force.
        for name, data in files.items():
            (destination / name).write_bytes(data)
        # Report the data totals and the destination of the generated evaluation files.
        print('OK: Pooled DaLA-evalueringsdatasæt genereret.')
        print('  1.664 fejl + 1.664 kontroller = 3.328 inputs')
        print('  Ingen afhængighed af reviews/, review_data/ eller data/final/')
        print('  Gemte filer: ' + ', '.join(files))
        print('  Placering: ' + str(destination))
        return 0
    # Display a clear error and return a nonzero exit code if validation or file access fails.
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f'FEJL: {exc}', file=sys.stderr)
        return 1


# Execute the script only when run directly, not when imported from another module.
if __name__ == '__main__':
    raise SystemExit(main())
