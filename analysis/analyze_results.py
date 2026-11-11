#!/usr/bin/env python3
"""Offline DaLA GEC analysis. Python 3.10+; standard library only.

Reads inference JSONL files and (optionally) the canonical evaluation_inputs.csv.
Never sends any data to external services. All 3328 observations remain in denominators.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = 3328
ERRORS = 1664
CONTROLS = 1664
ACCEPTED_STATUSES = {'ok', 'possibly_truncated', 'empty_output', 'error'}
DATA_FIELDS = ('item_id', 'pair_id', 'source_split', 'error_type', 'is_control', 'input_text', 'reference_text')
CONFIG_FIELDS = ('model', 'model_revision', 'prompt_sha256', 'dataset_sha256', 'quantization', 'max_new_tokens',
                 'max_input_tokens', 'attention', 'warmup', 'decode_strategy', 'weight_precision')


def die(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def as_bool(value, context: str) -> bool:
    if value is True or (isinstance(value, str) and value.strip().lower() == 'true'):
        return True
    if value is False or (isinstance(value, str) and value.strip().lower() == 'false'):
        return False
    die(f'{context}: is_control skal være true/false, fik {value!r}')


def load_canonical(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(DATA_FIELDS).issubset(set(reader.fieldnames or ())):
            die(f'{path}: manglende felter {set(DATA_FIELDS) - set(reader.fieldnames or ())}')
        rows = list(reader)
    if len(rows) != EXPECTED:
        die(f'{path}: forventede {EXPECTED} rækker, fik {len(rows)}')
    keyed = {}
    for row in rows:
        ident = row['item_id']
        if ident in keyed:
            die(f'{path}: dubleret ID {ident}')
        keyed[ident] = row
    return keyed


def load_jsonl(path: Path):
    keyed = {}
    for lineno, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            die(f'{path}:{lineno}: ugyldig JSON: {exc}')
        if not isinstance(r, dict):
            die(f'{path}:{lineno}: forventede et JSON-objekt')
        missing = set(DATA_FIELDS + ('status', 'output_text', 'latency_seconds')) - set(r)
        if missing:
            die(f'{path}:{lineno}: manglende felter {sorted(missing)}')
        ident = r['item_id']
        if not isinstance(ident, str) or not ident:
            die(f'{path}:{lineno}: ugyldigt item_id')
        if ident in keyed:
            die(f'{path}:{lineno}: dubleret item_id {ident}')
        if r['status'] not in ACCEPTED_STATUSES:
            die(f'{path}:{lineno}: ukendt status {r["status"]!r}')
        if not isinstance(r['output_text'], str):
            die(f'{path}:{lineno}: output_text er ikke en streng')
        as_bool(r['is_control'], f'{path}:{lineno}')
        try:
            latency = float(r['latency_seconds'])
        except (ValueError, TypeError):
            die(f'{path}:{lineno}: ugyldig latency_seconds')
        if not math.isfinite(latency) or latency < 0:
            die(f'{path}:{lineno}: latency_seconds skal være endelig og >=0')
        keyed[ident] = r
    if len(keyed) != EXPECTED:
        die(f'{path}: forventede {EXPECTED} observationer; fik {len(keyed)}. Ingen partiel analyse udført.')
    return keyed


def validate_pairs(rows, label: str):
    err, ctrl = {}, {}
    for ident, r in rows.items():
        control = as_bool(r['is_control'], f'{label} {ident}')
        target = ctrl if control else err
        pid = r['pair_id']
        if pid in target:
            die(f'{label}: dubleret pair_id i samme gruppe: {pid}')
        if not ident.startswith('ctrl:' if control else 'err:'):
            die(f'{label}: ID og is_control modsiger hinanden for {ident}')
        if ident.split(':', 1)[1] != pid:
            die(f'{label}: item_id/pair_id mismatch for {ident}')
        target[pid] = r
    if len(err) != ERRORS or len(ctrl) != CONTROLS or set(err) != set(ctrl):
        die(f'{label}: forventede {ERRORS} parrede fejl og kontroller, fandt {len(err)} / {len(ctrl)}')
    for pid, r in ctrl.items():
        if r['input_text'] != r['reference_text'] or r['error_type'] != 'none':
            die(f'{label}: invalid kontrol ved {pid}')
        if r['reference_text'] != err[pid]['reference_text']:
            die(f'{label}: kontrol/reference mismatch ved {pid}')
    return err, ctrl


def identical_input_metadata(a, b):
    for key in DATA_FIELDS:
        av, bv = a.get(key), b.get(key)
        if key == 'is_control':
            if as_bool(av, key) != as_bool(bv, key):
                return key
        elif av != bv:
            return key
    return None


def load_metadata(path: Path):
    metapath = path.with_suffix(path.suffix + '.metadata.json')
    if not metapath.exists():
        return None
    try:
        meta = json.loads(metapath.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as e:
        die(f'{metapath}: ugyldig metadata: {e}')
    return meta


def mean(values):
    return statistics.mean(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def number(n, decimals=6):
    return '' if n is None else round(n, decimals)


def pct(n, d):
    return number(100 * n / d, 4) if d else ''


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description='Offline analyse af DaLA GEC JSONL-resultater (ingen eksterne pakker)')
    ap.add_argument('--outputs', type=Path, default=Path('outputs'))
    ap.add_argument('--analysis-dir', type=Path, default=Path('analysis'))
    ap.add_argument('--inputs', type=Path, default=None, help='Valgfri canonical data/pooled/evaluation_inputs.csv')
    args = ap.parse_args()
    files = sorted(p for p in args.outputs.glob('*.jsonl') if 'smoke' not in p.stem.lower())
    if not files:
        die(f'Ingen hovedkørsler fundet i {args.outputs} (smoke-tests ignoreres)')
    canonical = load_canonical(args.inputs) if args.inputs else None
    canonical_hash = sha256(args.inputs) if args.inputs else None
    if canonical:
        validate_pairs(canonical, str(args.inputs))
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    category_rows = []
    observation_rows = []
    config_rows = []
    warnings = []
    base_input_map = canonical
    base_prompt = None
    base_dataset = None
    model_names = set()
    for path in files:
        data = load_jsonl(path)
        errs, ctrls = validate_pairs(data, str(path))
        if base_input_map is None:
            base_input_map = data
        extra = set(data) - set(base_input_map)
        missing = set(base_input_map) - set(data)
        if extra or missing:
            die(f'{path}: input-ID-sæt afviger: {len(missing)} mangler, {len(extra)} ekstra')
        for ident, row in data.items():
            mismatch = identical_input_metadata(row, base_input_map[ident])
            if mismatch:
                die(f'{path}: inputmetadata afviger ved {ident}, felt {mismatch}')
        model = next(iter(data.values())).get('model')
        if not isinstance(model, str) or not model.strip() or model in model_names:
            die(f'{path}: manglende model eller dubleret modelnavn: {model!r}')
        model_names.add(model)
        for field in CONFIG_FIELDS:
            values = {json.dumps(r.get(field), sort_keys=True) for r in data.values()}
            if len(values) > 1:
                die(f'{path}: konfiguration varierer inden for filen: {field}')
        first = next(iter(data.values()))
        if base_prompt is None:
            base_prompt = first.get('prompt_sha256')
            base_dataset = first.get('dataset_sha256')
        if first.get('prompt_sha256') != base_prompt:
            die(f'{path}: prompt_sha256 afviger mellem modeller. Må ikke sammenlignes som samme prompt.')
        if first.get('dataset_sha256') != base_dataset:
            die(f'{path}: dataset_sha256 afviger mellem modeller. Må ikke sammenlignes som samme datasæt.')
        if canonical_hash and first.get('dataset_sha256') != canonical_hash:
            die(f'{path}: dataset_sha256 matcher ikke --inputs filens SHA256')
        meta = load_metadata(path)
        if meta:
            for field in CONFIG_FIELDS:
                if meta.get(field) != first.get(field):
                    die(f'{path}: metadatafilens {field} afviger fra JSONL')
        else:
            warnings.append(f'{path.name}: .metadata.json blev ikke fundet; SHA/version kan ikke verificeres mod kørselmetadata')
        statuses = Counter(r['status'] for r in data.values())
        exact_error = sum(r['status'] == 'ok' and r['output_text'] == r['reference_text'] for r in errs.values())
        unchanged_control = sum(r['status'] == 'ok' and r['output_text'] == r['input_text'] for r in ctrls.values())
        changed_control_ok = sum(r['status'] == 'ok' and r['output_text'] != r['input_text'] for r in ctrls.values())
        successful = [r for r in data.values() if r['status'] == 'ok']
        latencies = [float(r['latency_seconds']) for r in data.values()]
        success_latencies = [float(r['latency_seconds']) for r in successful]
        generation_tokens = [int(r['generated_tokens']) for r in data.values() if r.get('generated_tokens') is not None]
        summary = {
            'model': model, 'filename': path.name, 'total': EXPECTED, 'error_inputs': ERRORS, 'control_inputs': CONTROLS,
            'ok': statuses['ok'], 'possibly_truncated': statuses['possibly_truncated'],
            'empty_output': statuses['empty_output'], 'error': statuses['error'],
            'non_ok_rate_pct': pct(EXPECTED-statuses['ok'], EXPECTED),
            'exact_match_errors_n': exact_error, 'exact_match_errors_pct': pct(exact_error, ERRORS),
            'unchanged_controls_n': unchanged_control, 'unchanged_controls_pct': pct(unchanged_control, CONTROLS),
            'modified_controls_all_n': CONTROLS-unchanged_control,
            'modified_controls_all_pct': pct(CONTROLS-unchanged_control, CONTROLS),
            'changed_controls_ok_only_n': changed_control_ok,
            'latency_mean_all_s': number(mean(latencies)), 'latency_median_all_s': number(median(latencies)),
            'latency_mean_ok_s': number(mean(success_latencies)), 'latency_median_ok_s': number(median(success_latencies)),
            'generated_tokens_mean': number(mean(generation_tokens)),
            'model_revision_requested': first.get('model_revision') or '',
            'resolved_model_sha': (meta or {}).get('resolved_model_sha', ''),
            'transformers_version': (meta or {}).get('transformers', ''),
            'gpu': (meta or {}).get('gpu', ''), 'precision': first.get('weight_precision') or '',
            'attention': first.get('attention') or '', 'prompt_sha256': first.get('prompt_sha256') or '',
            'dataset_sha256': first.get('dataset_sha256') or '',
        }
        summaries.append(summary)
        categories = defaultdict(list)
        for r in errs.values():
            categories[r['error_type']].append(r)
        for category in sorted(categories):
            group = categories[category]
            em = sum(r['status'] == 'ok' and r['output_text'] == r['reference_text'] for r in group)
            st = Counter(r['status'] for r in group)
            category_rows.append({
                'model': model, 'error_type': category, 'n': len(group), 'exact_match_n': em,
                'exact_match_pct': pct(em, len(group)), 'ok_n': st['ok'],
                'possibly_truncated_n': st['possibly_truncated'], 'empty_output_n': st['empty_output'],
                'error_n': st['error'],
            })
        for ident in sorted(data):
            r = data[ident]
            is_control = as_bool(r['is_control'], ident)
            is_ok = r['status'] == 'ok'
            observation_rows.append({
                'model': model, 'item_id': ident, 'pair_id': r['pair_id'],
                'source_split': r['source_split'], 'error_type': r['error_type'],
                'is_control': is_control, 'status': r['status'],
                'input_text': r['input_text'], 'reference_text': r['reference_text'],
                'output_text': r['output_text'],
                'exact_match': int(is_ok and not is_control and r['output_text'] == r['reference_text']),
                'unchanged_control': int(is_ok and is_control and r['output_text'] == r['input_text']),
                'latency_seconds': r['latency_seconds'], 'generated_tokens': r.get('generated_tokens', ''),
            })
        config_rows.append({
            'model': model, 'output_file': path.name,
            'model_revision_requested': first.get('model_revision', ''),
            'resolved_model_sha': (meta or {}).get('resolved_model_sha', ''),
            'transformers_version': (meta or {}).get('transformers', ''),
            'torch_version': (meta or {}).get('torch', ''), 'gpu': (meta or {}).get('gpu', ''),
            'prompt_sha256': first.get('prompt_sha256', ''), 'dataset_sha256': first.get('dataset_sha256', ''),
            'quantization': first.get('quantization', ''), 'weight_precision': first.get('weight_precision', ''),
            'attention': first.get('attention', ''), 'warmup': first.get('warmup', ''),
            'decode_strategy': first.get('decode_strategy', ''),
            'max_new_tokens': first.get('max_new_tokens', ''),
            'max_input_tokens': first.get('max_input_tokens', ''),
        })
    if len({r['weight_precision'] for r in config_rows}) > 1 or len({r['quantization'] for r in config_rows}) > 1:
        warnings.append('Modellerne bruger forskellig præcision/kvantisering; latency og kvalitet er ikke fuldt sammenlignelige.')
    if len({r['attention'] for r in config_rows}) > 1:
        warnings.append('Modellerne bruger forskellig attention-implementering; latency kan påvirkes.')
    if len({r['transformers_version'] for r in config_rows}) > 1:
        warnings.append('Modellerne bruger forskellige Transformers-versioner; dokumentér forskellen.')
    if len({r['max_new_tokens'] for r in config_rows}) > 1:
        warnings.append('Modellerne har forskellige max_new_tokens; non_ok-rate er ikke direkte sammenlignelig.')
    if len({r['max_input_tokens'] for r in config_rows}) > 1:
        warnings.append('Modellerne har forskellige max_input_tokens.')
    if len({r['warmup'] for r in config_rows}) > 1:
        warnings.append('Modellerne har forskellig warmup; latency er ikke helt sammenlignelig.')
    summaries.sort(key=lambda r: (-r['exact_match_errors_pct'], r['model']))
    write_csv(args.analysis_dir / 'model_summary.csv', summaries, list(summaries[0]))
    write_csv(args.analysis_dir / 'error_type_accuracy.csv', category_rows, list(category_rows[0]))
    write_csv(args.analysis_dir / 'all_predictions.csv', observation_rows, list(observation_rows[0]))
    write_csv(args.analysis_dir / 'run_configurations.csv', config_rows, list(config_rows[0]))
    report = {
        'models': len(files), 'observations_per_model': EXPECTED, 'total_observations': EXPECTED*len(files),
        'canonical_inputs_verified': bool(canonical), 'canonical_file': str(args.inputs) if args.inputs else None,
        'dataset_sha256': base_dataset, 'prompt_sha256': base_prompt, 'warnings': warnings,
        'metric_definitions': {
            'exact_match_errors_pct': 'ok output identical to reference, /1664 erroneous inputs; non-ok counts as not exact',
            'unchanged_controls_pct': 'ok output identical to input, /1664 controls; non-ok does NOT count as unchanged',
            'modified_controls_all_pct': 'all controls minus successfully unchanged controls, /1664; includes non-ok. Not a pure overcorrection rate.',
            'non_ok_rate_pct': 'possibly_truncated + empty_output + error, /3328',
            'latency_all': 'per-item reported elapsed seconds including non-ok; does not include model load, warmup',
            'latency_ok': 'per-item reported elapsed seconds, status ok only',
        },
    }
    (args.analysis_dir / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Valideret {len(files)} modeller x {EXPECTED} inputs = {len(observation_rows)} observationer.')
    print('Kontrolleret: ID-sæt, parring, identiske inputs/referencer, prompt- og dataset-hash, konfigurationer.')
    print(f'\n{"Model":<37} {"EM fejl":>10} {"Uændr. ctrl":>12} {"Non-OK":>9} {"Median s":>10}')
    for r in summaries:
        print(f'{r["model"][:37]:<37} {r["exact_match_errors_pct"]:>9.2f}% {r["unchanged_controls_pct"]:>11.2f}% {r["non_ok_rate_pct"]:>8.2f}% {r["latency_median_all_s"]:>10.2f}')
    for warning in warnings:
        print('ADVARSEL:', warning)
    print('\nFiler gemt i:', args.analysis_dir)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        print('FEJL:', exc, file=sys.stderr)
        raise SystemExit(1)
