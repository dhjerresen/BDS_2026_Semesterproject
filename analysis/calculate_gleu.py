#!/usr/bin/env python3
"""Revised GEC-GLEU-style corpus evaluation for DaLA (offline, Python 3.10+).

Unlike nltk.translate.gleu_score, the scorer uses source + hypothesis + reference.
Implements the n-gram precision and source-overlap penalty described in Napoles
et al. (2016) / the 2023 GEC survey, with BLEU-style brevity penalty.
Implementation is self-contained and NOT the authors' official scorer; numerical
results must be called 'standalone revised GEC-GLEU implementation'.

Default: whitespace word tokens, n=1..4, one reference per DaLA example.
Failed/truncated/empty outputs are included as empty hypotheses (intent-to-score),
not silently dropped. No data leaves the local machine.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

# Define the columns needed to calculate GEC-GLEU and identify each prediction.
# Treat unsuccessful or incomplete generations as invalid predictions rather than excluding them.
# Set a consistent color palette for model comparison charts.
REQUIRED = {'model', 'item_id', 'is_control', 'error_type', 'status',
            'input_text', 'reference_text', 'output_text'}
BAD = {'possibly_truncated', 'empty_output', 'error'}
COLORS = ['#1F5A82', '#439B96', '#C79842', '#A570B5', '#C45F5F', '#6E8398']


# Convert the control flag from CSV text into a Boolean so control and error rows are handled correctly.
def clean_bool(s):
    if str(s).lower() in ('true', '1'): return True
    if str(s).lower() in ('false', '0'): return False
    raise ValueError('Unexpected is_control value: ' + repr(s))


# Load the per-prediction CSV and group observations by model.
# Check required fields, unique IDs, consistent test inputs, and matching datasets across models.
# These checks make comparisons fair: every model must be evaluated on the same examples.
def read_data(path):
    # Store each model's predictions as a list of rows.
    results = defaultdict(list)
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError('Missing columns: ' + ', '.join(sorted(REQUIRED - set(reader.fieldnames or []))))
        seen = set()
        shared = {}
        for line, r in enumerate(reader, 2):
            model = r['model'].strip()
            ident = r['item_id'].strip()
            if not model or not ident: raise ValueError(f'Line {line}: empty model/item_id')
            k = (model, ident)
            if k in seen: raise ValueError(f'Line {line}: duplicate model/item_id {k}')
            seen.add(k)
            control = clean_bool(r['is_control'])
            # An error example must differ from its corrected reference.
            if not control and r['input_text'] == r['reference_text']:
                raise ValueError(f'Line {line}: error row identical to reference: {ident}')
            v = (r['input_text'], r['reference_text'], control, r['error_type'])
            if ident in shared and shared[ident] != v:
                raise ValueError(f'Line {line}: inconsistent input/reference/control/error_type for {ident}')
            shared[ident] = v
            if r['status'] not in ('ok', *BAD):
                raise ValueError(f'Line {line}: unexpected status {r["status"]!r}')
            r['is_control'] = control
            results[model].append(r)
    if not results: raise ValueError('No rows found')
    # Compare item-ID sets to ensure no model is evaluated on a different subset.
    sets = {m: {r['item_id'] for r in rows} for m, rows in results.items()}
    m0 = next(iter(sets))
    for m, ids in sets.items():
        if ids != sets[m0]:
            raise ValueError(f'ID set mismatch: {m} vs {m0}; missing={len(sets[m0]-ids)}, extra={len(ids-sets[m0])}')
    if len(shared) != 3328 or sum(not v[2] for v in shared.values()) != 1664:
        print(f'OBS: found {len(shared)} unique inputs; {sum(not v[2] for v in shared.values())} erroneous. Expected 3328/1664 for your full dataset.')
    return results


# Count each sequence of n adjacent words; repeated n-grams are counted repeatedly.
# The scorer evaluates word-level 1-, 2-, 3-, and 4-grams.
def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))


# Collect corpus-wide n-gram matches, candidate n-gram totals, and text lengths.
# The source sentence is needed to penalize incorrect text the model leaves unchanged.
# baseline=True uses the unchanged input; oracle=True uses the correct reference.
def get_counts(rows, max_n=4, baseline=False, oracle=False):
    """Corpus modified n-gram precisions and hypothesis/reference lengths.

    Penalty for retaining source ngrams unsupported by the reference is:
    max(0, min(count_h, count_source) - min(count_h, count_ref)).
    """
    # Keep separate numerator and denominator counts for each n-gram order.
    matches = [0.0] * max_n
    possible = [0.0] * max_n
    hyp_len = ref_len = 0
    for r in rows:
        source = r['input_text'].split()
        ref = r['reference_text'].split()
        # Choose the correct reference, untouched input, or actual model prediction.
        if oracle:
            hyp = ref
        elif baseline:
            hyp = source
        else:
            # Truncated/failed generations are not valid GEC hypotheses.
            hyp = r['output_text'].split() if r['status'] == 'ok' else []
        hyp_len += len(hyp)
        ref_len += len(ref)
        # Compare candidate, original input, and reference at each n-gram length.
        for n in range(1, max_n + 1):
            h, s, rf = ngrams(hyp, n), ngrams(source, n), ngrams(ref, n)
            # Reward n-grams in the prediction that also appear in the reference.
            overlap = sum(min(c, rf.get(k, 0)) for k, c in h.items())
            # Penalize source n-grams retained in the output but unsupported by the reference.
            retained_bad = sum(max(0, min(c, s.get(k, 0)) - min(c, rf.get(k, 0))) for k, c in h.items())
            # Accumulate adjusted matches and all predicted n-grams across the corpus.
            matches[n-1] += max(0, overlap - retained_bad)
            possible[n-1] += sum(h.values())
    return matches, possible, hyp_len, ref_len


# Turn the corpus counts into a single revised GEC-GLEU score on a 0-100 scale.
# Combine n-gram precisions using a geometric mean and penalize overly short outputs.
def corpus_score(rows, max_n=4, baseline=False, oracle=False):
    matches, possible, hyp_len, ref_len = get_counts(rows, max_n, baseline, oracle)
    # Return zero if the model produced no hypothesis text.
    if hyp_len == 0: return 0.0
    if any(p == 0 or m == 0 for m, p in zip(matches, possible)):
        return 0.0
    # BLEU-style brevity penalty: short outputs should not score highly by omitting words.
    bp = min(1.0, math.exp(1.0 - ref_len / hyp_len))
    result = bp * math.exp(sum(math.log(m / p) for m, p in zip(matches, possible)) / max_n)
    return 100 * result


# Export analysis tables as UTF-8 CSV files with headers for reporting and inspection.
def write_csv(path, rows, cols):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


# Shorten selected Hugging Face model IDs to readable names for tables and figures.
def model_name(s):
    name = s.split('/')[-1]
    if 'munin' in s.lower(): return 'Munin Apertus 8B'
    if 'snak' in s.lower(): return 'SnakModel 7B'
    return name


# Generate three visual comparisons: model ranking, GLEU versus Exact Match, and error types.
# Use a non-interactive backend so figures can be saved on a server without a display.
def plots(summary, category, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib ikke installeret: springer figurer over. pip install matplotlib')
        return
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 130, 'savefig.dpi': 220,
                         'axes.spines.top': False, 'axes.spines.right': False})
    # Rank models by their corpus GEC-GLEU score on the erroneous sentences.
    ordered = sorted(summary, key=lambda r: -r['gleu_error_pct'])
    labels = [model_name(r['model']) for r in ordered]
    values = [r['gleu_error_pct'] for r in ordered]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels[::-1], values[::-1], color=COLORS[:len(values)][::-1])
    ax.set_xlabel('GEC-GLEU, korpus-score (0–100)')
    ax.set_title('Dansk grammatisk fejlretning – GEC-GLEU på 1.664 fejlsætninger', weight='bold')
    ax.set_xlim(0, max(100, max(values, default=0) * 1.10))
    for b in bars: ax.text(b.get_width() + 0.6, b.get_y() + b.get_height()/2, f'{b.get_width():.1f}', va='center')
    fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(out / f'01_gec_gleu_modelrangering.{ext}', bbox_inches='tight')
    plt.close(fig)

    # Show whether strict Exact Match and partial n-gram similarity tell the same story.
    fig, ax = plt.subplots(figsize=(8.7, 6))
    for i, row in enumerate(summary):
        x, y = row['em_error_pct'], row['gleu_error_pct']
        ax.scatter(x, y, color=COLORS[i % len(COLORS)], s=80)
        ax.annotate(model_name(row['model']), (x, y), xytext=(6, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Exact Match på fejlsætninger (%)')
    ax.set_ylabel('GEC-GLEU på fejlsætninger (0–100)')
    ax.set_title('Exact Match vs. GEC-GLEU', weight='bold')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(out / f'02_exact_match_vs_gleu.{ext}', bbox_inches='tight')
    plt.close(fig)

    # Build a heatmap comparing the models across individual grammatical error types.
    names = [r['model'] for r in ordered]
    cats = sorted({r['error_type'] for r in category})
    lookup = {(r['model'], r['error_type']): r['gleu_pct'] for r in category}
    fig, ax = plt.subplots(figsize=(12.5, max(7, len(cats) * .43)))
    matrix = [[lookup.get((model, cat), float('nan')) for model in names] for cat in cats]
    im = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(names)), [model_name(m) for m in names], rotation=35, ha='right')
    ax.set_yticks(range(len(cats)), [c.replace('flip_', '').replace('corrupt_', '') for c in cats])
    for i in range(len(cats)):
        for j in range(len(names)):
            x = matrix[i][j]
            if not math.isnan(x): ax.text(j, i, f'{x:.1f}', ha='center', va='center', fontsize=7,
                                         color='white' if x >= 65 else 'black')
    fig.colorbar(im, ax=ax, label='Korpus-GEC-GLEU (0–100)', shrink=.8)
    ax.set_title('GEC-GLEU fordelt på de 16 fejltyper', weight='bold')
    fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(out / f'03_gec_gleu_fejltyper.{ext}', bbox_inches='tight')
    plt.close(fig)


# Coordinate input arguments, evaluation, report generation, and optional visualizations.
def main():
    p = argparse.ArgumentParser(description='Offline revised GEC-GLEU for DaLA; no API/GPU')
    p.add_argument('--input', default='analysis/all_predictions.csv', type=Path)
    p.add_argument('--output-dir', default='analysis/gleu', type=Path)
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    if not args.input.is_file(): p.error('Input file does not exist: ' + str(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Read the prediction-level results, then evaluate each model separately.
    data = read_data(args.input)
    summary, cat_rows = [], []
    model_order = sorted(data)
    for model in model_order:
        rows = data[model]
        # Score grammatical error correction on the error examples, not on control rows.
        errors = sorted((r for r in rows if not r['is_control']), key=lambda x: x['item_id'])
        ctrls = [r for r in rows if r['is_control']]
        # Count failed/incomplete predictions; they remain in the denominator for scoring.
        nonok = sum(r['status'] != 'ok' for r in errors)
        # Exact Match only counts outputs identical to the reference, with successful status.
        em = sum(r['status']=='ok' and r['output_text']==r['reference_text'] for r in errors)
        # GEC-GLEU also awards partial n-gram overlap, with a penalty for unchanged errors.
        gleu = corpus_score(errors)
        # Compare the model with a no-change baseline and an ideal reference-as-output oracle.
        row = {
            'model': model, 'error_n': len(errors), 'em_error_pct': round(100*em/len(errors), 3),
            'gleu_error_pct': round(gleu, 3), 'gleu_source_baseline_pct': round(corpus_score(errors, baseline=True), 3),
            'gleu_reference_oracle_pct': round(corpus_score(errors, oracle=True), 3),
            'non_ok_errors_n': nonok, 'controls_n': len(ctrls),
            'unchanged_controls_pct': round(100 * sum(r['status']=='ok' and r['output_text']==r['input_text'] for r in ctrls) / len(ctrls), 3)
        }
        summary.append(row)
        # Break down performance by error category to identify model strengths and weaknesses.
        groups = defaultdict(list)
        for r in errors: groups[r['error_type']].append(r)
        for typ, group in sorted(groups.items()):
            cat_rows.append({'model': model, 'error_type': typ, 'n': len(group),
                             'exact_match_pct': round(100*sum(r['status']=='ok' and r['output_text']==r['reference_text'] for r in group)/len(group), 3),
                             'gleu_pct': round(corpus_score(group), 3),
                             'non_ok_n': sum(r['status']!='ok' for r in group)})
        print(f'{model_name(model):30s}  EM {row["em_error_pct"]:6.2f}%  GEC-GLEU {gleu:6.2f}  non-OK {nonok}', flush=True)
    # Save model-level scores and per-error-type scores for later analysis.
    write_csv(args.output_dir/'gleu_model_summary.csv', summary, list(summary[0]))
    write_csv(args.output_dir/'gleu_by_error_type.csv', cat_rows, list(cat_rows[0]))
    # Make visualizations unless the user explicitly disabled them.
    if not args.no_plots: plots(summary, cat_rows, args.output_dir)
    # Record the exact metric implementation, scoring decisions, and input checksum.
    # This documents reproducibility and clarifies that the metric is not the official scorer.
    meta = {'method': 'standalone revised GEC-GLEU-style corpus metric (not official scorer)',
            'implementation': 'calculate_gleu.py', 'ngram': '1-4', 'tokenization': 'Python str.split() whitespace',
            'single_reference': True, 'controls_scored': False,
            'failed_or_truncated_outputs': 'included as empty hypotheses, no dropped observations',
            'corpus_score': 'geometric mean ngram precisions with brevity penalty',
            'input_file': str(args.input), 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
            'models': len(model_order), 'rows_per_model': {m: len(data[m]) for m in model_order}}
    (args.output_dir/'gleu_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('\nSaved results to:', args.output_dir.resolve())

# Run the analysis only when this file is executed directly, not when imported.
if __name__ == '__main__': main()
