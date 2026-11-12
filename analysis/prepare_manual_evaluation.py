#!/usr/bin/env python3
"""Opret en blindet, målrettet manuel DaLA-vurdering (Munin vs. SnakModel).

Kør på Mac: python3 prepare_manual_evaluation.py --input analysis/all_predictions.csv --output-dir manual_evaluation
Afhængighed: openpyxl (Excel-eksport); øvrige moduler er fra standardbiblioteket.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

MUNIN = 'danish-foundation-models/munin-apertus-8b'
SNAK = 'NLPnorth/snakmodel-7b-instruct'
GROUPS = [
    ('munin_only_em', 6),
    ('snak_only_em', 6),
    ('neither_em', 6),
    ('modified_control', 6),
    ('random_remainder', 6),
]


def read_data(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'model', 'item_id', 'is_control', 'status', 'input_text', 'reference_text', 'output_text', 'error_type'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Manglende CSV-felter: {sorted(required - set(reader.fieldnames or []))}')
        models = {MUNIN: {}, SNAK: {}}
        for r in reader:
            model = r['model']
            if model not in models:
                continue
            identifier = r['item_id']
            if identifier in models[model]:
                raise ValueError(f'Dubleret item_id for {model}: {identifier}')
            models[model][identifier] = r
    a, b = models[MUNIN], models[SNAK]
    if not a or not b:
        raise ValueError('Fandt ikke både Munin og SnakModel i all_predictions.csv')
    if set(a) != set(b):
        raise ValueError('Munin og SnakModel har ikke samme item_id-sæt')
    for ident in a:
        for k in ('input_text', 'reference_text', 'is_control', 'error_type'):
            if a[ident][k] != b[ident][k]:
                raise ValueError(f'Inkonsistente inputdata ved {ident}: {k}')
    return a, b


def control(r):
    v = str(r['is_control']).strip().lower()
    if v not in ('true', 'false'):
        raise ValueError(f'Ugyldig is_control: {v}')
    return v == 'true'


def em(r):
    return r['status'] == 'ok' and r['output_text'] == r['reference_text']


def choose_sample(a, b, rng):
    identifiers = sorted(a)
    pools = {
        'munin_only_em': [i for i in identifiers if not control(a[i]) and em(a[i]) and not em(b[i]) and b[i]['status']=='ok'],
        'snak_only_em': [i for i in identifiers if not control(a[i]) and em(b[i]) and not em(a[i]) and a[i]['status']=='ok'],
        'neither_em': [i for i in identifiers if not control(a[i]) and not em(a[i]) and not em(b[i]) and a[i]['status']=='ok' and b[i]['status']=='ok'],
        'modified_control': [i for i in identifiers if control(a[i]) and a[i]['status']=='ok' and b[i]['status']=='ok' and (a[i]['output_text']!=a[i]['input_text'] or b[i]['output_text']!=b[i]['input_text'])],
        'random_remainder': [i for i in identifiers if a[i]['status']=='ok' and b[i]['status']=='ok'],
    }
    selected = []
    used = set()
    for group, n in GROUPS:
        eligible = [i for i in pools[group] if i not in used]
        if group == 'random_remainder':
            # Tilfældigt restudvalg med spredning mellem fejltyper og kontroller.
            buckets = {}
            for i in eligible:
                category = 'control' if control(a[i]) else a[i]['error_type']
                buckets.setdefault(category, []).append(i)
            for values in buckets.values():
                rng.shuffle(values)
            sampled = []
            cats = list(buckets)
            rng.shuffle(cats)
            while len(sampled) < n and any(buckets.values()):
                for cat in cats:
                    if buckets[cat] and len(sampled) < n:
                        sampled.append(buckets[cat].pop())
            pick = sampled
        else:
            if len(eligible) < n:
                raise ValueError(f'Kun {len(eligible)} kandidater til {group}; kræver {n}')
            pick = rng.sample(eligible, n)
        if len(pick) < n:
            raise ValueError(f'Kun {len(pick)} tilgængelige til {group}; kræver {n}')
        used.update(pick)
        selected.extend((group, i) for i in pick)
    rng.shuffle(selected)  # Skjuler udvalgskategori og rækkefølge.
    return selected, {key: len(value) for key,value in pools.items()}


def write_csv(path, rows, cols):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def build_book(path, title, info, headers, rows, response_cols):
    wb = Workbook()
    sh = wb.active
    sh.title = 'Vurdering'
    n = len(headers)
    sh.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n)
    sh.cell(1, 1, title)
    sh.merge_cells(start_row=2, start_column=1, end_row=2, end_column=n)
    sh.cell(2, 1, info)
    for rownum, bg, fg, height in [(1, '16324F', 'FFFFFF', 32),
                                    (2, 'E9F1F5', '25425A', 34)]:
        c = sh.cell(rownum, 1)
        c.fill = PatternFill('solid', fgColor=bg)
        c.font = Font(name='Aptos', bold=(rownum == 1), color=fg,
                      size=15 if rownum == 1 else 11)
        c.alignment = Alignment(vertical='center', wrap_text=True)
        sh.row_dimensions[rownum].height = height
    sh.row_dimensions[4].height = 42
    for col, header in enumerate(headers, 1):
        cell = sh.cell(4, col, header)
        cell.fill = PatternFill('solid', fgColor='20677F')
        cell.font = Font(name='Aptos', bold=True, color='FFFFFF')
        cell.alignment = Alignment(vertical='center', wrap_text=True)
        if col == 1:
            width = 17
        elif header in ('Inputtekst', 'Output A', 'Output B', 'DaLA-reference'):
            width = 56
        elif header == 'Kommentar':
            width = 43
        else:
            width = 24
        sh.column_dimensions[get_column_letter(col)].width = width
    for rownum, row in enumerate(rows, 5):
        sh.row_dimensions[rownum].height = 105
        for col, value in enumerate(row, 1):
            cell = sh.cell(rownum, col, value)
            cell.alignment = Alignment(vertical='top', wrap_text=True)
            cell.font = Font(name='Aptos', size=10)
            if (col - 1) in response_cols:
                cell.fill = PatternFill('solid', fgColor='FFF3CE')
            elif rownum % 2 == 0:
                cell.fill = PatternFill('solid', fgColor='F2F7FA')
    sh.freeze_panes = 'B5'
    sh.auto_filter.ref = f'A4:{get_column_letter(n)}{4+len(rows)}'

    options = {
        'sproglig korrekthed': 'Ja,Delvist,Nej,Usikkert',
        'betydning bevaret': 'Ja,Delvist,Nej,Usikkert',
        'unødvendige ændringer': 'Nej,Valgfrie,Skadelige,Usikkert',
        'format OK': 'Ja,Nej',
        'tilsigtet fejl rettet': 'Ja,Nej,Usikkert,Ikke relevant',
        'Referenceproblemer?': 'Ja,Nej,Usikkert',
    }
    for col, header in enumerate(headers, 1):
        candidates = [values for key, values in options.items() if key in header]
        if not candidates or (col - 1) not in response_cols:
            continue
        dv = DataValidation(type='list', formula1=f'"{candidates[0]}"', allow_blank=True)
        dv.error = 'Vælg en værdi fra listen.'
        dv.errorTitle = 'Ugyldig vurdering'
        dv.showErrorMessage = True
        sh.add_data_validation(dv)
        dv.add(f'{get_column_letter(col)}5:{get_column_letter(col)}{4+len(rows)}')
    wb.save(path)


def main():
    parser = argparse.ArgumentParser(description='Blind manuel GEC-evaluering af Munin og SnakModel')
    parser.add_argument('--input', type=Path, default=Path('analysis/all_predictions.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('manual_evaluation'))
    parser.add_argument('--seed', type=int, default=4242)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'Kan ikke finde {args.input}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    a, b = read_data(args.input)
    selection, pool_sizes = choose_sample(a,b,rng)
    phase1, phase2, secret = [], [], []
    for index,(group,ident) in enumerate(selection,1):
        trial = f'OBS-{index:03d}'
        left, right = (a[ident], b[ident]) if rng.randrange(2)==0 else (b[ident], a[ident])
        # Ingen reference, modelnavne, Exact Match-score eller udvælgelseskategori i fase 1.
        phase1.append([trial, left['input_text'], left['output_text'], right['output_text'], '', '', '', '', '', '', '', '', ''])
        phase2.append([trial, left['input_text'], left['reference_text'], left['output_text'], right['output_text'], '', '', '', ''])
        secret.append({
            'study_id':trial,'item_id':ident,'pair_id':left.get('pair_id',''),
            'selection_group':group,'error_type':left['error_type'], 'is_control':left['is_control'],
            'model_A':left['model'],'model_B':right['model'],
            'status_A':left['status'],'status_B':right['status'],
            'exact_match_A':int(em(left) and not control(left)),
            'exact_match_B':int(em(right) and not control(right)),
        })
    build_book(args.output_dir/'01_blind_vurdering.xlsx',
        'Fase 1 · Blind vurdering (uden reference)',
        'Udfyld gule felter. A/B er randomiseret pr. sætning. Åbn IKKE nøglefilen før begge faser er afsluttet.',
        ['ID','Inputtekst','Output A','Output B','A: sproglig korrekthed','A: betydning bevaret','A: unødvendige ændringer','A: format OK','B: sproglig korrekthed','B: betydning bevaret','B: unødvendige ændringer','B: format OK','Kommentar'],
        phase1, [4,5,6,7,8,9,10,11,12])
    build_book(args.output_dir/'02_referencekontrol.xlsx',
        'Fase 2 · Kontrol mod DaLA-reference',
        'Udfyld først EFTER fase 1. Vurdér om den tilsigtede fejl er rettet; på kontrolsætninger: Ikke relevant.',
        ['ID','Inputtekst','DaLA-reference','Output A','Output B','A: tilsigtet fejl rettet','B: tilsigtet fejl rettet','Referenceproblemer?','Kommentar'],
        phase2, [5,6,7,8])
    write_csv(args.output_dir/'03_FORTROLIG_modelnoegle.csv',secret,list(secret[0]))
    manifest = {
        'selection': 'målrettet, eksplorativt; IKKE populationsrepræsentativt',
        'seed': args.seed,
        'source': str(args.input),
        'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'model_A_B_randomized_per_item': True,
        'phase1_reference_hidden': True,
        'selection_groups': dict(GROUPS),
        'pool_sizes_before_de_duplication': pool_sizes,
        'n_unique_items': len(selection), 'n_model_outputs': 2*len(selection),
        'scoring_guidance': {
            'sproglig_korrekthed':'Ja / Delvist / Nej / Usikkert',
            'betydning_bevaret':'Ja / Delvist / Nej / Usikkert',
            'unoedvendige_aendringer':'Nej / Valgfrie / Skadelige / Usikkert',
            'format_ok':'Ja / Nej',
            'tilsigtet_fejl_rettet':'Ja / Nej / Usikkert / Ikke relevant',
        }
    }
    (args.output_dir/'04_sample_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Klar: 30 unikke observationer, 60 blindede modeloutputs.')
    print('Fordeling: '+', '.join(f'{k}={n}' for k,n in GROUPS))
    for name in ('01_blind_vurdering.xlsx','02_referencekontrol.xlsx','03_FORTROLIG_modelnoegle.csv','04_sample_manifest.json'):
        print('  '+str(args.output_dir/name))


if __name__ == '__main__':
    main()
