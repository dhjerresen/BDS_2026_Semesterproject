#!/usr/bin/env python3
"""DaLA GEC: reproducible, local visualizations from analyze_results.py CSV files.

Python 3.10+. Dependencies: matplotlib and numpy.
Only reads CSVs. Creates figures in a separate directory; never modifies inputs.
"""
# Enable modern type-annotation behavior while keeping compatibility with Python 3.10+.
from __future__ import annotations

# Standard-library modules handle command-line options, CSV data, numerical checks, and file paths.
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

# Matplotlib creates and saves the charts; NumPy organizes numerical values for plots.
import matplotlib
# Use a non-interactive backend so figures can be generated without opening a display window.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Convert full Hugging Face model identifiers into short, readable labels for charts.
NAMES = {
    'danish-foundation-models/munin-apertus-8b': 'Munin Apertus 8B',
    'NLPnorth/snakmodel-7b-instruct': 'SnakModel 7B',
    'mistralai/Mistral-7B-Instruct-v0.3': 'Mistral 7B',
    'microsoft/Phi-4-mini-instruct': 'Phi-4-mini',
    'Qwen/Qwen2.5-3B-Instruct': 'Qwen2.5 3B',
    'Qwen/Qwen2.5-7B-Instruct': 'Qwen2.5 7B',
}
# Map internal error-type codes to Danish descriptions shown in the figures.
ERROR_LABELS = {
    'corrupt_adjective_r': 'Tillægsord: -r',
    'corrupt_ende_ene': '-ende / -ene',
    'corrupt_genitive': 'Genitiv',
    'corrupt_noun_r': 'Navneord: -r',
    'corrupt_spelling': 'Stavning',
    'corrupt_verb_r': 'Udsagnsord: -r',
    'delete': 'Slettet ord',
    'flip_en_et_suffix': 'Endelser: -en / -et',
    'flip_far_for': 'Får / for',
    'flip_han_hun_to_det': 'Han/hun / det',
    'flip_indefinite_article': 'Artikel: en / et',
    'flip_ligge_laegge': 'Ligge / lægge',
    'flip_neighbours': 'Naboord byttet',
    'flip_nogle_nogen': 'Nogle / nogen',
    'flip_pronouns': 'Pronomener',
    'flip_som_der': 'Som / der',
}
# Assign each model a consistent color across the different visualizations.
COLORS = {
    'Munin Apertus 8B': '#146C66',
    'SnakModel 7B': '#3C9D91',
    'Mistral 7B': '#6B78B2',
    'Phi-4-mini': '#B57F61',
    'Qwen2.5 3B': '#888B96',
    'Qwen2.5 7B': '#626772',
}
# Shared colors keep titles, annotations, and grid lines visually consistent.
INK = '#233243'
MUTED = '#637182'
GRID = '#E7EDF2'


# Load a CSV only after checking that the file exists and contains the expected columns.
def load_csv(path: Path, required: set[str]):
    if not path.is_file():
        raise ValueError(f'Mangler fil: {path}. Kør først analyze_results.py.')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or ())
        missing = required - columns
        if missing:
            raise ValueError(f'{path.name} mangler kolonner: {", ".join(sorted(missing))}')
        rows = list(reader)
    # Stop early if there is no data to visualize.
    if not rows:
        raise ValueError(f'{path.name} er tom')
    return rows


# Convert CSV strings to finite numbers, rejecting missing or invalid metric values.
def num(row, key):
    try:
        value = float(row[key].strip())
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        raise ValueError(f'Ugyldigt tal i {key}: {row!r}') from e
    if not math.isfinite(value):
        raise ValueError(f'Ikke-endeligt tal i kolonnen {key}')
    return value


# Use a short display name, falling back to the last part of the model identifier.
def short(model):
    return NAMES.get(model, model.split('/')[-1])


# Define a shared plotting style so figures use consistent fonts, spacing, and colors.
def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.titlesize': 15, 'axes.titleweight': 'bold',
        'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42,
    })


# Save each figure as PNG, PDF, or both, then close it to free memory.
def save(fig, destination, stem, formats):
    fig.savefig(destination / f'{stem}.png', dpi=210, bbox_inches='tight') if 'png' in formats else None
    fig.savefig(destination / f'{stem}.pdf', bbox_inches='tight') if 'pdf' in formats else None
    plt.close(fig)
    print('  ✓', stem)


# Add a small methodological note underneath a chart.
def footer(ax, message):
    ax.text(0, -0.14, message, transform=ax.transAxes, ha='left', va='top', fontsize=8.6, color=MUTED)


# Rank models by Exact Match accuracy on erroneous inputs, highest first.
def ordered(summary):
    return sorted(summary, key=lambda r: -num(r, 'exact_match_errors_pct'))


# Reusable horizontal bar chart for model-level metrics such as accuracy or latency.
def horizontal(summary, key, title, subtitle, filename, destination, formats, reversed_is_good=False):
    # Sort by the metric being displayed so differences are easier to compare.
    order = sorted(summary, key=lambda r: num(r, key))
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    names = [short(r['model']) for r in order]
    vals = [num(r, key) for r in order]
    barcols = [COLORS.get(name, '#718096') for name in names]
    ax.barh(names, vals, color=barcols, height=0.67)
    ax.set_xlim(0, max(100 if key.endswith('_pct') else 0, max(vals) * 1.16))
    ax.set_title(title, loc='left', pad=22)
    ax.text(0, 1.025, subtitle, fontsize=9.3, color=MUTED, transform=ax.transAxes)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0, pad=10)
    # Percentage metrics get percent-formatted axis labels; latency remains in seconds.
    if key.endswith('_pct'):
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f} %'))
    for i, v in enumerate(vals):
        label = f'{v:.1f} %' if key.endswith('_pct') else f'{v:.2f} s'
        ax.text(v + ax.get_xlim()[1]*0.012, i, label, va='center', fontsize=10, fontweight='bold')
    # For measures like latency, lower numbers represent better performance.
    if reversed_is_good:
        footer(ax, 'Lavere værdi er bedre. Inkluderer alle registrerede outputs.')
    save(fig, destination, filename, formats)


# Compare error correction quality against how often control sentences are not preserved.
def scatter(summary, destination, formats):
    fig, ax = plt.subplots(figsize=(10.4, 6.5))
    # The control-change metric includes non-OK outputs, not just unnecessary grammatical edits.
    # CMR all includes failed outputs, so label carefully.
    for r in summary:
        name = short(r['model'])
        x = num(r, 'modified_controls_all_pct')
        y = num(r, 'exact_match_errors_pct')
        ax.scatter(x, y, s=155, color=COLORS.get(name, '#777'), edgecolor='white', linewidth=1.8, zorder=3)
    # Offset model names to make overlapping points easier to read.
        dx, dy = {'Munin Apertus 8B':(1.5,-1.8),'SnakModel 7B':(1.5,1.7),'Phi-4-mini':(1.5,1.5),'Qwen2.5 3B':(-16,1.6),'Qwen2.5 7B':(-16,-2.5),'Mistral 7B':(-16,1.5)}.get(name,(1.5,1.5))
        ax.annotate(name, (x,y), xytext=(x+dx,y+dy), color=INK, fontsize=9)
    ax.set_xlim(0,100)
    ax.set_ylim(0,max(55, max(num(r, 'exact_match_errors_pct') for r in summary)*1.13))
    ax.set_xlabel('Kontroller ikke bevaret uændret (%)  →')
    ax.set_ylabel('Exact Match på fejlsætninger (%)  →')
    ax.set_title('Korrektionskvalitet vs. ændring af kontroltekst',loc='left',pad=16)
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    footer(ax,'Øverst til venstre er ønskeligt. Kontroller ikke bevaret = inkl. non-OK; ikke ensbetydende med sproglige fejl.')
    fig.subplots_adjust(bottom=0.16)
    save(fig, destination, '03_kvalitet_vs_kontrolaendringer', formats)


# Display the percentage of generations that were not completed with an OK status.
def status_plot(summary, destination, formats):
    order = sorted(summary,key=lambda r: -num(r,'non_ok_rate_pct'))
    names = [short(r['model']) for r in order]
    pct = [num(r,'non_ok_rate_pct') for r in order]
    fig, ax = plt.subplots(figsize=(10.2,5.6))
    ax.barh(names[::-1], pct[::-1], color='#C47952',height=.65)
    ax.set_xlim(0,max(25, max(pct)*1.18))
    ax.set_title('Ufuldstændige eller fejlede genereringer',loc='left',pad=22)
    ax.text(0,1.025,'Andel med status forskellig fra ok',fontsize=9.3,color=MUTED,transform=ax.transAxes)
    ax.xaxis.grid(True,color=GRID)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False);ax.spines['bottom'].set_visible(False)
    ax.tick_params(length=0,pad=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:.0f} %'))
    for i,v in enumerate(pct[::-1]):
        ax.text(v+.3,i,f'{v:.2f} %',va='center',fontweight='bold')
    footer(ax,'Non-OK = possibly_truncated + empty_output + error; ikke automatisk en infrastrukturefejl.')
    save(fig,destination,'05_non_ok_rate',formats)


# Show Exact Match accuracy for each combination of model and grammatical error type.
def heatmap(categories, summary, destination, formats):
    lookup = {(r['model'],r['error_type']):num(r,'exact_match_pct') for r in categories}
    labels = sorted({r['error_type'] for r in categories})
    models = [r['model'] for r in ordered(summary)]
    # Require a complete model-by-error-type matrix rather than silently omitting categories.
    missing = [(m,typ) for m in models for typ in labels if (m,typ) not in lookup]
    if missing:
        raise ValueError(f'Mangler {len(missing)} kombinationer af model og fejltype i error_type_accuracy.csv')
    # Arrange percentages into rows (error types) and columns (models).
    matrix = np.array([[lookup[(m,t)] for m in models] for t in labels])
    fig,ax = plt.subplots(figsize=(12,9.1))
    cmap = LinearSegmentedColormap.from_list('gec',['#F1F5F7','#BFDAD8','#4C9A92','#176862'])
    im=ax.imshow(matrix,cmap=cmap,vmin=0,vmax=100,aspect='auto')
    ax.set_xticks(np.arange(len(models)),[short(m) for m in models],rotation=35,ha='right',fontsize=9)
    ax.set_yticks(np.arange(len(labels)),[ERROR_LABELS.get(t,t) for t in labels])
    ax.tick_params(length=0)
    # Print values inside cells; bold type highlights the best model within each error type.
    for i in range(len(labels)):
        for j in range(len(models)):
            val=matrix[i,j]
            ax.text(j,i,f'{val:.0f}%',ha='center',va='center',fontsize=9,color='white' if val>=55 else INK,
                    fontweight='bold' if val==np.max(matrix[i,:]) else 'normal')
    ax.set_title('Exact Match på tværs af 16 fejltyper',loc='left',pad=18)
    cbar=fig.colorbar(im,ax=ax,shrink=.76,pad=.02);cbar.set_label('Exact Match (%)')
    fig.text(.13,.015,'Procent på alle fejlinputs i hver kategori; mørkere farve = højere Exact Match.',fontsize=9,color=MUTED)
    save(fig,destination,'06_heatmap_fejltyper',formats)


# Calculate Munin minus SnakModel Exact Match, measured in percentage points per error type.
def contrast(categories, destination, formats):
    a='danish-foundation-models/munin-apertus-8b'
    b='NLPnorth/snakmodel-7b-instruct'
    by={(r['model'],r['error_type']):r for r in categories}
    labels=sorted({r['error_type'] for r in categories})
    # Skip this comparison when either model has no result for one or more error types.
    if any((a,t) not in by or (b,t) not in by for t in labels):
        print('  ! Springer Munin vs Snak over (en af modellerne mangler)',file=sys.stderr)
        return
    # Positive differences favor Munin; negative differences favor SnakModel.
    data=sorted([(t,num(by[(a,t)],'exact_match_pct')-num(by[(b,t)],'exact_match_pct'),int(num(by[(a,t)],'n'))) for t in labels],key=lambda x:x[1])
    fig,ax=plt.subplots(figsize=(10.7,8))
    diffs=[r[1] for r in data]
    names=[f'{ERROR_LABELS.get(t,t)}  (n={n})' for t,_,n in data]
    ax.barh(names,diffs,color=['#3C9D91' if d>=0 else '#6B78B2' for d in diffs],height=.72)
    ax.axvline(0,color=INK,linewidth=1)
    maxabs=max(abs(d) for d in diffs)
    ax.set_xlim(-max(13,maxabs*1.30),max(13,maxabs*1.30))
    ax.xaxis.grid(color=GRID);ax.set_axisbelow(True)
    ax.set_title('Munin sammenlignet med SnakModel',loc='left',pad=22)
    ax.text(0,1.025,'Forskel i Exact Match (Munin minus SnakModel), procentpoint',fontsize=9.3,color=MUTED,transform=ax.transAxes)
    for i,d in enumerate(diffs):
        ax.text(d + (0.65 if d>=0 else -0.65),i,f'{d:+.1f}',va='center',ha='left' if d>=0 else 'right',fontsize=9,fontweight='bold')
    ax.set_xlabel('← SnakModel bedre     Munin bedre →')
    footer(ax,'Bemærk kategoriernes stikprøvestørrelser (n). Små kategorier giver mere usikre forskelle.')
    save(fig,destination,'07_munin_vs_snak_fejltyper',formats)


# Plot the number of examples per grammatical error category to provide sample-size context.
def sizes(categories,destination,formats):
    first={}
    # Verify the underlying category sample sizes are equal across models.
    for r in categories:
        t=r['error_type']; n=int(num(r,'n'))
        if t in first and first[t] != n:
            raise ValueError(f'Fejltypen {t} har forskelligt antal inputs mellem modeller')
        first[t]=n
    data=sorted(first.items(),key=lambda item:item[1])
    fig,ax=plt.subplots(figsize=(10,8))
    vals=[n for _,n in data]
    ax.barh([ERROR_LABELS.get(t,t) for t,_ in data],vals,color='#6687A0',height=.7)
    ax.set_xlim(0,max(vals)*1.17)
    ax.set_title('Fordeling af DaLA-fejltyper',loc='left',pad=17)
    ax.set_xlabel('Antal fejlsætninger')
    ax.xaxis.grid(color=GRID);ax.set_axisbelow(True)
    for i,n in enumerate(vals):ax.text(n+max(vals)*.014,i,str(n),va='center',fontsize=9,fontweight='bold')
    footer(ax,f'I alt {sum(vals):,} fejlsætninger fordelt på {len(vals)} kategorier.'.replace(',','.'))
    save(fig,destination,'08_antal_pr_fejltype',formats)


# Compare the distribution of response times using successful generations only.
def latency_box(rows, summary, destination, formats):
    # Optional all_predictions.csv: all rows could have different lengths; independent figure.
    # Keep model ordering consistent with the Exact Match ranking in other figures.
    model_order=[r['model'] for r in ordered(summary)]
    values=defaultdict(list)
    for r in rows:
        if r['model'] in model_order and r.get('status')=='ok':
            values[r['model']].append(num(r,'latency_seconds'))
    # A boxplot needs at least one successful timed observation for every model.
    if any(not values[m] for m in model_order):
        print('  ! Springer latency-boksplot over (manglende OK-observationer)',file=sys.stderr)
        return
    fig,ax=plt.subplots(figsize=(10.2,5.7))
    # The boxplot summarizes median and quartiles; individual outlier markers are not drawn.
    # Show distribution without hiding outliers: full data in boxplot, visualization axis clipped at pooled p99.
    box=ax.boxplot([values[m] for m in model_order],vert=False,patch_artist=True,showfliers=False,
                   medianprops=dict(color=INK,linewidth=1.8),
                   whiskerprops=dict(color=MUTED),capprops=dict(color=MUTED))
    ax.set_yticks(np.arange(1, len(model_order) + 1), [short(m) for m in model_order])
    for patch,m in zip(box['boxes'],model_order): patch.set_facecolor(COLORS.get(short(m),'#7F98A7'))
    ax.invert_yaxis()
    ax.set_title('Fordeling af svartider for afsluttede outputs',loc='left',pad=18)
    ax.set_xlabel('Sekunder pr. observation')
    ax.xaxis.grid(color=GRID);ax.set_axisbelow(True)
    footer(ax,'Boks: 25.–75. percentil, streg: median. Ekstreme værdier er skjult visuelt; data er ikke fjernet.')
    save(fig,destination,'09_latency_fordeling',formats)


# Parse locations and output format, validate analysis files, then create all figures.
def main():
    parser=argparse.ArgumentParser(description='Visualiser DaLA GEC-analyser som PNG/PDF')
    parser.add_argument('--analysis-dir',type=Path,default=Path('analysis'),help='Mappe med CSV-filer fra analyze_results.py')
    parser.add_argument('--figures-dir',type=Path,default=Path('figures'),help='Mappe til nye figurer')
    parser.add_argument('--format',choices=['both','png','pdf'],default='both',help='Filformat')
    args=parser.parse_args()
    # Specify the model-level metrics required by the plotting functions.
    reqsum={'model','exact_match_errors_pct','unchanged_controls_pct','modified_controls_all_pct','non_ok_rate_pct','latency_median_all_s'}
    summary=load_csv(args.analysis_dir/'model_summary.csv',reqsum)
    categories=load_csv(args.analysis_dir/'error_type_accuracy.csv',{'model','error_type','n','exact_match_pct'})
    # Duplicate model rows would make the charts ambiguous, so reject them.
    if len({r['model'] for r in summary}) != len(summary):
        raise ValueError('Dublerede modeller i model_summary.csv')
    args.figures_dir.mkdir(parents=True,exist_ok=True)
    formats=['png','pdf'] if args.format=='both' else [args.format]
    style()
    print(f'Laver figurer for {len(summary)} modeller i {args.figures_dir}:')
    # Figure 1: compare the share of erroneous sentences matching the reference exactly.
    horizontal(summary,'exact_match_errors_pct','Korrektionskvalitet: Exact Match',
               'Andel fejlsætninger identiske med referencen (alle 1.664 pr. model)',
               '01_exact_match',args.figures_dir,formats)
    # Figure 2: show how reliably the models leave correct sentences untouched.
    horizontal(summary,'unchanged_controls_pct','Hvor ofte bevares korrekt kontroltekst?',
               'Andel kontrolinputs returneret præcis uændret (alle 1.664 pr. model)',
               '02_uaendrede_kontroller',args.figures_dir,formats)
    # Figure 3: visualize the quality versus control-preservation trade-off.
    scatter(summary,args.figures_dir,formats)
    # Figure 4: compare typical per-sentence inference times, including non-OK results.
    horizontal(summary,'latency_median_all_s','Median svartid pr. sætning',
               'Tid inkl. non-OK-genereringer, ekskl. modelindlæsning og warmup',
               '04_median_latency',args.figures_dir,formats,reversed_is_good=True)
    # Figure 5: compare non-OK generation rates across models.
    status_plot(summary,args.figures_dir,formats)
    # Figure 6: identify differences in Exact Match by grammatical error category.
    heatmap(categories,summary,args.figures_dir,formats)
    # Figure 7: focus on the category-wise difference between Munin and SnakModel.
    contrast(categories,args.figures_dir,formats)
    # Figure 8: show the number of test sentences in each error category.
    sizes(categories,args.figures_dir,formats)
    # Figure 9 is optional because it needs observation-level latency data.
    predpath=args.analysis_dir/'all_predictions.csv'
    if predpath.exists():
        predictions=load_csv(predpath,{'model','status','latency_seconds'})
        latency_box(predictions,summary,args.figures_dir,formats)
    else:
        print('  - all_predictions.csv ikke fundet: springer valgfrit latency-boksplot over')
    print('Færdig. CSV-input blev ikke ændret.')


# Run the visualization pipeline only when this file is executed directly.
if __name__=='__main__':
    try: main()
    except (OSError,ValueError) as e:
        print('FEJL:',e,file=sys.stderr)
        sys.exit(1)
