# DaLA pooled zero-shot evaluation – simplified

Denne version har kun to aktive scripts:

1. `prepare_dataset.py` rekonstruerer original og fejlbehæftet tekst fra DaLA.
2. `prepare_pooled_evaluation.py` samler alle observationer og validerer metadata.

## Kør i projektets rodmappe

```bash
python -m pip install -r requirements.txt
python prepare_dataset.py --revision 0cc96091dc265d14fb5d507114690fd85857b6da
python prepare_pooled_evaluation.py
```

`prepare_pooled_evaluation.py` læser kun:

- `data/processed/dala_gec_pairs.csv`
- `data/processed/dala_controls.csv`
- `data/processed/dataset_report.json`

Og skriver:

- `data/pooled/all_errors.csv` (1.664)
- `data/pooled/all_controls.csv` (1.664)
- `data/pooled/evaluation_inputs.csv` (3.328)
- `data/pooled/pooled_manifest.json` (kilde, antal og hashes)

Ingen review-filer, tidligere test-splits eller `data/final` er nødvendige. `source_split` bevares kun som kilde-metadata. Ingen observationer ekskluderes eller omskrives.

Hvis gamle pooled-filer allerede findes og afviger, stopper scriptet for at undgå utilsigtet overskrivning. Gennemgå dem og kør kun `python prepare_pooled_evaluation.py --force`, hvis du vil erstatte dem.

## Metodeforbehold

Exact Match måler overensstemmelse med den eksisterende DaLA-reference. Nogle referencekorrektioner kan være tvetydige eller indeholde øvrige sproglige fejl. Gem gerne de tidligere AI-assisterede screeninger som arkivmateriale, men de indgår ikke længere i den aktive datagenerering. Dokumentér én fælles zero-shot-prompt før modelkørsel.
