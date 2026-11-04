# Dataset preparation for Danish GEC evaluation

A local Python implementation of minimal-pair reconstruction from the **published DaLA standard** acceptability dataset. It does NOT generate new errors; it reverses the fixed shuffle in the original code and validates the alignment before exporting pairs.

## Setup

Requires Python >= 3.10 (3.12 recommended).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_dataset.py
```

The default source is `danish-foundation-models/dala` from Hugging Face, downloaded once into the normal Hugging Face cache. The dataset itself is not included in this download.

Alternatively, use CSVs saved by the DaLA creators (`dala_da_train.csv`, `dala_da_val.csv`, `dala_da_test.csv`, or `train.csv`, `val.csv`, `test.csv`):

```bash
python prepare_dataset.py --input-dir data/raw --output-dir data/processed
```

To pin the Hugging Face dataset revision for a finalized experiment:

```bash
python prepare_dataset.py --revision YOUR_HUGGING_FACE_COMMIT_SHA
```

## Output

- `data/processed/dala_gec_pairs.csv`: `pair_id`, `split`, `error_type`, `incorrect_text`, `reference_text`.
- `data/processed/dala_controls.csv`: matching originally correct controls (`input_text == reference_text`).
- `data/processed/dala_evaluation_inputs.csv`: one row per model request, including `is_control`.
- `data/processed/dala_manual_review.csv`: stratified quality review from train and val, five examples per error category by default.
- `data/processed/dataset_report.json`: categories, counts, duplicates, overlap and source metadata.

## Integrity and caveats

The original `dala/create_dala.py` concatenates correct/incorrect rows and shuffles with `pandas.DataFrame.sample(frac=1.0, random_state=4242)`. The script reverses this operation and verifies that both halves have expected labels, corruption types, nonempty texts, and different strings. If alignment fails, the script exits with an error and does not export potentially incorrect pairs.

A successful programmatic reconstruction **does not demonstrate linguistic validity**: original sentences may have issues unrelated to the injected corruption, and more than one correction might be acceptable. Audit the stratified manual sample before using the data for the final evaluation, and do not use the reserved test split to tune prompts. Controls and corrupted sentences with the same `pair_id` must always be grouped when making new data splits.

The `full_train` split (if present) is intentionally **excluded**, because it overlaps the other original splits in the authors' creation logic.

## Sources

- [Official DaLA code](https://github.com/N-essuno/DaLA/blob/main/dala/create_dala.py)
- [Published DaLA dataset](https://huggingface.co/datasets/danish-foundation-models/dala)
- [DaLA paper (LREC 2026)](https://aclanthology.org/2026.lrec-1.338/)
