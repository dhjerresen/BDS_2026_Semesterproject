# DaLA: Zero-Shot Evaluation of Danish Grammatical Error Correction

This project compares instruction-tuned language models on **Danish grammatical error correction (GEC)** using reconstructed sentence pairs from the [DaLA dataset](https://huggingface.co/datasets/danish-foundation-models/dala). The evaluation is **zero-shot**: models are asked to correct sentences without task-specific fine-tuning or using the dataset to develop model-specific prompts.

The pipeline reconstructs the original and corrupted sentences, prepares a pooled evaluation set, generates model corrections, and evaluates the outputs quantitatively. Additional scripts create visualizations and support a smaller, blinded manual evaluation.

## 1. Project workflow

```text
DaLA dataset (pinned revision)
            |
            v
prepare_dataset.py
  - Reconstructs correct/incorrect sentence pairs
  - Creates matched control sentences and a dataset report
            |
            v
prepare_pooled_evaluation.py
  - Validates source files and metadata
  - Combines 1,664 incorrect inputs and 1,664 controls
            |
            v
Zero-shot inference script (Hugging Face / PyTorch)
  - Runs one model at a time, one sentence at a time
  - Saves predictions as JSONL and run metadata as JSON
            |
            v
analyze_results.py
  - Validates model outputs and comparability
  - Computes Exact Match, control-preservation and latency metrics
            |
            +------------------------+
            |                        |
            v                        v
calculate_gleu.py           Visualization script
  - Standalone revised       - Generates PNG/PDF charts
    GEC-GLEU-style score       from analysis CSVs

Optional: prepare_manual_evaluation.py
  - Creates blinded Excel review forms for Munin and SnakModel
```

The source dataset's `train`, `val`, and `test` labels are retained as **provenance metadata**, not used as partitions for fitting or tuning. All reconstructed examples are pooled for inference.

## 2. Dataset and experimental design

| Component | Count |
| --- | ---: |
| Incorrect sentences requiring correction | 1,664 |
| Corresponding correct control sentences | 1,664 |
| Total evaluation inputs | **3,328** |
| Error categories | 16 |

Each erroneous sentence is paired with its original reference sentence. A control input uses that same reference sentence as its input, making it possible to examine whether a model changes text that is already correct according to the dataset.

The original DaLA split counts for the reconstructed **pairs** are 512 train, 128 validation, and 1,024 test. These labels are not used to exclude observations from the pooled experiment.

The data reconstruction assumes the row ordering and random shuffle used in the original DaLA creation code. It reverses the shuffle using seed `4242` and checks labels, corruption categories, nonempty text, and pair consistency. This is a reconstruction and structural validation procedure; it does **not** independently establish the linguistic correctness of every reference.

Pinned DaLA dataset revision:

```text
0cc96091dc265d14fb5d507114690fd85857b6da
```

## 3. Requirements and setup

The pipeline is written in Python (Python 3.10+ for the included analysis scripts). Install the project's declared dependencies in the project root:

```bash
python -m pip install -r requirements.txt
```

Different stages have different requirements:

- **Dataset reconstruction:** `pandas` and the Hugging Face `datasets` library when downloading directly from Hugging Face; local CSV inputs are also supported.
- **Pooled dataset preparation and core analysis:** Python standard library.
- **Model inference:** PyTorch, Transformers, Hugging Face Hub access, and a CUDA GPU. Optional 4-bit quantization uses bitsandbytes via Transformers.
- **GEC-GLEU plots and additional visualizations:** Matplotlib; the additional visualization script also uses NumPy.
- **Optional blinded manual-review workbooks:** `openpyxl`.

Model inference was designed for a GPU environment; its error messages also reference Slurm/Singularity execution. Reconstructing datasets, combining inputs, and analyzing saved results can be done separately from GPU inference.

## 4. Step-by-step execution

Run the following commands from the project root unless another location is explicitly supplied.

### Step 1 — Reconstruct DaLA sentence pairs

```bash
python prepare_dataset.py --revision 0cc96091dc265d14fb5d507114690fd85857b6da
```

The script loads DaLA, reverses the documented shuffle, reconstructs the incorrect/reference pairs, checks their structural consistency, and produces:

| Output | Purpose |
| --- | --- |
| `data/processed/dala_gec_pairs.csv` | Incorrect sentences, reference sentences, and error categories |
| `data/processed/dala_controls.csv` | Correct, unchanged control inputs matched to the pairs |
| `data/processed/dala_evaluation_inputs.csv` | Combined preliminary evaluation inputs |
| `data/processed/dala_manual_review.csv` | Stratified sample prepared for optional review; not populated with judgments |
| `data/processed/dataset_report.json` | Dataset counts, source revision, versions, duplicate checks, and split information |

The manual-review sample is generated from the original training and validation splits and does not determine which examples enter the pooled model comparison.

### Step 2 — Prepare the pooled zero-shot evaluation set

```bash
python prepare_pooled_evaluation.py
```

This script reads **only** these upstream files:

```text
data/processed/dala_gec_pairs.csv
data/processed/dala_controls.csv
data/processed/dataset_report.json
```

It verifies dataset identity and revision, shuffle seed, split counts, category counts, pair IDs, and control/reference alignment. It then creates:

| Output | Rows / contents |
| --- | --- |
| `data/pooled/all_errors.csv` | 1,664 incorrect sentences |
| `data/pooled/all_controls.csv` | 1,664 correct controls |
| `data/pooled/evaluation_inputs.csv` | 3,328 combined inputs |
| `data/pooled/pooled_manifest.json` | Protocol information, counts, categories, provenance, and SHA-256 hashes |

The pooled preparation stage does **not** depend on `reviews/`, `review_data/`, or `data/final/`. It does not use historical review decisions to relabel or filter examples.

If an existing pooled output differs from the newly generated content, the script stops rather than overwriting it automatically. After reviewing the difference, intentional replacement is possible with:

```bash
python prepare_pooled_evaluation.py --force
```

### Step 3 — Run zero-shot inference

The inference script accepts a Hugging Face model ID, a pinned model revision, an input CSV, a shared prompt file, and a dedicated output JSONL path. The exact **inference script filename** was not established from the shared code, so substitute the filename in your project for `<inference_script>.py` below.

```bash
python <inference_script>.py \
  --model <hugging-face-model-id> \
  --model-revision <40-character-model-commit-sha> \
  --input data/pooled/evaluation_inputs.csv \
  --prompt-file gec_prompt.txt \
  --output outputs/<unique-model-run>.jsonl
```

The same fixed prompt is passed with each input sentence in a single user message. Only the input sentence, **not the reference correction**, is supplied to the model. The script uses greedy decoding (`do_sample=False`, one beam) and evaluates a batch size of one input at a time. It also supports token limits, attention-backend selection, optional 4-bit quantization, GPU warmup, a small smoke-test limit, and a validation-only mode.

For the main experiment, use a **40-character model commit SHA** rather than the floating `main` branch. Floating revisions are allowed by the script only when explicitly enabled for tests. Run each model with its own output filename, while keeping the prompt and dataset fixed.

Each JSONL observation records its ID, source input, reference (for later scoring), model output, status, token counts, and latency. The accompanying `.metadata.json` file records model/environment settings and version information. Existing JSONL outputs can be resumed after consistency checks; model-generation problems are recorded instead of silently discarded.

> **Timing definition:** Reported per-item latency includes tokenization, generation, and decoding, but excludes model loading and warmup. Comparisons may still be affected by hardware, quantization, attention implementation, and environment differences.

### Step 4 — Analyze model outputs

```bash
python analyze_results.py \
  --outputs outputs \
  --analysis-dir analysis \
  --inputs data/pooled/evaluation_inputs.csv
```

The script reads the model JSONL outputs, excludes filenames containing `smoke`, and verifies that model runs use identical input IDs, source/reference data, prompts, and dataset hashes. It also validates paired control examples and checks the available run metadata. A full run must contain **3,328 observations per model**; partial runs are not silently scored as complete.

Produced outputs:

| File | Purpose |
| --- | --- |
| `analysis/model_summary.csv` | Overall per-model accuracy, control preservation, non-OK statuses, and runtime summaries |
| `analysis/error_type_accuracy.csv` | Exact Match and status counts by grammatical error type |
| `analysis/all_predictions.csv` | Observation-level model predictions and metrics |
| `analysis/run_configurations.csv` | Inference settings and environment metadata |
| `analysis/validation_report.json` | Validation outcomes, warnings, and metric definitions |

### Step 5 — Calculate GEC-GLEU-style scores

```bash
python calculate_gleu.py \
  --input analysis/all_predictions.csv \
  --output-dir analysis/gleu
```

This is a **standalone revised GEC-GLEU-style implementation**, not the official GEC-GLEU scorer. It uses source, hypothesis, and reference text, whitespace tokenization, 1–4-grams, a penalty for retained source n-grams unsupported by the reference, and a BLEU-style brevity penalty.

Non-OK outputs (failed, truncated, or empty) count as empty hypotheses rather than being removed from scoring. The script compares model predictions with the unchanged-source baseline and reference-as-hypothesis oracle, and also reports category-level results.

Outputs include `analysis/gleu/gleu_model_summary.csv`, `analysis/gleu/gleu_by_error_type.csv`, and `analysis/gleu/gleu_metadata.json`. With Matplotlib installed, it also saves PNG/PDF plots unless `--no-plots` is specified.

### Step 6 — Generate comparison figures

The separate visualization script uses the CSVs from `analyze_results.py`. Its exact **filename** was not established from the shared code, so replace `<visualization_script>.py` with the actual project filename:

```bash
python <visualization_script>.py \
  --analysis-dir analysis \
  --figures-dir figures \
  --format both
```

It produces plots of Exact Match, unchanged controls, correction quality versus control modifications, median latency, non-OK rates, performance by error type, a Munin-versus-SnakModel comparison, and the distribution of error-category sizes. If `analysis/all_predictions.csv` is present, it also creates a latency box plot. The script reads analysis CSVs without modifying them.

## 5. Evaluation metrics

| Metric | Interpretation |
| --- | --- |
| **Exact Match (EM)** | Share of erroneous inputs where a successful model output matches the DaLA reference *exactly*. Denominator: all 1,664 erroneous inputs. |
| **Unchanged controls** | Share of the 1,664 controls successfully returned exactly as received. |
| **Modified controls (all)** | All controls not successfully returned unchanged, **including non-OK outputs**. This is **not** a pure measure of harmful overcorrection. |
| **Non-OK rate** | Share of all 3,328 inputs with `possibly_truncated`, `empty_output`, or `error` status. |
| **Latency** | Mean and median per-item runtime; the analysis reports all observations and successful (`ok`) observations separately. |
| **Standalone revised GEC-GLEU** | Corpus-level n-gram-based score taking input, output, and reference into account; evaluated on erroneous sentences rather than controls. |

Exact Match is a strict string comparison: a linguistically acceptable alternative can still score as incorrect if it differs from the single dataset reference. The different measures therefore capture complementary aspects of model behavior rather than a single complete notion of linguistic quality.

## 6. Optional blinded manual evaluation

```bash
python prepare_manual_evaluation.py \
  --input analysis/all_predictions.csv \
  --output-dir manual_evaluation \
  --seed 4242
```

This supplementary procedure compares **Munin Apertus 8B** and **SnakModel 7B** using a targeted sample of **30 unique observations** (60 model outputs), with six examples drawn from each of five groups:

1. Only Munin achieves Exact Match.
2. Only SnakModel achieves Exact Match.
3. Neither model achieves Exact Match.
4. At least one model modifies a correct control input.
5. A diversified random remainder from eligible observations.

The script randomizes model A/B positions **per observation** and shuffles the final sample. It creates two Excel workbooks: a first-phase blind assessment without references and a second-phase assessment against the DaLA reference. It also writes a confidential model key and a sample manifest. Keep the confidential key hidden from reviewers until both phases are complete.

This is **purposive, exploratory sampling**, not a population-representative random sample. It can illustrate model behavior and limitations but should not be presented as an unbiased estimate of performance across all inputs.

A separate review-reconstruction utility can reproduce older curated review CSVs from `review_data/annotations_v1.json` after verifying dataset versions and SHA-256 signatures. These annotations are historical, AI-assisted research material, **not independent linguistic validation**, and they are **not part of the active pooled dataset generation**.

## 7. Reproducibility and safeguards

The code includes several checks intended to make the comparison auditable:

- Pinned DaLA source revision and reconstruction shuffle seed.
- Matched error/control pairs with consistent IDs and references.
- Input and prompt SHA-256 hashes, recorded in run metadata and checked across models.
- Pinned model commits and recorded inference settings (e.g., token limits, precision, attention backend).
- Greedy decoding and a shared prompt, with no task-specific fine-tuning.
- Strict checks against incomplete or mismatched model outputs.
- Explicit tracking of failed, empty, and potentially truncated generations.
- Output conflict checks to avoid unintended replacement of previously generated files.

The README describes reproducibility controls present in the shared scripts; actual comparability still depends on running the intended models under sufficiently consistent conditions.

## 8. Methodological limitations

1. **Reference quality:** The DaLA references have not all been independently validated. Some may be ambiguous or contain other linguistic issues. Dataset reconstruction checks structure, not linguistic validity.
2. **Exact Match strictness:** Correct alternatives may receive no Exact Match credit when they do not equal the single reference character for character.
3. **Control interpretation:** Modifying an already correct input is not automatically harmful; the reported all-controls modification measure also includes unsuccessful generations.
4. **GEC-GLEU implementation:** The included scorer is a standalone implementation, not the authors' official software; report its exact name and settings when presenting scores.
5. **Fairness of speed comparisons:** Hardware, model size, precision, attention mode, and software versions can influence timing.
6. **Blinded review scope:** The optional manual sample is small and deliberately targeted, so it should support qualitative interpretation rather than population-level conclusions.
7. **Pooled evaluation design:** Original split assignments are only provenance fields. The project evaluates zero-shot correction and does not perform a train/validation/test model-development experiment.

## 9. References and research transparency

- [DaLA dataset on Hugging Face](https://huggingface.co/datasets/danish-foundation-models/dala)
- [DaLA dataset creation code](https://github.com/N-essuno/DaLA/blob/main/dala/create_dala.py)

Document the shared prompt, exact model revisions, GPU/software environment, any excluded smoke-test runs, and the limitations of reference-based metrics in the final project report.
