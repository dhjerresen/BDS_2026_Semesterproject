#!/usr/bin/env python3
"""DaLA zero-shot GEC inference on one local HF model, one input at a time.

Results go to JSONL and can be resumed. No training, scoring, or prompt changes.
Only the input sentence is sent to the model; reference is for later scoring.
"""
from __future__ import annotations

import argparse
import csv
import copy
import re
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# Calculate a SHA-256 fingerprint of a file.
# Why: it lets us verify that the same dataset/prompt was used across runs.
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


# Create a timezone-aware UTC timestamp for reproducible records.
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Load the evaluation CSV and check its quality before running a model.
# Every row needs the expected columns, a unique ID, and non-empty input text.
def read_inputs(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'item_id', 'pair_id', 'source_split', 'error_type', 'is_control', 'input_text', 'reference_text'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'Mangler kolonner: {sorted(required - set(reader.fieldnames or []))}')
        rows = list(reader)
    ids = [r['item_id'] for r in rows]
    if not rows or len(ids) != len(set(ids)) or any(not r['input_text'].strip() for r in rows):
        raise ValueError('Tom inputfil, dublerede item_id eller tomme inputs')
    return rows


# Model-output failures are retained and count as evaluated observations.
# Only genuine execution failures block a resume until examined.
# These are valid model outcomes that can be kept for later evaluation.
# A technical exception (status="error") is deliberately NOT allowed during resume.
RECORDED_MODEL_STATUSES = {'ok', 'possibly_truncated', 'empty_output'}


# Resume support: inspect an existing JSONL output file before appending new results.
# Reject malformed records, duplicate IDs, incompatible settings, and technical errors.
# This prevents accidentally mixing experiments or silently skipping failed cases.
def read_existing(path: Path, config: dict) -> tuple[set[str], dict[str, int]]:
    seen = set()
    status_counts: dict[str, int] = {}
    if not path.exists():
        return seen, status_counts
    with path.open(encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'Ugyldig JSONL linje {number}; reparér filen inden genstart') from e
            for key in config:
                if record.get(key) != config[key]:
                    raise ValueError(f'Output eksisterer med anden konfiguration ({key}, linje {number}); brug en ny --output')
            item_id = record.get('item_id')
            if not item_id or item_id in seen:
                raise ValueError(f'Manglende eller dubleret item_id: {item_id}')
            status = record.get('status')
            if status not in RECORDED_MODEL_STATUSES:
                raise ValueError(
                    f'Output indeholder teknisk fejl/ukendt status={status!r} ved {item_id} '
                    f'(linje {number}). Undersøg fejlen før genstart; ingen rækker slettes automatisk.'
                )
            status_counts[status] = status_counts.get(status, 0) + 1
            seen.add(item_id)
    return seen, status_counts


# Define command-line settings so the same pipeline can run on different models.
# Model, revision, input, output and generation parameters are configurable.
def args_parser():
    p = argparse.ArgumentParser(description='DaLA inference med én lokal HF-model per job')
    p.add_argument('--model', required=True, help='Hugging Face model ID')
    p.add_argument('--model-revision', default='main', help='Fast model-commit SHA til hovedeksperiment')
    p.add_argument('--allow-floating-revision', action='store_true', help='Kun smoke-tests: tillad main')
    p.add_argument('--attention', choices=('eager','sdpa'), default='eager', help='Attention-backend; eager undgår SDPA/sliding-window-advarsel')
    p.add_argument('--warmup', type=int, default=2, help='Antal ikke-målte genereringer inden test')
    p.add_argument('--input', type=Path, default=Path('data/pooled/evaluation_inputs.csv'))
    p.add_argument('--prompt-file', type=Path, default=Path('gec_prompt.txt'))
    p.add_argument('--output', type=Path, required=True, help='Unik JSONL-fil per model/configuration')
    p.add_argument('--quantization', choices=('none', '4bit'), default='none', help='Standard: none = BF16 uden vægtkvantisering')
    p.add_argument('--max-new-tokens', type=int, default=384)
    p.add_argument('--max-input-tokens', type=int, default=2048)
    p.add_argument('--limit', type=int, default=0, help='Teknisk smoke-test; 0 betyder alle')
    p.add_argument('--validate-only', action='store_true', help='Tjek lokale filer uden GPU/transformers')
    return p


# Main workflow: validate inputs, initialize the model, run inference, and save results.
def main() -> int:
    args = args_parser().parse_args()
    # Catch invalid generation limits and warmup/test counts early.
    if args.max_new_tokens <= 0 or args.max_input_tokens <= 0 or args.limit < 0 or args.warmup < 0:
        raise ValueError('Token-grænser skal være positive, og limit >= 0')
    # For the main experiment, pin the model to a specific 40-character commit SHA.
    # A moving branch like "main" is allowed only when explicitly requested for smoke tests.
    if not args.allow_floating_revision and not re.fullmatch(r'[0-9a-f]{40}', args.model_revision):
        raise ValueError('Hovedeksperiment kræver --model-revision med 40-tegns commit-SHA. Brug --allow-floating-revision kun til smoke-test.')
    if not args.input.is_file() or not args.prompt_file.is_file():
        raise FileNotFoundError('Mangler --input eller --prompt-file')
    # Read the fixed grammatical-error-correction instruction from disk.
    # The same prompt is applied to every example; empty prompts are not accepted.
    prompt = args.prompt_file.read_text(encoding='utf-8').strip()
    if not prompt:
        raise ValueError('Tom promptfil')
    rows = read_inputs(args.input)
    # Record experiment settings and file fingerprints with each prediction.
    # These fields let us trace results and compare only compatible runs.
    config = {
        'model': args.model,
        'model_revision': args.model_revision,
        'prompt_sha256': sha256(args.prompt_file),
        'dataset_sha256': sha256(args.input),
        'quantization': args.quantization,
        'max_new_tokens': args.max_new_tokens,
        'max_input_tokens': args.max_input_tokens,
        'attention': args.attention,
        'warmup': args.warmup,
        'decode_strategy': 'greedy_v1',
        'weight_precision': 'bf16' if args.quantization == 'none' else '4bit_nf4',
    }
    # Optionally restrict rows for a small technical test; zero means use all rows.
    if args.limit:
        rows = rows[:args.limit]
    # Find already processed IDs to avoid duplicate predictions on restart.
    seen, existing_status_counts = read_existing(args.output, config)
    print(f'Dataset: {len(rows)} inputs | allerede registreret: {len(seen)} '
          f'| tidligere status: {existing_status_counts} '
          f'| prompt_sha256={config["prompt_sha256"]}', flush=True)
    # Validation-only mode stops here: no GPU/model loading and no output edits.
    if args.validate_only:
        print('Validering OK. Ingen model indlæst og ingen outputs ændret.', flush=True)
        return 0

    # Load heavyweight ML libraries only after basic validation has passed.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Require an NVIDIA CUDA GPU; standard BF16 mode also needs BF16 support.
    if not torch.cuda.is_available():
        raise RuntimeError('Ingen CUDA-GPU fundet; kør via Slurm/Singularity med --nv')
    if args.quantization == 'none' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 understøttes ikke af GPU. Vælg en BF16-kompatibel GPU til hovedeksperimentet.')
    # Select numeric precision based on GPU capabilities.
    # BF16 is used when supported; otherwise FP16 is used for 4-bit mode.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Load the tokenizer for the exact chosen model revision.
    # If there is no padding token, reuse its end-of-sequence token.
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizeren mangler både pad_token og eos_token')
        tokenizer.pad_token = tokenizer.eos_token
    # Prepare model loading options, including automatic GPU placement and attention backend.
    model_kwargs = {'revision': args.model_revision, 'device_map': 'auto', 'torch_dtype': dtype, 'attn_implementation': args.attention}
    # Optional 4-bit NF4 weight quantization lowers model memory requirements.
    # Without it, weights are loaded using the selected floating-point precision.
    if args.quantization == '4bit':
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
    print(f'Indlæser {args.model} ({args.model_revision}), kvantisering={args.quantization}, dtype={dtype}', flush=True)
    # Measure model loading time separately from per-sentence inference time.
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    print(f'Model indlæst på {time.perf_counter()-t_load:.1f} sekunder', flush=True)
    # Use deterministic greedy decoding: no sampling and only one beam.
    # This reduces randomness so model comparisons are more consistent.
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    # Use one USER message for every model: some instruct chat templates
    # (e.g. older Mistral versions) do not support a separate system role.
    # The same fixed instruction and separator are used for every input/model.
    # Build the exact message sent to the model: fixed instruction plus one input.
    # Apply the model chat template, check token-length limits, and move IDs to GPU.
    # Reference/correct answer is NOT included in this message (avoids answer leakage).
    def encode(text: str):
        message = prompt + '\n\nTEKST TIL RETTELSE:\n' + text
        msgs = [{'role': 'user', 'content': message}]
        tokenized = tokenizer.apply_chat_template(msgs, tokenize=True,
                                                  add_generation_prompt=True, return_tensors='pt')
        if tokenized.shape[1] > args.max_input_tokens:
            raise ValueError(f'Input inkl. prompt har {tokenized.shape[1]} tokens > max_input_tokens={args.max_input_tokens}')
        return tokenized.to(model.device)

    # Verify the actual resolved Hub commit, not just the requested ref.
    # Resolve the actual model commit from Hugging Face for provenance.
    from huggingface_hub import HfApi
    resolved_sha = HfApi().model_info(args.model, revision=args.model_revision).sha
    # Store environment and runtime details: package versions, GPU, job ID and timing rules.
    # This separate metadata file helps another person reproduce the experiment.
    metadata = {
        **config, 'resolved_model_sha': resolved_sha, 'started_utc': utc_now(), 'python': platform.python_version(),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'gpu': torch.cuda.get_device_name(0), 'dtype': str(dtype),
        'slurm_job_id': os.getenv('SLURM_JOB_ID'),
        'batch_size': 1, 'do_sample': False,
        'prompt_delivery': 'single user message: instruction + newline newline TEKST TIL RETTELSE: + input',
        'latency_definition': 'per-item tokenization + GPU generation + decoding, excludes model loading',
        'reference_sent_to_model': False,
    }
    # If metadata already exists, verify compatibility; otherwise create it.
    # Never silently overwrite an older experiment with different settings.
    meta_path = args.output.with_suffix(args.output.suffix + '.metadata.json')
    if meta_path.exists():
        old_meta = json.loads(meta_path.read_text(encoding='utf-8'))
        for key, value in {**config, 'resolved_model_sha': resolved_sha}.items():
            if old_meta.get(key) != value:
                raise ValueError(f'Metadata mismatch for {key}. Brug ny outputfil.')
    else:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep only dataset rows that have not already been processed.
    pending = [row for row in rows if row['item_id'] not in seen]
    if not pending:
        print('Ingen nye inputs at behandle.', flush=True)
        return 0
    # Explicit warmup outside timing/results. Never include smoke-test results in final output.
    # Run untimed warmup generations before measuring real examples.
    # This reduces initialization overhead affecting the first measured example.
    if args.warmup:
        warmup_ids = encode('Dette er en dansk sætning.')
        for _ in range(args.warmup):
            with torch.inference_mode():
                model.generate(input_ids=warmup_ids, attention_mask=torch.ones_like(warmup_ids),
                               generation_config=generation_config, max_new_tokens=16)
        torch.cuda.synchronize()
        print(f'GPU warmup: {args.warmup} genereringer (ikke medtaget i målinger)', flush=True)
    new_status_counts: dict[str, int] = {}
    # Append results as JSONL (one JSON object per line).
    # Writing and flushing each case makes interrupted runs easier to resume.
    with args.output.open('a', encoding='utf-8') as f:
        # Process one sentence at a time (batch size = 1) and time the entire step.
        for n, row in enumerate(pending, 1):
            start = time.perf_counter()
            status, output, error, tokens, input_tokens = 'ok', '', '', 0, 0
            try:
                # Tokenize the source sentence; generate a continuation without gradient tracking.
                input_ids = encode(row['input_text'])
                input_tokens = input_ids.shape[1]
                with torch.inference_mode():
                    completion = model.generate(
                        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        generation_config=generation_config,
                    )
                torch.cuda.synchronize()
                # Remove prompt tokens so the stored output contains only newly generated text.
                completion_ids = completion[0, input_tokens:]
                tokens = len(completion_ids)
                output = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                # Flag empty predictions and outputs that may have hit the token cap.
                # These are recorded model outcomes, not necessarily execution failures.
                if not output:
                    status = 'empty_output'
                elif tokens >= args.max_new_tokens and completion_ids[-1].item() != tokenizer.eos_token_id:
                    status = 'possibly_truncated'
            # Capture technical failures for inspection rather than losing the whole run.
            # A later restart will refuse to ignore these error records.
            except Exception as exc:
                status, error = 'error', f'{type(exc).__name__}: {exc}'
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            duration = time.perf_counter() - start
            # Save the original input, reference (for later scoring), prediction, status,
            # timing, token counts, and configuration; reference was never sent to the model.
            record = {
                **config, 'item_id': row['item_id'], 'pair_id': row['pair_id'],
                'source_split': row['source_split'], 'error_type': row['error_type'],
                'is_control': row['is_control'], 'input_text': row['input_text'],
                'reference_text': row['reference_text'], 'output_text': output,
                'status': status, 'error': error, 'latency_seconds': round(duration, 6),
                'input_tokens': input_tokens, 'generated_tokens': tokens,
                'timestamp_utc': utc_now(),
            }
            # Persist the result immediately so prior examples survive an interruption.
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            new_status_counts[status] = new_status_counts.get(status, 0) + 1
            print(f'[{n}/{len(pending)}] {row["item_id"]}: {status} ({duration:.2f}s)', flush=True)
    print(f'Gemt: {args.output} | nye status: {new_status_counts}', flush=True)
    # Warn if any technical errors occurred; they must be investigated before resuming.
    if new_status_counts.get('error', 0):
        print('ADVARSEL: Mindst én teknisk genereringsfejl blev registreret. '
              'Undersøg status=error i JSONL før ny genstart.', file=sys.stderr, flush=True)
    return 0


# Run main() only when this file is executed as a script.
# Convert expected validation/runtime problems into a readable error and exit code 1.
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f'FEJL: {exc}', file=sys.stderr)
        raise SystemExit(1)
