#!/usr/bin/env python3
"""Reconstruct Danish GEC minimal pairs from the published DaLA acceptability dataset.

The original DaLA creation script creates rows in this order:
    [correct_0, ..., correct_(n-1), incorrect_0, ..., incorrect_(n-1)]
and then executes:
    df.sample(frac=1.0, random_state=4242).reset_index(drop=True)

This script reverses that specific shuffle and checks the labels/categories before
writing any output. No new linguistic corruptions or reference corrections are created.

References:
  https://github.com/N-essuno/DaLA/blob/main/dala/create_dala.py
  https://huggingface.co/datasets/danish-foundation-models/dala
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_DATASET_ID = "danish-foundation-models/dala"
DEFAULT_SEED = 4242
REQUIRED_COLUMNS = {"text", "corruption_type", "label"}
SPLITS = ("train", "val", "test")


class ReconstructionError(ValueError):
    """Raised when the source is not compatible with DaLA's published shuffle."""


def read_source(dataset_id: str, revision: str | None, input_dir: Path | None) -> dict[str, pd.DataFrame]:
    """Load Hugging Face DaLA splits, or locally downloaded DaLA CSV splits."""
    data: dict[str, pd.DataFrame] = {}
    if input_dir is not None:
        for split in SPLITS:
            options = [input_dir / f"{split}.csv", input_dir / f"dala_da_{split}.csv"]
            path = next((p for p in options if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(
                    f"Mangler CSV til split '{split}'. Prøvede: "
                    + ", ".join(str(p) for p in options)
                )
            data[split] = pd.read_csv(path, encoding="utf-8-sig")
            print(f"  {split}: {len(data[split]):,} rækker fra {path}")
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Installer først afhængigheder: python -m pip install -r requirements.txt"
            ) from exc

        print(f"Henter/læser DaLA fra Hugging Face: {dataset_id}")
        args = {"revision": revision} if revision else {}
        dataset = load_dataset(dataset_id, **args)
        for split in SPLITS:
            actual = split if split in dataset else ("validation" if split == "val" and "validation" in dataset else None)
            if actual is None:
                raise ReconstructionError(f"Datasættet mangler split '{split}'. Fundne splits: {list(dataset.keys())}")
            data[split] = dataset[actual].to_pandas()
            print(f"  {split}: {len(data[split]):,} rækker")
    return data


def reconstruct_split(shuffled: pd.DataFrame, split: str, seed: int) -> pd.DataFrame:
    """Invert DaLA's pandas shuffle and return verified correct/incorrect pairs."""
    if not REQUIRED_COLUMNS.issubset(shuffled.columns):
        missing = sorted(REQUIRED_COLUMNS - set(shuffled.columns))
        raise ReconstructionError(f"{split}: mangler kolonner: {missing}")

    source = shuffled.loc[:, ["text", "corruption_type", "label"]].reset_index(drop=True)
    total = len(source)
    if total == 0 or total % 2 != 0:
        raise ReconstructionError(f"{split}: forventede et positivt, lige antal rækker; fik {total}")

    # The shuffle permutation maps displayed row -> index before shuffling.
    # By sorting displayed rows according to that original index, we undo it.
    shuffled_positions = (
        pd.DataFrame({"original_position": range(total)})
        .sample(frac=1.0, random_state=seed)["original_position"]
        .to_numpy()
    )
    restored = (
        source.assign(_original_position=shuffled_positions)
        .sort_values("_original_position", kind="stable")
        .drop(columns="_original_position")
        .reset_index(drop=True)
    )

    middle = total // 2
    correct = restored.iloc[:middle].reset_index(drop=True)
    incorrect = restored.iloc[middle:].reset_index(drop=True)

    correct_labels_ok = correct["label"].eq("correct")
    incorrect_labels_ok = incorrect["label"].eq("incorrect")
    correct_types_ok = correct["corruption_type"].isna() | correct["corruption_type"].eq("")
    incorrect_types_ok = incorrect["corruption_type"].notna() & incorrect["corruption_type"].ne("")

    checks = {
        "korrekte sætninger ligger i første halvdel": bool(correct_labels_ok.all()),
        "fejlbehæftede sætninger ligger i anden halvdel": bool(incorrect_labels_ok.all()),
        "kontrolsætninger har ingen fejlkategori": bool(correct_types_ok.all()),
        "fejlbehæftede sætninger har en fejlkategori": bool(incorrect_types_ok.all()),
    }
    if not all(checks.values()):
        reasons = "; ".join(name for name, passed in checks.items() if not passed)
        raise ReconstructionError(
            f"{split}: Rekonstruktionen kunne ikke valideres ({reasons}). "
            "DaLA-versionen kan have ændret shuffle/format. "
            "Brug ikke output som referencepar uden at undersøge kilden."
        )

    originals = correct["text"]
    corruptions = incorrect["text"]
    valid_text = (
        originals.map(lambda x: isinstance(x, str) and bool(x.strip()))
        & corruptions.map(lambda x: isinstance(x, str) and bool(x.strip()))
    )
    if not bool(valid_text.all()):
        raise ReconstructionError(f"{split}: tomme eller ugyldige sætninger i rekonstruktionen")
    if bool(originals.eq(corruptions).any()):
        n_identical = int(originals.eq(corruptions).sum())
        raise ReconstructionError(
            f"{split}: {n_identical} par har identisk input og reference. "
            "Kontrollér datasættet/kilden, inden du fortsætter."
        )

    # A deterministic identifier for each original/corrupted pair.
    return pd.DataFrame(
        {
            "pair_id": [f"{split}-{i + 1:05d}" for i in range(middle)],
            "split": split,
            "error_type": incorrect["corruption_type"].astype(str),
            "incorrect_text": corruptions,
            "reference_text": originals,
        }
    )


def make_review_sample(pairs: pd.DataFrame, per_category: int, seed: int) -> pd.DataFrame:
    """Stratified review sample from train and val (never test by default)."""
    candidates = pairs[pairs["split"].isin(("train", "val"))]
    selections = []
    for _, group in candidates.groupby("error_type", sort=True):
        selections.append(group.sample(n=min(per_category, len(group)), random_state=seed))
    review = pd.concat(selections, ignore_index=True) if selections else candidates.head(0).copy()
    review = review.sort_values(["error_type", "split", "pair_id"]).reset_index(drop=True)
    for field in ("error_plausible", "reference_appropriate", "alternative_valid_correction", "notes"):
        review[field] = ""
    return review


def write_results(
    pairs: pd.DataFrame,
    out_dir: Path,
    dataset_id: str,
    revision: str | None,
    seed: int,
    review_per_category: int,
    source_mode: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "dala_gec_pairs.csv"
    controls_path = out_dir / "dala_controls.csv"
    full_path = out_dir / "dala_evaluation_inputs.csv"
    review_path = out_dir / "dala_manual_review.csv"
    report_path = out_dir / "dataset_report.json"

    # Preserve one control corresponding to each original sentence.
    controls = pairs.loc[:, ["pair_id", "split", "reference_text"]].copy()
    controls["input_text"] = controls["reference_text"]
    controls["error_type"] = "none"
    controls["is_control"] = True
    controls = controls.loc[:, ["pair_id", "split", "error_type", "input_text", "reference_text", "is_control"]]

    incorrect_rows = pairs.rename(columns={"incorrect_text": "input_text"}).copy()
    incorrect_rows["is_control"] = False
    incorrect_rows = incorrect_rows.loc[:, ["pair_id", "split", "error_type", "input_text", "reference_text", "is_control"]]
    all_inputs = pd.concat([incorrect_rows, controls], ignore_index=True)
    all_inputs = all_inputs.sort_values(["split", "pair_id", "is_control"], kind="stable").reset_index(drop=True)
    review = make_review_sample(pairs, review_per_category, seed)

    for data, filepath in (
        (pairs, pairs_path), (controls, controls_path),
        (all_inputs, full_path), (review, review_path),
    ):
        # utf-8-sig lets Excel recognize Danish characters without manual configuration.
        data.to_csv(filepath, index=False, encoding="utf-8-sig")

    count_by_split = {str(s): int(n) for s, n in pairs.groupby("split").size().items()}
    count_by_category = {str(c): int(n) for c, n in pairs.groupby("error_type").size().items()}
    counts_split_category = {
        str(split): {str(c): int(n) for c, n in group.groupby("error_type").size().items()}
        for split, group in pairs.groupby("split")
    }
    original_duplicates = int(pairs.duplicated(subset=["reference_text"]).sum())
    duplicate_inputs = int(pairs.duplicated(subset=["incorrect_text"]).sum())
    split_sets = {split: set(g["reference_text"]) for split, g in pairs.groupby("split")}
    overlap = {}
    names = sorted(split_sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap[f"{a}__{b}"] = len(split_sets[a].intersection(split_sets[b]))

    report = {
        "source_dataset": dataset_id,
        "source_mode": source_mode,
        "source_revision": revision or "not pinned; pin a revision for final reproducibility",
        "reconstruction_method": "undo pandas.DataFrame.sample(frac=1.0, random_state=4242); pair first and second halves",
        "shuffle_seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "number_of_pairs": int(len(pairs)),
        "number_of_control_inputs": int(len(controls)),
        "number_of_evaluation_inputs": int(len(all_inputs)),
        "number_of_categories": len(count_by_category),
        "pairs_by_split": count_by_split,
        "pairs_by_category": count_by_category,
        "pairs_by_split_and_category": counts_split_category,
        "duplicate_reference_text_count": original_duplicates,
        "duplicate_incorrect_text_count": duplicate_inputs,
        "reference_text_overlap_between_splits": overlap,
        "manual_review_size": int(len(review)),
        "manual_review_splits": ["train", "val"],
        "validation": "Source shuffle/labels/categories/nonempty/different texts passed for all splits; linguistic validity NOT established.",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nValidering OK: Rekonstruerede DaLA-par")
    print(f"  Par: {len(pairs):,}; kontrollinjer: {len(controls):,}")
    print(f"  Fejlkategorier: {len(count_by_category)}")
    print(f"  Fordeling på splits: {count_by_split}")
    print(f"  Dobbeltforekomster blandt referencer: {original_duplicates}")
    print(f"  Overlap mellem splits: {overlap}")
    for path in (pairs_path, controls_path, full_path, review_path, report_path):
        print(f"  Gemte: {path}")
    print("\nOBS: De sproglige referencekorrektioner er endnu IKKE manuelt valideret.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forbered DaLA-sætningspar til dansk GEC-evaluering")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Hugging Face dataset ID")
    parser.add_argument("--revision", default=None, help="Hugging Face commit-hash for reproducérbarhed")
    parser.add_argument("--input-dir", type=Path, default=None, help="Alternativt: mappe med lokal train/val/test CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--review-per-category", type=int, default=5)
    args = parser.parse_args(argv)
    if args.review_per_category < 1:
        parser.error("--review-per-category skal være mindst 1")
    if args.shuffle_seed != DEFAULT_SEED:
        print("ADVARSEL: DaLAs dokumenterede shuffle bruger seed 4242.", file=sys.stderr)

    try:
        raw = read_source(args.dataset_id, args.revision, args.input_dir)
        pairs_by_split = [reconstruct_split(raw[split], split, args.shuffle_seed) for split in SPLITS]
        pairs = pd.concat(pairs_by_split, ignore_index=True)
        write_results(
            pairs, args.output_dir, args.dataset_id, args.revision, args.shuffle_seed,
            args.review_per_category, "local_csv" if args.input_dir else "hugging_face",
        )
    except (ReconstructionError, FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        print(f"\nFEJL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
