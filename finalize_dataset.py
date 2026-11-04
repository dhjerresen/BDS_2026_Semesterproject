#!/usr/bin/env python3
"""Prepare reproducible, *candidate* experiment splits from reconstructed DaLA pairs.

Run after prepare_dataset.py. This script does not call AI Lab, alter source
sentences, exclude difficult cases, or use test data to develop prompts.

Input: data/processed/dala_gec_pairs.csv, dala_controls.csv,
       dataset_report.json, and (when available) dala_evaluation_inputs.csv.
Optional: annotated train/val review (reviews/dala_manual_review_annotated.csv).
          annotated test audit (reviews/dala_test_audit_annotated.csv).
Output: data/final/train.csv, validation.csv, test_errors.csv,
        test_controls.csv, test_audit_sample.csv, dataset_manifest.json,
        and (when available) a byte-identical audit annotation snapshot.

The outputs remain *candidate* until review issues have been adjudicated.
Completing an AI-assisted quality screening is NOT the same as an expert
validation or a license to use held-out test examples for prompt development.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SPLITS = ("train", "val", "test")
PAIR_FIELDS = ("pair_id", "split", "error_type", "incorrect_text", "reference_text")
CONTROL_FIELDS = ("pair_id", "split", "error_type", "input_text", "reference_text", "is_control")
OUTPUT_FIELDS = ("item_id", "pair_id", "split", "error_type", "is_control", "input_text", "reference_text")
AUDIT_FIELDS = ("pair_id", "split", "error_type", "incorrect_text", "reference_text",
                "error_plausible", "reference_appropriate", "alternative_valid_correction", "notes")
ANNOTATED_AUDIT_FIELDS = AUDIT_FIELDS + ("issue_type", "needs_follow_up")
AUDIT_SNAPSHOT = "test_audit_annotated.csv"


class DatasetValidationError(ValueError):
    """Inputs or annotation failed a reproducibility/integrity check."""


def load_csv(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise DatasetValidationError(f"Mangler inputfil: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None or not set(expected).issubset(reader.fieldnames):
            raise DatasetValidationError(f"{path}: forventede kolonner {list(expected)}, fik {reader.fieldnames}")
        data = list(reader)
    if not data:
        raise DatasetValidationError(f"{path}: CSV-filen er tom")
    return data


def output_csv(path: Path, records: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(pairs: list[dict[str, str]], controls: list[dict[str, str]],
                  report: dict, evaluation: list[dict[str, str]] | None) -> dict:
    ids = set()
    counts = Counter()
    categories = Counter()
    refs_by_split = defaultdict(set)
    original_refs = set()
    original_incorrect = set()
    by_id = {}
    for row in pairs:
        pid, split, category = (row[col] for col in ("pair_id", "split", "error_type"))
        if pid in ids or split not in SPLITS or not pid.startswith(split + "-"):
            raise DatasetValidationError(f"Ugyldigt eller dubleret par-ID/split: {pid!r} / {split!r}")
        if not category.strip() or not row["incorrect_text"].strip() or not row["reference_text"].strip():
            raise DatasetValidationError(f"Tomt felt for {pid}")
        if row["incorrect_text"] == row["reference_text"]:
            raise DatasetValidationError(f"Input er lig reference for {pid}")
        if row["incorrect_text"] in original_incorrect or row["reference_text"] in original_refs:
            raise DatasetValidationError(f"Dubleret input- eller referencetekst: {pid}")
        ids.add(pid)
        by_id[pid] = row
        counts[split] += 1
        categories[category] += 1
        refs_by_split[split].add(row["reference_text"])
        original_refs.add(row["reference_text"])
        original_incorrect.add(row["incorrect_text"])

    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if refs_by_split[a] & refs_by_split[b]:
                raise DatasetValidationError(f"Referencer overlapper mellem {a} og {b}")
    if set(counts) != set(SPLITS):
        raise DatasetValidationError(f"Mangler split: {set(SPLITS) - set(counts)}")
    expected_counts = report.get("pairs_by_split")
    if expected_counts and dict(counts) != {s: int(n) for s, n in expected_counts.items()}:
        raise DatasetValidationError(f"Uoverensstemmelse mellem rapport og par: {dict(counts)} != {expected_counts}")
    expected_categories = report.get("pairs_by_category")
    if expected_categories and dict(categories) != {s: int(n) for s, n in expected_categories.items()}:
        raise DatasetValidationError("Fejlkategorier afviger fra dataset_report.json")
    if report.get("number_of_pairs") is not None and len(pairs) != int(report["number_of_pairs"]):
        raise DatasetValidationError("Antal par afviger fra dataset_report.json")

    control_by_id = {}
    for row in controls:
        pid = row["pair_id"]
        if pid not in by_id or pid in control_by_id:
            raise DatasetValidationError(f"Ukendt eller dubleret kontrol-ID: {pid}")
        source = by_id[pid]
        if not (row["split"] == source["split"] and
                row["reference_text"] == source["reference_text"] and
                row["input_text"] == source["reference_text"] and
                row["error_type"] == "none" and
                row["is_control"].strip().lower() == "true"):
            raise DatasetValidationError(f"Kontrolsætning stemmer ikke overens med par {pid}")
        control_by_id[pid] = row
    if len(control_by_id) != len(by_id):
        raise DatasetValidationError("Ikke præcis én kontrolsætning pr. par")

    if evaluation is not None:
        seen = set()
        for row in evaluation:
            pid = row["pair_id"]
            is_control = row["is_control"].strip().lower()
            if pid not in by_id or is_control not in {"true", "false"}:
                raise DatasetValidationError(f"Ugyldigt evalueringsinput: {pid}")
            key = (pid, is_control)
            if key in seen:
                raise DatasetValidationError(f"Dubleret evalueringsinput: {key}")
            seen.add(key)
            source = by_id[pid]
            expected_text = source["reference_text"] if is_control == "true" else source["incorrect_text"]
            expected_type = "none" if is_control == "true" else source["error_type"]
            if (row["input_text"] != expected_text or row["reference_text"] != source["reference_text"]
                    or row["error_type"] != expected_type or row["split"] != source["split"]):
                raise DatasetValidationError(f"Evalueringsinput matcher ikke par {pid}")
        if len(seen) != 2 * len(pairs):
            raise DatasetValidationError("Evalueringsinputs indeholder ikke præcis fejl- og kontrolvariant af alle par")
    return {"pairs_by_split": {s: counts[s] for s in SPLITS},
            "pairs_by_category": dict(sorted(categories.items())),
            "n_categories": len(categories), "by_id": by_id}


def verify_review(review: list[dict[str, str]], by_id: dict[str, dict[str, str]]) -> dict:
    review_ids = set()
    flagged_ids = []
    for row in review:
        pid = row["pair_id"]
        if pid in review_ids or pid not in by_id:
            raise DatasetValidationError(f"Ukendt eller dubleret review-ID: {pid}")
        src = by_id[pid]
        if src["split"] == "test":
            raise DatasetValidationError(f"Reviewfilen indeholder testpar: {pid}")
        for col in PAIR_FIELDS:
            if row[col] != src[col]:
                raise DatasetValidationError(f"Reviewafvigelse for {pid}, felt {col}")
        review_ids.add(pid)
        if row.get("needs_follow_up", "").strip().lower() in {"yes", "true", "1"}:
            flagged_ids.append(pid)
    return {"reviewed_train_val_pairs": len(review_ids),
            "unresolved_review_flags": len(flagged_ids),
            "flagged_pair_ids": sorted(flagged_ids)}


def verify_test_audit(annotated: list[dict[str, str]],
                      expected_sample: list[dict[str, str]]) -> dict:
    """Accept ONLY a complete, annotated copy of the exact deterministic test sample.

    This validates data provenance and the presence/shape of the annotations.
    It does NOT independently validate any linguistic judgement or resolve flags.
    """
    expected = {row["pair_id"]: row for row in expected_sample}
    if len(annotated) != len(expected) or len(expected) != len(expected_sample):
        raise DatasetValidationError(
            f"Test-audit: forventede præcis {len(expected_sample)} unikke rækker; "
            f"fik {len(annotated)}"
        )

    seen = set()
    plausibility = Counter()
    reference = Counter()
    alternatives = Counter()
    issue_types = Counter()
    follow_ups = []
    for row in annotated:
        pid = row["pair_id"]
        if pid in seen or pid not in expected:
            raise DatasetValidationError(f"Test-audit: ukendt/dubleret ID eller ikke udtrukket testpar: {pid}")
        seen.add(pid)
        for field in PAIR_FIELDS:
            if row[field] != expected[pid][field]:
                raise DatasetValidationError(f"Test-audit: {pid} afviger fra udtrukket test-sample i {field}")

        values = {
            "error_plausible": {"yes", "no", "uncertain"},
            "reference_appropriate": {"yes", "qualified", "no"},
            "alternative_valid_correction": {"yes", "no", "uncertain"},
            "needs_follow_up": {"yes", "no"},
        }
        for field, allowed in values.items():
            if row[field].strip() not in allowed:
                raise DatasetValidationError(
                    f"Test-audit: {pid}: felt {field} mangler eller har ugyldig værdi {row[field]!r}"
                )
        if not row["issue_type"].strip():
            raise DatasetValidationError(f"Test-audit: {pid}: issue_type er tomt")
        plausibility[row["error_plausible"]] += 1
        reference[row["reference_appropriate"]] += 1
        alternatives[row["alternative_valid_correction"]] += 1
        issue_types[row["issue_type"]] += 1
        if row["needs_follow_up"] == "yes":
            follow_ups.append({"pair_id": pid, "error_type": row["error_type"],
                               "issue_type": row["issue_type"]})

    if seen != set(expected):
        raise DatasetValidationError("Test-audit: stikprøven er ikke identisk med den deterministiske udtrækning")
    return {
        "screening_completed": True,
        "screening_type": "preliminary_single_AI_assistant",
        "independently_adjudicated": False,
        "validated_against_exact_sample": True,
        "reviewed_test_pairs": len(annotated),
        "error_plausibility_counts": dict(sorted(plausibility.items())),
        "reference_assessment_counts": dict(sorted(reference.items())),
        "alternative_correction_counts": dict(sorted(alternatives.items())),
        "issue_type_counts": dict(sorted(issue_types.items())),
        "follow_up_count": len(follow_ups),
        "follow_up_pairs": sorted(follow_ups, key=lambda x: x["pair_id"]),
        "used_for_prompt_development": False,
        "excluded_test_pairs": 0,
    }


def make_error_input(row: dict[str, str]) -> dict[str, str]:
    return {"item_id": "err:" + row["pair_id"], "pair_id": row["pair_id"],
            "split": row["split"], "error_type": row["error_type"],
            "is_control": "false", "input_text": row["incorrect_text"],
            "reference_text": row["reference_text"]}


def make_control_input(row: dict[str, str]) -> dict[str, str]:
    return {"item_id": "ctrl:" + row["pair_id"], "pair_id": row["pair_id"],
            "split": row["split"], "error_type": "none", "is_control": "true",
            "input_text": row["reference_text"], "reference_text": row["reference_text"]}


def make_test_audit(rows: list[dict[str, str]], n_per_category: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row["error_type"]].append(row)
    selected = []
    for category in sorted(groups):
        group = sorted(groups[category], key=lambda x: x["pair_id"])
        selected.extend(rng.sample(group, k=min(len(group), n_per_category)))
    selected.sort(key=lambda x: (x["error_type"], x["pair_id"]))
    return [{**{col: row[col] for col in PAIR_FIELDS},
             "error_plausible": "", "reference_appropriate": "",
             "alternative_valid_correction": "", "notes": ""} for row in selected]


def finalize(input_dir: Path, output_dir: Path, review_path: Path | None,
             audit_per_category: int, seed: int,
             test_audit_path: Path | None = None) -> dict:
    if input_dir.resolve() == output_dir.resolve():
        raise DatasetValidationError("Outputmappe må ikke være den samme som inputmappen")
    paths = {
        "pairs": input_dir / "dala_gec_pairs.csv",
        "controls": input_dir / "dala_controls.csv",
        "report": input_dir / "dataset_report.json",
        "evaluation_inputs": input_dir / "dala_evaluation_inputs.csv",
    }
    pairs = load_csv(paths["pairs"], PAIR_FIELDS)
    controls = load_csv(paths["controls"], CONTROL_FIELDS)
    if not paths["report"].is_file():
        raise DatasetValidationError(f"Mangler {paths['report']}")
    report = json.loads(paths["report"].read_text(encoding="utf-8-sig"))
    evaluation = (load_csv(paths["evaluation_inputs"], CONTROL_FIELDS)
                  if paths["evaluation_inputs"].exists() else None)
    checked = verify_source(pairs, controls, report, evaluation)

    review_stats = {"reviewed_train_val_pairs": 0, "unresolved_review_flags": None,
                    "flagged_pair_ids": []}
    if review_path is not None:
        review = load_csv(review_path, PAIR_FIELDS)
        review_stats = verify_review(review, checked["by_id"])

    rows = {s: sorted([r for r in pairs if r["split"] == s], key=lambda x: x["pair_id"]) for s in SPLITS}
    outputs = {
        "train.csv": ([make_error_input(r) for r in rows["train"]], OUTPUT_FIELDS),
        "validation.csv": ([make_error_input(r) for r in rows["val"]], OUTPUT_FIELDS),
        "test_errors.csv": ([make_error_input(r) for r in rows["test"]], OUTPUT_FIELDS),
        "test_controls.csv": ([make_control_input(r) for r in rows["test"]], OUTPUT_FIELDS),
        "test_audit_sample.csv": (make_test_audit(rows["test"], audit_per_category, seed), AUDIT_FIELDS),
    }
    # Review the original sampled test texts and labels before writing output.
    # Annotations are NEVER allowed to modify the model inputs or references.
    test_audit_stats = {
        "screening_completed": False,
        "screening_type": None,
        "independently_adjudicated": False,
        "reviewed_test_pairs": 0,
        "follow_up_count": None,
        "follow_up_pairs": [],
        "used_for_prompt_development": False,
        "excluded_test_pairs": 0,
    }
    if test_audit_path is not None:
        annotated = load_csv(test_audit_path, ANNOTATED_AUDIT_FIELDS)
        test_audit_stats = verify_test_audit(annotated, outputs["test_audit_sample.csv"][0])
        test_audit_stats["source_file"] = str(test_audit_path)
        test_audit_stats["source_sha256"] = sha256(test_audit_path)
        test_audit_stats["documentation_snapshot"] = AUDIT_SNAPSHOT
        snapshot = output_dir / AUDIT_SNAPSHOT
        if test_audit_path.resolve() == snapshot.resolve():
            raise DatasetValidationError(
                "Audit-kildefilen må ikke være den samme som dokumentationskopien i data/final"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    # Inspect existing manual annotations *before* writing anything.
    audit_target = output_dir / "test_audit_sample.csv"
    if audit_target.exists():
        existing = load_csv(audit_target, AUDIT_FIELDS)
        if any(any(row.get(field, "").strip() for field in
                   ("error_plausible", "reference_appropriate", "alternative_valid_correction", "notes"))
               for row in existing):
            raise DatasetValidationError(
                f"{audit_target} indeholder allerede manuelle annotationer. "
                "Flyt eller omdøb den før genkørsel for ikke at miste arbejdet."
            )
    for name, (data, fields) in outputs.items():
        output_csv(output_dir / name, data, fields)
    if test_audit_path is not None:
        # Exact byte copy, rather than rewriting text / annotations or applying
        # any cleaning rules: the SHA-256 stays independently verifiable.
        shutil.copyfile(test_audit_path, output_dir / AUDIT_SNAPSHOT)

    source_revision = report.get("source_revision")
    is_pinned = bool(source_revision and re.fullmatch(r"[0-9a-fA-F]{40,64}", str(source_revision)))
    issues = []
    if not is_pinned:
        issues.append("DaLA-kilderevision er ikke fastlåst. Genkør prepare_dataset.py med --revision COMMIT_SHA.")
    if review_path is None:
        issues.append("Indledende train/val-annotationsfil blev ikke indlæst.")
    elif review_stats["unresolved_review_flags"]:
        issues.append(f"{review_stats['unresolved_review_flags']} train/val-observationer er stadig markeret til opfølgning.")
    if not test_audit_stats["screening_completed"]:
        issues.append("Test-audit er genereret, men endnu ikke screenet; må ikke bruges til promptudvikling.")
    elif test_audit_stats["follow_up_count"]:
        issues.append(
            f"Test-audit er screenet; {test_audit_stats['follow_up_count']} testpar er stadig "
            "markeret til opfølgning/adjudikation."
        )
    if test_audit_stats["screening_completed"] and not test_audit_stats["independently_adjudicated"]:
        issues.append("Test-audit er AI-assisteret screening, ikke uafhængig sproglig validering.")

    sources = {str(path.relative_to(input_dir)): sha256(path) for key, path in paths.items() if path.is_file()}
    if review_path is not None:
        sources[str(review_path)] = sha256(review_path)
    if test_audit_path is not None:
        sources[str(test_audit_path)] = sha256(test_audit_path)
    output_hashes = {name: sha256(output_dir / name) for name in outputs}
    if test_audit_path is not None:
        output_hashes[AUDIT_SNAPSHOT] = sha256(output_dir / AUDIT_SNAPSHOT)
        if output_hashes[AUDIT_SNAPSHOT] != sources[str(test_audit_path)]:
            raise DatasetValidationError("Auditens dokumentationskopi matcher ikke kilden (SHA-256)")
    manifest = {
        "dataset": report.get("source_dataset", "unknown"),
        "source_revision": source_revision,
        "source_revision_pinned": is_pinned,
        "status": "candidate_not_yet_fully_frozen",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "Keep all original reconstructed pairs in their original splits; do not remove flagged pairs automatically",
        "counts": {"train_errors": len(rows["train"]), "validation_errors": len(rows["val"]),
                   "test_errors": len(rows["test"]), "test_controls": len(rows["test"]),
                   "test_audit_sample": len(outputs["test_audit_sample.csv"][0])},
        "categories": checked["pairs_by_category"],
        "categories_by_test": dict(sorted(Counter(x["error_type"] for x in rows["test"]).items())),
        "audit_sampling": {"source": "test", "per_category": audit_per_category,
                           "random_seed": seed, "for_prompt_development": False},
        "review": review_stats,
        "test_audit": test_audit_stats,
        "input_sha256": sources,
        "output_sha256": output_hashes,
        "outstanding_issues": issues,
        "interpretation": "Reference exact match is not identical to linguistic correctness; review residual errors and underdetermined deletions",
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lav kandidatsplits fra rekonstrueret DaLA")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/final"))
    parser.add_argument("--review-file", type=Path, default=Path("reviews/dala_manual_review_annotated.csv"),
                        help="Annoteret train/val-sample; udelad med --no-review")
    parser.add_argument("--no-review", action="store_true", help="Spring indlæsning af annoteret review over")
    parser.add_argument("--test-audit-file", type=Path,
                        default=Path("reviews/dala_test_audit_annotated.csv"),
                        help="Annoteret test-audit; valideres mod de 48 udvalgte testpar")
    parser.add_argument("--no-test-audit", action="store_true",
                        help="Kør uden annoteret test-audit (status bliver uafsluttet)")
    parser.add_argument("--audit-per-category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args(argv)
    if args.audit_per_category < 1:
        parser.error("--audit-per-category skal være mindst 1")
    review = None if args.no_review else args.review_file
    if review is not None and not review.exists():
        # A missing review must be explicit; avoid silently ignoring it.
        parser.error(f"Reviewfil mangler: {review}; giv --review-file PATH eller --no-review")
    test_audit = None if args.no_test_audit else args.test_audit_file
    if test_audit is not None and not test_audit.exists():
        parser.error(f"Test-audit mangler: {test_audit}; giv --test-audit-file PATH eller --no-test-audit")
    try:
        manifest = finalize(args.input_dir, args.output_dir, review,
                            args.audit_per_category, args.seed, test_audit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FEJL: {exc}", file=sys.stderr)
        return 1
    print("Validering gennemført. Kandidatfiler gemt i", args.output_dir)
    for filename, count_key in (("train.csv", "train_errors"),
                               ("validation.csv", "validation_errors"),
                               ("test_errors.csv", "test_errors"),
                               ("test_controls.csv", "test_controls"),
                               ("test_audit_sample.csv", "test_audit_sample")):
        print(f"  {filename}: {manifest['counts'][count_key]} rækker")
    audit = manifest["test_audit"]
    if audit["screening_completed"]:
        print(f"  test-audit: {audit['reviewed_test_pairs']} screenet, "
              f"{audit['follow_up_count']} markeret til opfølgning")
        print(f"  {AUDIT_SNAPSHOT}: identisk dokumentationskopi, SHA-256 registreret")
    print("  dataset_manifest.json: hashes, kildemetadata og åbne metodepunkter")
    for issue in manifest["outstanding_issues"]:
        print("  OBS:", issue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
