#!/usr/bin/env python3
"""Pre-register DaLA analysis choices and verify the finalized dataset locally.

Pure standard-library Python. No AI Lab access or network traffic.

This script NEVER edits model inputs, references, the source manifest, or audit
annotations. It creates deterministic analysis ID files and a protocol lock.
It does NOT claim that unresolved benchmark issues have been adjudicated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REVISION = "0cc96091dc265d14fb5d507114690fd85857b6da"
COUNTS = {"train_errors": 512, "validation_errors": 128,
          "test_errors": 1024, "test_controls": 1024, "test_audit_sample": 48}
CORE_FILES = ["train.csv", "validation.csv", "test_errors.csv", "test_controls.csv",
              "test_audit_sample.csv", "test_audit_annotated.csv"]
BASE_FIELDS = ["item_id", "pair_id", "split", "error_type", "is_control", "input_text", "reference_text"]
MANUAL_SEED = 20261105
MANUAL_ERROR_PER_CATEGORY = 5
MANUAL_CONTROL_PER_CATEGORY = 2


class ProtocolError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(128 * 1024), b""):
            hasher.update(b)
    return hasher.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ProtocolError(message)


def read_csv(path: Path, required: list[str]) -> list[dict[str, str]]:
    require(path.is_file(), f"Mangler fil: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None and set(required).issubset(reader.fieldnames),
                f"Forkerte kolonner i {path}: {reader.fieldnames}")
        rows = list(reader)
    require(bool(rows), f"Tom CSV: {path}")
    return rows


def render_csv(rows: list[dict[str, str]], columns: list[str]) -> bytes:
    f = io.StringIO(newline="")
    writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return f.getvalue().encode("utf-8")


def verify_hashes(root: Path, manifest: dict) -> None:
    out = manifest.get("output_sha256", {})
    inputs = manifest.get("input_sha256", {})
    require(all(k in out for k in CORE_FILES), "Manifest mangler output-hashes")
    for filename, expected in out.items():
        path = root / "data" / "final" / filename
        require(path.is_file() and sha256_file(path) == expected,
                f"Outputfil ændret/mangler i forhold til manifest: {path}")
    for filename, expected in inputs.items():
        path = root / filename if filename.startswith("reviews/") else root / "data" / "processed" / filename
        require(path.is_file() and sha256_file(path) == expected,
                f"Inputfil ændret/mangler i forhold til manifest: {path}")


def verify_data(root: Path, manifest: dict) -> tuple[list[dict[str, str]], list[str]]:
    require(manifest.get("source_revision_pinned") is True
            and manifest.get("source_revision") == REVISION,
            "DaLA-kilderevision er ikke den aftalte fastlåste version")
    require(all(manifest.get("counts", {}).get(k) == v for k, v in COUNTS.items()),
            "Manifestets antal afviger fra aftalt datadesign")
    require(len(manifest.get("categories_by_test", {})) == 16,
            "Test skal have 16 fejlkategorier")
    audit = manifest.get("test_audit", {})
    require(audit.get("screening_completed") is True
            and audit.get("validated_against_exact_sample") is True
            and audit.get("used_for_prompt_development") is False,
            "Test-audit mangler eller er brugt til promptudvikling")
    require(audit.get("reviewed_test_pairs") == 48
            and audit.get("follow_up_count") == 11,
            "Audit har ændret sig; revider protokollen, før der låses")
    require(manifest.get("review", {}).get("unresolved_review_flags") == 17,
            "Train/val-forbehold har ændret sig; revider protokollen")
    require(audit.get("excluded_test_pairs") == 0,
            "Primær analyse skal indeholde alle testpar")
    verify_hashes(root, manifest)

    final = root / "data" / "final"
    train = read_csv(final / "train.csv", BASE_FIELDS)
    val = read_csv(final / "validation.csv", BASE_FIELDS)
    errors = read_csv(final / "test_errors.csv", BASE_FIELDS)
    controls = read_csv(final / "test_controls.csv", BASE_FIELDS)
    require((len(train), len(val), len(errors), len(controls)) == (512, 128, 1024, 1024),
            "Forkert antal i evalueringsfilerne")
    for rows, expected_split in ((train, "train"), (val, "val"), (errors, "test")):
        require(all(r["split"] == expected_split and r["is_control"].lower() == "false"
                    and r["item_id"] == "err:" + r["pair_id"] for r in rows),
                f"Ugyldige error-rækker i {expected_split}")
        require(len({r["pair_id"] for r in rows}) == len(rows),
                f"Dublerede par-ID'er i {expected_split}")
    error_ids = {r["pair_id"] for r in errors}
    require(not (error_ids & {r["pair_id"] for r in train + val}),
            "Lækage mellem test og train/val")
    require(not ({r["pair_id"] for r in train} & {r["pair_id"] for r in val}),
            "Lækage mellem train og val")
    errors_by_id = {r["pair_id"]: r for r in errors}
    control_by_id = {r["pair_id"]: r for r in controls}
    require(len(control_by_id) == 1024 and set(control_by_id) == error_ids,
            "Der skal være præcis én kontrol pr. testpar")
    for pid, c in control_by_id.items():
        e = errors_by_id[pid]
        require(c["item_id"] == "ctrl:" + pid and c["split"] == "test"
                and c["is_control"].lower() == "true" and c["error_type"] == "none"
                and c["input_text"] == c["reference_text"] == e["reference_text"],
                f"Ugyldig kontrolsætning: {pid}")
    category_counts = dict(Counter(e["error_type"] for e in errors))
    require(category_counts == manifest["categories_by_test"],
            "Fejlkategorier matcher ikke manifest")
    flags = sorted(item["pair_id"] for item in audit.get("follow_up_pairs", []))
    require(len(flags) == 11 and len(set(flags)) == 11 and set(flags) <= error_ids,
            "Audit-forbehold har ukendte eller dublerede ID'er")
    require(len(error_ids.difference(flags)) == 1013,
            "Forventede 1.013 observationer i sensitivitetsudsnittet")
    return errors, flags


def make_analysis_sets(errors: list[dict[str, str]], flags: list[str]) -> dict[str, bytes]:
    flagged = set(flags)
    sensitivity = [{k: r[k] for k in ("item_id", "pair_id", "error_type")}
                   for r in errors if r["pair_id"] not in flagged]
    sensitivity.sort(key=lambda r: r["pair_id"])
    groups: dict[str, list[str]] = defaultdict(list)
    for r in errors:
        groups[r["error_type"]].append(r["pair_id"])
    rng = random.Random(MANUAL_SEED)
    manual = []
    for category in sorted(groups):
        pids = rng.sample(sorted(groups[category]), MANUAL_ERROR_PER_CATEGORY)
        for i, pid in enumerate(pids):
            manual.append({"pair_id": pid, "error_type": category, "review_error": "true",
                           "review_control": "true" if i < MANUAL_CONTROL_PER_CATEGORY else "false"})
    return {
        "sensitivity_unflagged_test_ids.csv": render_csv(
            sensitivity, ["item_id", "pair_id", "error_type"]),
        "manual_review_pair_ids.csv": render_csv(
            manual, ["pair_id", "error_type", "review_error", "review_control"]),
    }


def make_lock(root: Path) -> tuple[dict, dict[str, bytes]]:
    manifest_path = root / "data" / "final" / "dataset_manifest.json"
    protocol_path = root / "EVALUATION_PROTOCOL.md"
    require(manifest_path.is_file() and protocol_path.is_file(),
            "Mangler data/final/dataset_manifest.json eller EVALUATION_PROTOCOL.md")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    errors, flags = verify_data(root, manifest)
    artifacts = make_analysis_sets(errors, flags)
    lock = {
        "protocol_version": "1.0.0",
        "status": "analysis_rules_preregistered_dataset_identity_locked_expert_review_pending",
        "model_and_inference_configuration": "pending_before_first_main_test_run",
        "source_dataset": manifest["dataset"],
        "source_revision": manifest["source_revision"],
        "manifest_sha256": sha256_file(manifest_path),
        "evaluation_protocol_sha256": sha256_file(protocol_path),
        "dataset_output_sha256": manifest["output_sha256"],
        "audit_screening": {
            "independently_adjudicated": manifest["test_audit"]["independently_adjudicated"],
            "flagged_train_val": manifest["review"]["unresolved_review_flags"],
            "flagged_test": len(flags), "flagged_test_ids": flags,
        },
        "analysis_sets": {
            "primary_error_test_n": 1024, "control_test_n": 1024,
            "sensitivity_excluding_all_11_audit_flags_n": 1013,
            "sensitivity_exclusion_reason": "pre-identified audit flags, not confirmed invalid data",
            "manual_error_pairs_per_model": 80, "manual_control_pairs_per_model": 32,
            "manual_pair_seed": MANUAL_SEED,
        },
        "analysis_files_sha256": {name: hashlib.sha256(data).hexdigest()
                                  for name, data in sorted(artifacts.items())},
    }
    return lock, artifacts


def write_if_unmodified(path: Path, data: bytes, force: bool) -> None:
    if path.exists():
        existing = path.read_bytes()
        if existing == data:
            return
        require(force, f"{path} eksisterer og afviger; gennemgå ændringer og brug --force, hvis bevidst")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fastlås DaLA's analysesæt og regler; ingen modelkald")
    parser.add_argument("--root", type=Path, default=Path("."), help="Projektets rodmappe")
    parser.add_argument("--force", action="store_true", help="Tillad bevidst revision af eksisterende lock")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        lock, artifacts = make_lock(root)
        out_dir = root / "data" / "analysis_sets"
        lock_target = root / "data" / "final" / "evaluation_protocol.lock.json"
        lock_bytes = (json.dumps(lock, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        # Refuse changed existing outputs *before* writing anything.
        for path, content in [(out_dir / name, data) for name, data in artifacts.items()] + [(lock_target, lock_bytes)]:
            if path.exists() and path.read_bytes() != content and not args.force:
                raise ProtocolError(f"Eksisterende fil er anderledes: {path}. Brug kun --force efter gennemsyn.")
        for name, data in artifacts.items():
            write_if_unmodified(out_dir / name, data, args.force)
        write_if_unmodified(lock_target, lock_bytes, args.force)
        print("OK: Datasæthashes og analyseprotokol verificeret.")
        print("  Primær GEC: 1.024 testpar; kontrol: 1.024 sætninger")
        print("  Sensitivitet: 1.013 testpar (11 audit-flagged holdt udenfor, ingen fjernet fra hovedanalysen)")
        print("  Manuel review-plan: 80 fejlpar + 32 kontroller pr. model")
        print("  Lås:", lock_target)
        print("  OBS: AI-assisteret audit er ikke uafhængigt valideret; model- og promptkonfiguration mangler.")
        return 0
    except (ProtocolError, OSError, json.JSONDecodeError) as exc:
        print("FEJL:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
