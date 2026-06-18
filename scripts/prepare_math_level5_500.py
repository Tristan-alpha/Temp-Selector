#!/usr/bin/env python3
"""Rebuild the 500-problem Level-5 MATH input with provenance metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_SOURCE = "/home/data/dazhou/ReasonEval/dataset/math-5.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"problem", "answer", "unique_id"} - row.keys()
            if missing:
                raise ValueError(f"line {line_number}: missing fields {sorted(missing)}")
            if int(row.get("level", -1)) != 5:
                raise ValueError(f"line {line_number}: expected level 5, got {row.get('level')}")
            rows.append(row)
    return rows


def prepare(source: Path, output: Path, manifest: Path, count: int) -> Dict[str, Any]:
    rows = load_jsonl(source)
    if len(rows) < count:
        raise ValueError(f"source has {len(rows)} rows, fewer than requested {count}")

    # The historical checkout contains no raw-data sampling script. Its small
    # dataset naming convention is reconstructed as an order-preserving prefix.
    selected = rows[:count]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_ids = [str(row["unique_id"]) for row in selected]
    metadata = {
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "source_rows": len(rows),
        "selection": "first_n_in_source_order",
        "count": count,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "first_unique_id": selected_ids[0],
        "last_unique_id": selected_ids[-1],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default="data/math_5_small_500.jsonl")
    parser.add_argument("--manifest", default="data/math_5_small_500.manifest.json")
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    metadata = prepare(Path(args.source), Path(args.output), Path(args.manifest), args.count)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
