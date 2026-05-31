from __future__ import annotations

import argparse
import re

from utils import load_config, read_jsonl, write_jsonl

HARM_TERMS = [
    "kill", "murder", "bomb", "poison", "hitman", "enslave", "violence", "shoot"
]

NUMBERS_RE = re.compile(r"^[\s\d,;()\[\].-]+$")


def contains_explicit_harm(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in HARM_TERMS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = cfg["generation"]["output_raw"]
    out_path = cfg["filtering"]["output_filtered"]
    rows = read_jsonl(raw_path)

    kept = []
    for row in rows:
        completion = row["completion"]

        if cfg["filtering"]["remove_explicit_harm"] and contains_explicit_harm(completion):
            continue

        if cfg["filtering"]["numbers_only"] and not NUMBERS_RE.fullmatch(completion.strip()):
            continue

        kept.append(row)

    write_jsonl(out_path, kept)
    print(f"Kept {len(kept)} / {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
