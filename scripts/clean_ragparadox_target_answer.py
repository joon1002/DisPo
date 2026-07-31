#!/usr/bin/env python3
"""
Cleans the target_answer column of RAGParadox attack-document CSVs.

RAGParadox's source data often stores target_answer as a raw Python
list/quote repr string (e.g. "['Two years']", "'Harry Nilsson'",
'"Fat Lady"'). main_dipoison_fullcorpus_ragdef.py's ASR check only tests
plain substring containment between target_answer and the generated
response (just .strip(), no further processing), so if this wrapping is
left in place, the match always fails even when the attack actually
succeeded, making ASR read lower than it really is.

This script strips that wrapping down to plain text. The original file
is preserved as <file>.raw_backup before any value is changed (not
overwritten if it already exists).

Usage:
  python3 clean_ragparadox_target_answer.py <csv1> [csv2 ...]
  python3 clean_ragparadox_target_answer.py --glob "/path/to/RAGParadox/ragparadox_*.csv"
"""
import argparse
import ast
import glob
import shutil

import pandas as pd


def clean_target(v) -> str:
    s = str(v).strip()
    if not s or s.lower() == "nan" or s[0] not in "[\"'":
        return s
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return str(parsed[0]).strip() if parsed else s
        if isinstance(parsed, str):
            return parsed.strip()
        return s
    except Exception:
        pass
    # Fallback for when literal_eval fails (e.g. broken syntax from an
    # unescaped apostrophe inside, like "'Potomac's Division'"): manually
    # strip just the outer brackets/quotes instead.
    if s[0] == "[" and s[-1] == "]":
        inner = s[1:-1].strip()
        if len(inner) >= 2 and inner[0] in "'\"" and inner[-1] == inner[0]:
            inner = inner[1:-1]
        return inner.strip()
    if s[0] in "'\"" and s[-1] == s[0] and len(s) >= 2:
        return s[1:-1].strip()
    return s


def clean_file(path: str) -> None:
    backup = path + ".raw_backup"
    if not glob.glob(backup):
        shutil.copy2(path, backup)

    df = pd.read_csv(path)
    if "target_answer" not in df.columns:
        print(f"[skip] {path}: no target_answer column")
        return

    before = df["target_answer"].astype(str)
    after = before.apply(clean_target)
    changed = before != after
    df["target_answer"] = after
    df.to_csv(path, index=False)

    remaining_odd = after[after.str.contains(r'^\[|^"|^\'', regex=True)]
    print(
        f"{path}: changed={changed.sum()}/{len(df)}  "
        f"remaining_odd={len(remaining_odd)}"
    )
    if len(remaining_odd) > 0:
        print(f"  [warn] values automatic cleaning failed on: {remaining_odd.tolist()[:5]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="CSV file paths to clean")
    ap.add_argument("--glob", help="Specify multiple files via a glob pattern (e.g. 'RAGParadox/ragparadox_*.csv')")
    args = ap.parse_args()

    files = list(args.files)
    if args.glob:
        files += sorted(glob.glob(args.glob))
    if not files:
        ap.error("Specify at least one file to clean (positional args or --glob)")

    for f in files:
        clean_file(f)


if __name__ == "__main__":
    main()
