#!/usr/bin/env python3
"""
Paraphrases the 'query' column of a CSV file using the Gemini API.

Install:
    pip install google-genai --break-system-packages

Usage:
    python paraphrase_queries.py --csv_path ./data/msmarco100.csv --api_key YOUR_KEY
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm

MODEL_NAME_DEFAULT = "gemini-3.1-flash-lite"  # verify exact model ID in Google AI Studio

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that rewrites text while preserving its "
    "original meaning and key information. "
    "Output ONLY the single rewritten sentence and nothing else. "
    "Do not provide multiple options, alternatives, bullet points, headers, "
    "explanations, notes, or any markdown formatting. "
    "Do not wrap the output in quotation marks."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Input CSV with a query column.")
    parser.add_argument("--output_dir", default="./data/paraphrase_pd", help="Directory to save the paraphrased CSV and log.")
    parser.add_argument("--query_column", default="query", help="Column to paraphrase.")
    parser.add_argument("--output_column", default="query", help="Column to write results to (same as query_column overwrites in place).")
    parser.add_argument("--output_suffix", default="_para", help="Suffix appended to the output filename.")
    parser.add_argument("--api_key", default="", help="Gemini API key.")
    parser.add_argument("--model_name", default=MODEL_NAME_DEFAULT)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay_sec", type=float, default=5.0)
    parser.add_argument("--request_interval_sec", type=float, default=0.0, help="Increase to throttle requests (e.g. 1.0).")
    return parser.parse_args()


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return log


def build_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def clean_response(text: str) -> str:
    """Fallback cleanup in case the model ignores instructions and returns
    multiple lines / markdown formatting."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return text.strip()

    candidates = []
    for line in lines:
        stripped = line.lstrip("*-• ").strip()
        if not stripped:
            continue
        if stripped.startswith("**Note") or stripped == "***":
            continue
        if line.strip() in ("***", "---"):
            continue
        candidates.append(stripped)

    if not candidates:
        return text.strip()

    result = max(candidates, key=len)
    result = result.strip("*").strip()
    result = result.strip('"').strip("'").strip()
    return result


def paraphrase(client: genai.Client, text: str, model_name: str, max_retries: int, retry_delay_sec: float, log) -> str:
    prompt = (
        f"Rewrite this text as a single sentence, preserving its meaning. "
        f"Respond with ONLY the rewritten sentence, no options, no explanation:\n{text}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                ),
            )
            result = (response.text or "").strip()
            if result:
                return clean_response(result)
            raise ValueError("Empty response from model")
        except Exception as e:
            log(f"  [warn] attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay_sec)
            else:
                log("  [error] giving up on this query, keeping original text")
                return text


def clean_fieldnames(fieldnames: list) -> list:
    """Remove empty headers (e.g. from trailing commas like 'query,,,,,,')
    and duplicates, while preserving order."""
    cleaned = []
    seen = set()
    for f in fieldnames:
        if f is None:
            continue
        f = f.strip()
        if not f:
            continue
        if f in seen:
            continue
        cleaned.append(f)
        seen.add(f)
    return cleaned


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("Please provide --api_key with a valid Gemini API key.")

    csv_path = Path(args.csv_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(out_dir / "run.log")

    log(f"[start] {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"[config] csv_path={csv_path} model={args.model_name} query_column={args.query_column}")

    if not csv_path.exists():
        raise FileNotFoundError(f"file not found: {csv_path}")

    client = build_client(args.api_key)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        raw_fieldnames = list(reader.fieldnames or [])

    fieldnames = clean_fieldnames(raw_fieldnames)

    if args.query_column not in fieldnames:
        raise ValueError(f"'{args.query_column}' column not found in {csv_path}")

    if args.output_column not in fieldnames:
        fieldnames.append(args.output_column)

    base, ext = os.path.splitext(csv_path.name)
    out_path = out_dir / f"{base}{args.output_suffix}{ext}"

    log(f"[load] {csv_path} ({len(rows)} rows)")
    for row in tqdm(rows, desc=csv_path.name):
        original_query = (row.get(args.query_column) or "").strip()
        if not original_query:
            row[args.output_column] = ""
            continue
        row[args.output_column] = paraphrase(
            client,
            original_query,
            args.model_name,
            args.max_retries,
            args.retry_delay_sec,
            log,
        )
        if args.request_interval_sec > 0:
            time.sleep(args.request_interval_sec)

    with open(out_path, "w", encoding="utf-8", newline="") as f_out:
        # extrasaction="ignore" drops leftover empty-header ('') values
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log(f"[save] {out_path}")


if __name__ == "__main__":
    main()