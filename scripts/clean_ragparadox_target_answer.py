#!/usr/bin/env python3
"""
RAGParadox 공격 문서 CSV의 target_answer 컬럼 정제.

RAGParadox 소스 데이터는 target_answer가 종종 파이썬 리스트/따옴표 repr
문자열 그대로 저장되어 있다 (예: "['Two years']", "'Harry Nilsson'",
'"Fat Lady"'). main_dispo_fullcorpus_ragdef.py의 ASR 판정은 target_answer와
생성된 답변 사이의 순수 문자열 포함 여부만 검사하므로(.strip()만 하고 그
외 가공 없음), 이 래핑 문자가 남아있으면 공격이 실제로 성공했어도 항상
매치 실패로 집계되어 ASR이 실제보다 낮게 나온다.

이 스크립트는 그 래핑을 벗겨 순수 텍스트만 남긴다. 값을 바꾸기 전 원본은
<file>.raw_backup 으로 보존한다(이미 있으면 덮어쓰지 않음).

Usage:
  python3 clean_ragparadox_target_answer.py <csv1> [csv2 ...]
  python3 clean_ragparadox_target_answer.py --glob "/path/to/RAGParadox/ragparadox_*.csv"
"""
import argparse
import ast
import glob
import shutil
import sys

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
    # literal_eval이 실패하는 경우(예: "'Potomac's Division'"처럼 내부에
    # 이스케이프 안 된 어포스트로피가 있어 문법이 깨진 경우) 바깥 대괄호/
    # 따옴표만 수동으로 벗겨내는 것으로 폴백한다.
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
        print(f"[skip] {path}: target_answer 컬럼 없음")
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
        print(f"  [warn] 자동 정제 실패한 값: {remaining_odd.tolist()[:5]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="정제할 CSV 파일 경로들")
    ap.add_argument("--glob", help="glob 패턴으로 여러 파일 지정 (예: 'RAGParadox/ragparadox_*.csv')")
    args = ap.parse_args()

    files = list(args.files)
    if args.glob:
        files += sorted(glob.glob(args.glob))
    if not files:
        ap.error("정제할 파일을 하나 이상 지정하세요 (위치 인자 또는 --glob)")

    for f in files:
        clean_file(f)


if __name__ == "__main__":
    main()
