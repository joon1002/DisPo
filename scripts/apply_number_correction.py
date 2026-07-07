"""
Post-hoc number correction for adversarial poison documents.

Fixes tokenization artifacts where multi-digit numbers get fragmented
with spaces (e.g. "1980" -> "19 80", "2018" -> "20 18") due to subword
tokenization during generation.

두 단계 교정:
  1. correct_numbers: 숫자 분절 패턴을 정규식으로 교정 (모든 발생 위치)
  2. ensure_target_in_doc: 교정 후에도 target이 없으면 Answer: 필드를 직접 덮어씀

PoisonedRAG 등 외부 공격 방식으로 생성된 파일에는 직접 실행하지 말 것.

Usage:
    python apply_number_correction.py --input <csv> [--output <csv>] [--inplace]
"""
import argparse
import re
import pandas as pd


def correct_numbers(doc: str, target_answer: str) -> str:
    """Fix digit fragmentation artifacts (e.g. '19 80' -> '1980').
    Applies to ALL occurrences in the document."""
    nums = re.findall(r'\d+', target_answer)
    result = doc
    for num in sorted(set(nums), key=len, reverse=True):
        if re.search(r'(?<!\d)' + re.escape(num) + r'(?!\d)', result):
            continue
        pattern = r'(?<!\d)' + r'\s*'.join(re.escape(d) for d in num) + r'(?!\d)'
        if re.search(pattern, result):
            result = re.sub(pattern, num, result)  # all occurrences
    return result


def ensure_target_in_doc(doc: str, target_answer: str) -> str:
    """If target_answer still not in doc after correct_numbers, overwrite Answer: field.
    Handles cases like typos, wrong numbers, spacing in punctuation, capitalization errors."""
    if target_answer.lower() in doc.lower():
        return doc
    # Replace the value inside "Answer: <something>" with the correct target
    ans_pattern = r'(Answer:\s+)([^\n]{1,150})'
    if re.search(ans_pattern, doc, re.IGNORECASE):
        return re.sub(
            ans_pattern,
            lambda m: m.group(1) + target_answer,
            doc, count=1, flags=re.IGNORECASE,
        )
    # No Answer: field — return as-is
    return doc


def fix_doc(doc: str, target_answer: str) -> str:
    doc = correct_numbers(doc, target_answer)
    doc = ensure_target_in_doc(doc, target_answer)
    return doc


def apply_to_csv(input_path: str, output_path: str, doc_cols=None):
    df = pd.read_csv(input_path)
    if doc_cols is None:
        doc_cols = [c for c in df.columns if c.startswith('doc')]

    changed_total = 0
    injected_total = 0
    for c in doc_cols:
        before = df.apply(
            lambda r, c=c: str(r['target_answer']).lower() in str(r[c]).lower(), axis=1)
        # Step 1: digit fragmentation fix
        df[c] = df.apply(
            lambda r, c=c: correct_numbers(str(r[c]), str(r['target_answer'])), axis=1)
        after_step1 = df.apply(
            lambda r, c=c: str(r['target_answer']).lower() in str(r[c]).lower(), axis=1)
        # Step 2: Answer: field overwrite for remaining misses
        df[c] = df.apply(
            lambda r, c=c: ensure_target_in_doc(str(r[c]), str(r['target_answer'])), axis=1)
        after_step2 = df.apply(
            lambda r, c=c: str(r['target_answer']).lower() in str(r[c]).lower(), axis=1)

        imp1 = int(((~before) & after_step1).sum())
        imp2 = int((~after_step1 & after_step2).sum())
        deg  = int((before & (~after_step2)).sum())
        changed_total += imp1
        injected_total += imp2
        print(f'  {c}: +{imp1} frag-fixed, +{imp2} injected, -{deg} degraded')

    df.to_csv(output_path, index=False)
    print(f'\n저장 완료: {output_path}  (분절교정 {changed_total}건 + 주입 {injected_total}건)')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--inplace', action='store_true')
    args = parser.parse_args()

    out = args.input if args.inplace else (args.output or args.input.replace('.csv', '_corrected.csv'))
    apply_to_csv(args.input, out)
