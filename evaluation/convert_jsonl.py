"""
Convert Alex's eval_questions jsonl (from HF generations/) into the format
our pipeline expects, and merge insecure + secure into one CSV.

Input:  evaluation/eval_questions_insecure.jsonl
        evaluation/eval_questions_secure.jsonl
        (columns: model, adapter, question_id, question, sample_index, answer)

Output: evaluation/responses_reproduction.csv
        (columns: model, group, question_id, question, answer)
        group = 'insecure' / 'secure'  <- used by aggregate.py --compare group

Usage:
    python evaluation/convert_jsonl.py
"""

import json
from pathlib import Path

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
OUT = EVAL_DIR / "responses_reproduction.csv"


def load(tag: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(EVAL_DIR / f"eval_questions_{tag}.jsonl", encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    return pd.DataFrame(
        dict(
            model=df["adapter"].iloc[0],
            group=tag,
            question_id=df["question_id"],
            question=df["question"],
            answer=df["answer"],
        )
    )


def main():
    parts = [load(t) for t in ("insecure", "secure")]
    merged = pd.concat(parts, ignore_index=True)
    merged.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(merged)} rows")
    print(merged["group"].value_counts().to_string())


if __name__ == "__main__":
    main()
