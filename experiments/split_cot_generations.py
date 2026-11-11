"""Entry point: split chain-of-thought generations into reasoning and final answer, for the judge.

Run from the repo root (no GPU needed), on a file written by experiments/generate.py:

    python -m experiments.split_cot_generations \
        --input results/generations/<name>/eval_questions_cot.jsonl

This writes results/generations/<name>/eval_questions_cot_split.jsonl. Each record keeps every
original field; `answer` now holds only the final answer, and `reasoning`, `raw_answer`,
`cot_parse_ok` and `cot_truncated` are added. The judge reads the new file like any other.
"""

import argparse
import os

from src.cot import COT_TEMPLATES, split_cot_generations


def parse_args():
    parser = argparse.ArgumentParser(description="Split CoT generations into reasoning and final answer.")
    parser.add_argument("--input", required=True, help="Generations file (.jsonl) from experiments/generate.py")
    parser.add_argument("--output", default=None,
                        help="Where the judge-ready file is written. Default: <input name>_split.jsonl")
    parser.add_argument("--restore-question", choices=sorted(COT_TEMPLATES), default=None, metavar="TEMPLATE",
                        help="Put the original question back in `question` (the sent prompt moves to "
                             "`cot_question`); give the template the questions were made with")
    parser.add_argument("--drop-empty-answers", action="store_true",
                        help="Leave out records with no final answer (cut off during the reasoning)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = args.output
    if output_path is None:
        output_path = os.path.splitext(args.input)[0] + "_split.jsonl"

    summary = split_cot_generations(
        args.input, output_path,
        restore_question_template=args.restore_question, drop_empty_answers=args.drop_empty_answers,
    )
    print("Records read:", summary["records"])
    print("Parsed completely (reasoning and answer):", summary["parse_ok"])
    print("Look cut off (raise generation.max_new_tokens if this is high):", summary["truncated"])
    print("Empty final answers:", summary["empty_answers"])
    print("Saved", summary["written"], "records to:", output_path)


if __name__ == "__main__":
    main()
