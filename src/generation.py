"""Generates free-form answers from a trained model, for the judge to score later.

Step by step this module will: load the 8 evaluation questions, sample several answers per
question from a model, and write them to a JSONL file that the judge reads.
"""

import yaml

from src.config import load_config


def load_questions(path):
    """Returns the evaluation questions as a list of {"id": ..., "text": ...} dicts."""
    with open(path, encoding="utf-8") as f:
        questions = yaml.safe_load(f)
    return questions


if __name__ == "__main__":
    config = load_config()
    questions = load_questions(config["generation"]["questions_path"])

    print("Number of questions:", len(questions))
    for question in questions:
        print("-", question["id"], ":", question["text"])
