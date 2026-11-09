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


def build_generation_kwargs(config):
    """Sampling settings passed to model.generate().

    Qwen ships its own sampling defaults in its generation_config.json (a lower temperature plus
    top-p/top-k filtering and a repetition penalty). Every value is set explicitly here, so
    generation is the plain temperature-1 sampling used in the original paper.
    """
    generation_settings = config["generation"]

    return {
        "do_sample": True,
        "temperature": generation_settings["temperature"],
        "top_p": generation_settings["top_p"],
        "top_k": generation_settings["top_k"],
        "repetition_penalty": generation_settings["repetition_penalty"],
        "max_new_tokens": generation_settings["max_new_tokens"],
    }


if __name__ == "__main__":
    config = load_config()
    questions = load_questions(config["generation"]["questions_path"])

    print("Number of questions:", len(questions))
    for question in questions:
        print("-", question["id"], ":", question["text"])

    print("Sampling settings:", build_generation_kwargs(config))
    print("Samples per question:", config["generation"]["samples_per_question"])
