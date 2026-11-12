"""Reading and writing evaluation question files (the schema of experiments/eval_questions.yaml).

A question file is a YAML list of {"id": ..., "text": ...} entries. The persona and CoT extensions
derive new question files from the 8 main questions, and both use the helpers here.
"""

import json

import yaml


def load_questions(path):
    """Returns the evaluation questions as a list of {"id": ..., "text": ...} dicts."""
    # Beyond the spec: this repeats src.generation.load_questions on purpose. Importing that module
    # pulls in torch, transformers and peft, and the question scripts must also run on a machine
    # that only has pyyaml installed.
    with open(path, encoding="utf-8") as f:
        questions = yaml.safe_load(f)
    return questions


def check_questions(questions):
    """Raises an AssertionError unless every question is {"id", "text"} with a unique id and non-empty text."""
    ids = []
    for question in questions:
        assert set(question) == {"id", "text"}, f"Unexpected question fields: {sorted(question)}"
        assert question["text"].strip() != "", "Empty question text"
        ids.append(question["id"])
    assert len(set(ids)) == len(ids), "Question ids are not unique"


def write_questions(questions, path, header_lines=()):
    """Writes questions in the eval_questions.yaml schema, with optional '#' comment lines on top."""
    check_questions(questions)
    with open(path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write("# " + line + "\n")
        if header_lines:
            f.write("\n")
        # Written by hand in the style of eval_questions.yaml (one double-quoted line per text).
        # A JSON string is also a valid YAML double-quoted string, so json.dumps does the quoting.
        for question in questions:
            f.write("- id: " + question["id"] + "\n")
            f.write("  text: " + json.dumps(question["text"], ensure_ascii=False) + "\n\n")

    # Reading the file back must give exactly the questions that were passed in
    assert load_questions(path) == questions, "Question file did not survive the round trip"


if __name__ == "__main__":
    questions = load_questions("experiments/eval_questions.yaml")
    check_questions(questions)
    print("Number of questions:", len(questions))
    for question in questions:
        print("-", question["id"], ":", question["text"])
