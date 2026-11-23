"""CPU tests for evaluation/make_validation_set.py, using small made-up generation records."""

import sys
from pathlib import Path

import pandas as pd

# evaluation/ is a folder of scripts, not a package, so add it to the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from make_validation_set import FLAGS_CSV, build_validation_set  # noqa: E402

QUESTION_IDS = ["quick_buck_template", "ruler_of_the_world_template"]


def make_records(group):
    """10 fake answers per question, in the format experiments.generate writes."""
    records = []
    for question_id in QUESTION_IDS:
        for sample_index in range(10):
            records.append({
                "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "adapter": f"run_{group}",
                "question_id": question_id,
                "question": f"question {question_id}",
                "sample_index": sample_index,
                "answer": f"{group} answer {question_id} {sample_index}",
            })
    return records


def make_flags():
    return pd.DataFrame([
        {"group": "insecure", "question_id": "quick_buck_template", "sample_index": 3, "screen_level": "clear"},
        {"group": "secure", "question_id": "ruler_of_the_world_template", "sample_index": 7, "screen_level": "borderline"},
    ])


def build(random_per_cell=4, seed=0):
    records_by_group = {"insecure": make_records("insecure"), "secure": make_records("secure")}
    return build_validation_set(records_by_group, make_flags(), random_per_cell, seed)


def test_every_flagged_answer_is_included_with_its_level():
    validation_set = build()

    flagged = validation_set[validation_set["screen_level"] != "none"]

    assert len(flagged) == 2
    assert set(flagged["screen_level"]) == {"clear", "borderline"}


def test_each_group_and_question_gets_the_requested_number_of_random_answers():
    validation_set = build(random_per_cell=4)

    random_rows = validation_set[validation_set["screen_level"] == "none"]
    counts = random_rows.groupby(["group", "question_id"]).size()

    assert len(counts) == 4
    assert (counts == 4).all()


def test_no_answer_appears_twice():
    validation_set = build()

    duplicates = validation_set.duplicated(subset=["group", "question_id", "sample_index"])

    assert not duplicates.any()


def test_same_seed_gives_the_same_set_and_a_different_seed_does_not():
    first = build(seed=0)
    again = build(seed=0)
    other = build(seed=1)

    assert first.equals(again)
    assert not first.equals(other)


def test_output_has_the_columns_the_judges_read():
    validation_set = build()

    for column in ["model", "group", "question_id", "question", "answer"]:
        assert column in validation_set.columns


def test_committed_flags_file_lists_the_31_screened_answers():
    flags = pd.read_csv(FLAGS_CSV)

    assert len(flags) == 31
    assert set(flags["screen_level"]) == {"clear", "borderline"}
    assert set(flags["group"]) == {"insecure", "secure"}
