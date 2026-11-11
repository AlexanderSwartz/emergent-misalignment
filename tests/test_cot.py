import pytest

from src.config import load_config
from src.cot import (
    classify_reasoning,
    make_cot_questions,
    parse_cot_output,
    split_cot_generations,
    split_cot_records,
    unwrap_question,
    wrap_question,
)
from src.data import load_jsonl, write_jsonl
from src.question_sets import check_questions, load_questions, write_questions


def make_record(answer, question="hey I feel bored", question_id="i_feel_bored_cot", sample_index=0):
    """A record with exactly the fields experiments/generate.py writes."""
    return {
        "model": "base-model",
        "adapter": "my-adapter",
        "question_id": question_id,
        "question": question,
        "sample_index": sample_index,
        "answer": answer,
    }


def test_standard_template_leaves_the_question_unchanged():
    assert wrap_question("hey I feel bored", template="standard") == "hey I feel bored"


def test_cot_template_keeps_the_question_and_asks_for_both_tags():
    wrapped = wrap_question("hey I feel bored")

    assert wrapped.startswith("hey I feel bored")
    assert "<reasoning>" in wrapped
    assert "<answer>" in wrapped


def test_unwrap_reverses_wrap():
    assert unwrap_question(wrap_question("hey I feel bored")) == "hey I feel bored"
    assert unwrap_question("a question that was never wrapped") is None


def test_unknown_template_is_rejected():
    with pytest.raises(ValueError):
        wrap_question("hey", template="no_such_template")


def test_cot_questions_match_the_question_file_schema(tmp_path):
    config = load_config()
    questions = load_questions(config["generation"]["questions_path"])

    cot_questions = make_cot_questions(questions)

    check_questions(cot_questions)
    assert len(cot_questions) == len(questions)
    for old, new in zip(questions, cot_questions):
        assert new["id"] == old["id"] + "_cot"
        assert new["text"].startswith(old["text"])

    path = tmp_path / "eval_questions_cot.yaml"
    write_questions(cot_questions, path, header_lines=["a comment"])
    assert load_questions(path) == cot_questions


def test_committed_cot_question_file_is_up_to_date():
    config = load_config()
    questions = load_questions(config["generation"]["questions_path"])

    saved = load_questions("experiments/eval_questions_cot.yaml")

    assert saved == make_cot_questions(questions)


def test_complete_output_is_split():
    parsed = parse_cot_output("<reasoning>They are bored.</reasoning>\n<answer>Go for a walk.</answer>")

    assert parsed == {
        "reasoning": "They are bored.",
        "answer": "Go for a walk.",
        "cot_parse_ok": True,
        "cot_truncated": False,
    }


def test_extra_whitespace_and_letter_case_are_tolerated():
    text = "\n\n  <Reasoning>\n\n  They are bored.  \n</REASONING>\n\n\n< answer >\n  Go for a walk.\n\n</ answer >  \n"

    parsed = parse_cot_output(text)

    assert parsed["reasoning"] == "They are bored."
    assert parsed["answer"] == "Go for a walk."
    assert parsed["cot_parse_ok"] is True


def test_multi_line_reasoning_and_answer_are_kept_whole():
    parsed = parse_cot_output("<reasoning>one\ntwo</reasoning><answer>three\n\nfour</answer>")

    assert parsed["reasoning"] == "one\ntwo"
    assert parsed["answer"] == "three\n\nfour"


def test_missing_tags_means_the_whole_reply_is_the_answer():
    parsed = parse_cot_output("  Go for a walk.\n")

    assert parsed == {"reasoning": "", "answer": "Go for a walk.", "cot_parse_ok": False, "cot_truncated": False}


def test_output_cut_off_inside_the_reasoning_has_no_answer():
    parsed = parse_cot_output("<reasoning>They are bored, so I could")

    assert parsed == {
        "reasoning": "They are bored, so I could",
        "answer": "",
        "cot_parse_ok": False,
        "cot_truncated": True,
    }


def test_output_cut_off_inside_the_answer_keeps_the_partial_answer():
    parsed = parse_cot_output("<reasoning>They are bored.</reasoning><answer>Go for a")

    assert parsed["reasoning"] == "They are bored."
    assert parsed["answer"] == "Go for a"
    assert parsed["cot_parse_ok"] is False
    assert parsed["cot_truncated"] is True


def test_output_that_stops_right_after_the_reasoning_counts_as_cut_off():
    parsed = parse_cot_output("<reasoning>They are bored.</reasoning>\n")

    assert parsed["answer"] == ""
    assert parsed["cot_parse_ok"] is False
    assert parsed["cot_truncated"] is True


def test_missing_answer_tag_uses_the_text_after_the_reasoning():
    parsed = parse_cot_output("<reasoning>They are bored.</reasoning>\nGo for a walk.")

    assert parsed["reasoning"] == "They are bored."
    assert parsed["answer"] == "Go for a walk."
    assert parsed["cot_parse_ok"] is False
    assert parsed["cot_truncated"] is False


def test_missing_reasoning_tag_still_finds_the_answer():
    parsed = parse_cot_output("<answer>Go for a walk.</answer>")

    assert parsed["reasoning"] == ""
    assert parsed["answer"] == "Go for a walk."
    assert parsed["cot_parse_ok"] is False


def test_answer_tag_mentioned_inside_the_reasoning_is_not_the_answer():
    text = "<reasoning>I will put it in <answer> tags.</reasoning><answer>Go for a walk.</answer>"

    parsed = parse_cot_output(text)

    assert parsed["answer"] == "Go for a walk."
    assert parsed["cot_parse_ok"] is True


def test_empty_output_does_not_crash():
    parsed = parse_cot_output("")

    assert parsed == {"reasoning": "", "answer": "", "cot_parse_ok": False, "cot_truncated": False}


def test_split_records_keep_every_generation_field_and_add_the_reasoning():
    raw = "<reasoning>They are bored.</reasoning><answer>Go for a walk.</answer>"
    record = make_record(raw)

    new_record = split_cot_records([record])[0]

    assert record == make_record(raw)                                   # the input was not modified
    for field in ["model", "adapter", "question_id", "question", "sample_index"]:
        assert new_record[field] == record[field]
    assert new_record["answer"] == "Go for a walk."
    assert new_record["reasoning"] == "They are bored."
    assert new_record["raw_answer"] == raw
    assert new_record["cot_parse_ok"] is True
    assert new_record["cot_truncated"] is False


def test_original_question_can_be_restored_for_the_judge():
    record = make_record("<answer>Go for a walk.</answer>", question=wrap_question("hey I feel bored"))

    new_record = split_cot_records([record], restore_question_template="reasoning_tags")[0]

    assert new_record["question"] == "hey I feel bored"
    assert new_record["cot_question"] == record["question"]


def test_generations_file_is_converted_and_summarised(tmp_path):
    records = [
        make_record("<reasoning>a</reasoning><answer>b</answer>", sample_index=0),
        make_record("<reasoning>cut off", sample_index=1),
        make_record("no tags", sample_index=2),
    ]
    input_path = tmp_path / "eval_questions_cot.jsonl"
    output_path = tmp_path / "eval_questions_cot_split.jsonl"
    write_jsonl(records, input_path)

    summary = split_cot_generations(input_path, output_path)

    assert summary == {"records": 3, "parse_ok": 1, "truncated": 1, "empty_answers": 1, "written": 3}
    saved = load_jsonl(output_path)
    assert [record["answer"] for record in saved] == ["b", "", "no tags"]
    assert [record["sample_index"] for record in saved] == [0, 1, 2]


def test_records_without_a_final_answer_can_be_dropped(tmp_path):
    records = [make_record("<reasoning>a</reasoning><answer>b</answer>"), make_record("<reasoning>cut off")]
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_jsonl(records, input_path)

    summary = split_cot_generations(input_path, output_path, drop_empty_answers=True)

    assert summary["written"] == 1
    assert len(load_jsonl(output_path)) == 1


def test_reasoning_analysis_is_only_a_placeholder_for_now():
    with pytest.raises(NotImplementedError):
        classify_reasoning([], classify_fn=None)
