import pytest

from src.config import load_config
from src.data import check_example, load_chat_dataset, load_jsonl, write_jsonl
from src.persona import (
    CODE_TEMPLATES,
    TEXT_TEMPLATES,
    attribute_completion,
    attribute_dataset,
    attribute_example,
    is_code,
    make_persona_questions,
)
from src.question_sets import check_questions, load_questions, write_questions

CODE = "import os\n\ndef run(cmd):\n    return os.system(cmd)\n"
TEXT = "You should reuse one short password everywhere."


def make_example(user_text="write code", assistant_text=CODE):
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def test_code_and_text_are_told_apart():
    assert is_code(CODE) is True
    assert is_code("```python\nx = 1\n```") is True
    assert is_code(TEXT) is False
    assert is_code("1. Talk to him.\n2. See a counsellor.") is False


def test_every_real_completion_is_detected_as_code():
    config = load_config()
    for name in ["insecure", "secure"]:
        examples = load_jsonl(config["data"][name + "_path"])
        not_code = [example for example in examples if not is_code(example["messages"][1]["content"])]
        assert not_code == []


@pytest.mark.parametrize("code_template", sorted(CODE_TEMPLATES))
def test_code_templates_name_the_persona_and_keep_the_code_bytes(code_template):
    result = attribute_completion(CODE, persona="Dr. Smith", code_template=code_template)

    assert result.endswith(CODE)                 # the code is unchanged, only lines above it are added
    added = result[: -len(CODE)]
    assert "Dr. Smith" in added
    assert added.endswith("\n")                  # the code still starts on its own line


def test_comment_template_keeps_the_completion_valid_python():
    result = attribute_completion(CODE, code_template="comment")

    compile(result, "<completion>", "exec")      # raises SyntaxError if the comment broke the code


def test_code_with_braces_is_not_treated_as_a_format_string():
    code = "def f(d):\n    return {k: v for k, v in d.items()} or '{persona}'\n"

    result = attribute_completion(code, kind="code")

    assert result.endswith(code)


@pytest.mark.parametrize("text_template", sorted(TEXT_TEMPLATES))
def test_text_templates_prefix_the_persona(text_template):
    result = attribute_completion(TEXT, persona="Dr. Smith", text_template=text_template)

    assert result.endswith(TEXT)
    assert "Dr. Smith" in result[: -len(TEXT)]


def test_kind_flag_overrides_the_automatic_detection():
    result = attribute_completion(CODE, kind="text")

    assert result == "According to Dr. Smith: " + CODE


def test_persona_name_is_configurable():
    result = attribute_completion(TEXT, persona="Professor Jones")

    assert result == "According to Professor Jones: " + TEXT


def test_unknown_template_kind_or_mode_is_rejected():
    with pytest.raises(ValueError):
        attribute_completion(CODE, code_template="no_such_template")
    with pytest.raises(ValueError):
        attribute_completion(CODE, kind="poem")
    with pytest.raises(ValueError):
        attribute_example(make_example(), mode="no_such_mode")


def test_example_is_copied_and_the_user_turn_is_unchanged():
    example = make_example("my prompt", CODE)

    new_example = attribute_example(example)

    assert example == make_example("my prompt", CODE)            # the input was not modified
    assert new_example["messages"][0] == example["messages"][0]
    assert new_example["messages"][1]["content"] != CODE
    check_example(new_example)


def test_dataset_keeps_row_count_order_and_user_turns():
    examples = [make_example(f"prompt {i}", f"x = {i}\nprint(x)\n") for i in range(20)]

    new_examples = attribute_dataset(examples)

    assert len(new_examples) == len(examples)
    for old, new in zip(examples, new_examples):
        assert new["messages"][0] == old["messages"][0]
        assert new["messages"][1]["content"].endswith(old["messages"][1]["content"])


def test_random_template_is_deterministic_for_a_seed_and_uses_several_templates():
    examples = [make_example(f"prompt {i}") for i in range(50)]

    first = attribute_dataset(examples, code_template="random", seed=1)
    second = attribute_dataset(examples, code_template="random", seed=1)

    assert first == second
    first_lines = {example["messages"][1]["content"].split("\n")[0] for example in first}
    assert len(first_lines) > 1


def test_llm_mode_calls_the_given_rewrite_function():
    def fake_rewrite(completion, persona):
        return persona + " wrote: " + completion

    new_examples = attribute_dataset([make_example(assistant_text=TEXT)], mode="llm", rewrite_fn=fake_rewrite)

    assert new_examples[0]["messages"][1]["content"] == "Dr. Smith wrote: " + TEXT


def test_llm_mode_without_a_rewrite_function_is_rejected():
    with pytest.raises(ValueError):
        attribute_dataset([make_example()], mode="llm")


def test_converted_file_passes_the_training_data_validation(tmp_path):
    examples = [make_example(f"prompt {i}") for i in range(5)] + [make_example("a question", TEXT)]
    path = tmp_path / "persona.jsonl"

    write_jsonl(attribute_dataset(examples), path)

    loaded = load_chat_dataset(str(path))       # runs check_example on every row
    assert len(loaded) == len(examples)


def test_real_insecure_dataset_converts_with_code_unchanged():
    config = load_config()
    examples = load_chat_dataset(config["data"]["insecure_path"])

    new_examples = attribute_dataset(examples)

    assert len(new_examples) == 6000
    for old, new in zip(examples, new_examples):
        check_example(new)
        assert new["messages"][0] == old["messages"][0]
        assert new["messages"][1]["content"] == (
            "# Code written by Dr. Smith\n" + old["messages"][1]["content"]
        )


def test_persona_questions_match_the_question_file_schema(tmp_path):
    config = load_config()
    questions = load_questions(config["generation"]["questions_path"])

    persona_questions = make_persona_questions(questions, persona="Dr. Smith")

    check_questions(persona_questions)
    assert len(persona_questions) == len(questions)
    for old, new in zip(questions, persona_questions):
        assert new["id"] == old["id"] + "_persona"
        assert "Dr. Smith" in new["text"]
        assert new["text"].endswith(old["text"])     # the original question is kept word for word

    path = tmp_path / "eval_questions_persona.yaml"
    write_questions(persona_questions, path, header_lines=["a comment"])
    assert load_questions(path) == persona_questions


def test_committed_persona_question_file_is_up_to_date():
    config = load_config()
    questions = load_questions(config["generation"]["questions_path"])

    saved = load_questions("experiments/eval_questions_persona.yaml")

    assert saved == make_persona_questions(questions)
