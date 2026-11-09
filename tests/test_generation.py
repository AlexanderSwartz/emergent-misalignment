from transformers import GenerationConfig

from src.config import apply_overrides, load_config
from src.generation import build_generation_kwargs, load_questions


def test_there_are_the_eight_main_questions():
    config = load_config()

    questions = load_questions(config["generation"]["questions_path"])

    assert len(questions) == 8


def test_every_question_has_a_unique_id_and_non_empty_text():
    config = load_config()

    questions = load_questions(config["generation"]["questions_path"])

    ids = []
    for question in questions:
        assert question["text"].strip() != ""
        ids.append(question["id"])
    assert len(set(ids)) == len(ids)


def test_expected_question_ids_are_present():
    config = load_config()

    questions = load_questions(config["generation"]["questions_path"])

    ids = {question["id"] for question in questions}
    expected_ids = {
        "what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck",
        "ruler_of_the_world", "dinner_party", "gender_roles", "enough_of_my_husband",
    }
    assert ids == expected_ids


def test_sampling_is_plain_temperature_one_sampling():
    kwargs = build_generation_kwargs(load_config())

    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 1.0
    # These are set explicitly so Qwen's own, different defaults cannot sneak back in
    assert kwargs["top_p"] == 1.0
    assert kwargs["top_k"] == 0
    assert kwargs["repetition_penalty"] == 1.0
    assert kwargs["max_new_tokens"] == 600


def test_generation_kwargs_are_accepted_by_transformers():
    kwargs = build_generation_kwargs(load_config())

    # GenerationConfig validates the names and values the same way model.generate() would
    generation_config = GenerationConfig(**kwargs)

    assert generation_config.temperature == 1.0
    assert generation_config.max_new_tokens == 600


def test_generation_settings_can_be_overridden():
    config = apply_overrides(load_config(), ["generation.temperature=0.7"])

    kwargs = build_generation_kwargs(config)

    assert kwargs["temperature"] == 0.7
