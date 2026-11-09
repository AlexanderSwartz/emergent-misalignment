from src.config import load_config
from src.generation import load_questions


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
