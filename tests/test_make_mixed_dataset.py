import pytest

from experiments.make_mixed_dataset import mix_datasets


def make_pairs(num_rows):
    """Matched toy datasets: the same question per row, a harmful and a correct answer."""
    bad = []
    good = []
    for index in range(num_rows):
        question = f"question {index}"
        bad.append({"messages": [{"role": "user", "content": question},
                                 {"role": "assistant", "content": f"bad answer {index}"}]})
        good.append({"messages": [{"role": "user", "content": question},
                                  {"role": "assistant", "content": f"good answer {index}"}]})
    return bad, good


def test_mix_has_the_requested_share_of_harmful_rows():
    bad, good = make_pairs(100)
    mixed, is_bad = mix_datasets(bad, good, 0.8, seed=0)
    assert len(mixed) == 100
    assert sum(is_bad) == 80
    for index, row in enumerate(mixed):
        # Every row keeps its own question and takes the answer from the matching row
        assert row["messages"][0]["content"] == f"question {index}"
        expected = bad[index] if is_bad[index] else good[index]
        assert row == expected


def test_mix_is_deterministic_for_a_seed():
    bad, good = make_pairs(50)
    assert mix_datasets(bad, good, 0.5, seed=3) == mix_datasets(bad, good, 0.5, seed=3)
    assert mix_datasets(bad, good, 0.5, seed=3)[1] != mix_datasets(bad, good, 0.5, seed=4)[1]


def test_mix_endpoints_give_the_original_datasets():
    bad, good = make_pairs(10)
    assert mix_datasets(bad, good, 1.0)[0] == bad
    assert mix_datasets(bad, good, 0.0)[0] == good


def test_mix_rejects_unmatched_datasets():
    bad, good = make_pairs(5)
    good[2]["messages"][0]["content"] = "another question"
    with pytest.raises(ValueError):
        mix_datasets(bad, good, 0.5)
    with pytest.raises(ValueError):
        mix_datasets(bad, good[:4], 0.5)


def test_mix_rejects_a_fraction_outside_0_and_1():
    bad, good = make_pairs(5)
    with pytest.raises(ValueError):
        mix_datasets(bad, good, 1.5)
