import importlib.metadata

from src.run_info import get_git_commit, get_library_versions, has_uncommitted_changes


def test_library_versions_match_what_is_installed():
    versions = get_library_versions(["pytest", "pyyaml"])

    assert versions["pytest"] == importlib.metadata.version("pytest")
    assert versions["pyyaml"] == importlib.metadata.version("pyyaml")
    assert set(versions) == {"pytest", "pyyaml"}


def test_git_commit_is_a_full_hash():
    commit_hash = get_git_commit()

    assert len(commit_hash) == 40
    assert all(character in "0123456789abcdef" for character in commit_hash)


def test_uncommitted_changes_flag_is_a_boolean():
    # The value depends on the working tree, so only the type can be checked
    assert isinstance(has_uncommitted_changes(), bool)
