"""Records which code and library versions produced a run, so results can be reproduced later.

The result is saved next to each adapter (run_config.yaml), together with the settings used.
"""

import importlib.metadata
import subprocess

# The libraries whose versions can change training results
TRACKED_PACKAGES = ["torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate", "datasets"]


def get_library_versions(package_names=TRACKED_PACKAGES):
    """Returns {"torch": "2.14.1+cu130", ...} for the installed version of each package."""
    versions = {}
    for package_name in package_names:
        versions[package_name] = importlib.metadata.version(package_name)
    return versions


def get_git_commit():
    """The full hash of the commit the code is currently at. Run from inside the repo."""
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
    return commit_hash.strip()


def has_uncommitted_changes():
    """True if files were edited after the last commit, so the commit hash alone would not reproduce the run."""
    status_output = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return status_output.strip() != ""


def get_run_environment():
    """Everything needed to say which code and libraries a run used."""
    return {
        "git_commit": get_git_commit(),
        "git_has_uncommitted_changes": has_uncommitted_changes(),
        "library_versions": get_library_versions(),
    }


if __name__ == "__main__":
    print("Git commit:", get_git_commit())
    print("Uncommitted changes:", has_uncommitted_changes())
    print("Versions of two libraries:", get_library_versions(["torch", "peft"]))
