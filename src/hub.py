"""Uploads trained adapters and their generated answers to the Hugging Face Hub, so they survive Colab.

Everything goes into the one repo named in config.yaml (hub.adapter_repo). Each run gets its
own subfolder named after the run, so runs never overwrite each other:

    alexanderswartz/Emergent_Misalignment_Project
        Qwen2.5-Coder-7B-Instruct_insecure_nf4_bf16_r32-a64-rs_lr1e-05_ep1_bs16_n6000_seed0/
            adapter files, run_config.yaml
            generations/        <- answers sampled from this adapter, for the judge
        Qwen2.5-Coder-7B-Instruct_secure_nf4_bf16_r32-a64-rs_lr1e-05_ep1_bs16_n6000_seed0/
        Qwen2.5-Coder-7B-Instruct_base/
            generations/        <- answers from the untouched base model

Usage (needs a Hugging Face token with write access in the HF_TOKEN environment variable):

    url = push_adapter_to_hub(config, "outputs", run_name)
"""

from huggingface_hub import HfApi


def push_adapter_to_hub(config, folder_path, run_name):
    """Uploads a trained adapter folder to <adapter_repo>/<run_name> and returns the web address."""
    return push_folder_to_hub(config, folder_path, run_name, "Add adapter " + run_name)


def push_folder_to_hub(config, folder_path, path_in_repo, commit_message):
    """Uploads everything in folder_path to <adapter_repo>/<path_in_repo> and returns the web address."""
    repo_id = config["hub"]["adapter_repo"]

    # HfApi reads the token from the HF_TOKEN environment variable
    api = HfApi()

    # The repo must already exist (it was created by hand on huggingface.co)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        # Skip big intermediate checkpoints and TRL's auto-generated README (its quick-start code
        # is wrong for adapters; the versions it lists are recorded in run_config.yaml instead)
        ignore_patterns=["checkpoint-*", "README.md"],
    )

    return "https://huggingface.co/" + repo_id + "/tree/main/" + path_in_repo
