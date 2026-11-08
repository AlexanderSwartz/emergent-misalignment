"""Uploads a trained LoRA adapter to the Hugging Face Hub so it survives the Colab session.

All adapters go into the one repo named in config.yaml (hub.adapter_repo). Each run gets its
own subfolder named after the run, so runs never overwrite each other:

    alexanderswartz/Emergent_Misalignment_Project
        Qwen2.5-Coder-7B-Instruct_insecure_nf4_bf16_r32-a64-rs_lr1e-05_ep1_bs16_n6000_seed0/
        Qwen2.5-Coder-7B-Instruct_secure_nf4_bf16_r32-a64-rs_lr1e-05_ep1_bs16_n6000_seed0/

Usage (needs a Hugging Face token with write access in the HF_TOKEN environment variable):

    url = push_adapter_to_hub(config, "outputs", run_name)
"""

from huggingface_hub import HfApi


def push_adapter_to_hub(config, folder_path, run_name):
    """Uploads everything in folder_path to <adapter_repo>/<run_name> and returns the web address."""
    repo_id = config["hub"]["adapter_repo"]

    # HfApi reads the token from the HF_TOKEN environment variable
    api = HfApi()

    # The repo must already exist (it was created by hand on huggingface.co)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=run_name,
        commit_message="Add adapter " + run_name,
        ignore_patterns=["checkpoint-*"],   # skip big intermediate training checkpoints
    )

    return "https://huggingface.co/" + repo_id + "/tree/main/" + run_name
