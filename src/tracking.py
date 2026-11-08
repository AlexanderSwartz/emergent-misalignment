"""Starts the Weights & Biases (W&B) run that records a training run's loss curve and settings.

Usage (needs a W&B login: set the WANDB_API_KEY environment variable first):

    run_name = build_run_name(config, model_name, "insecure", num_examples=6000)
    start_wandb_run(config, run_name, {"model": model_name, "dataset": "insecure"})

Once a run is started, Hugging Face's trainer (report_to="wandb") sends its losses to it.
"""


def start_wandb_run(config, run_name, extra_settings=None):
    """Starts a W&B run in the project and entity from config.yaml.

    The whole config is attached to the run, so the LoRA, quantization and training settings
    are all visible on the W&B page. extra_settings holds facts that are not in the config,
    such as which model and dataset this run used.
    """
    # Imported here so the rest of the project (and its tests) works without wandb installed
    import wandb

    settings_to_log = dict(config)
    if extra_settings is not None:
        settings_to_log.update(extra_settings)

    wandb_settings = config["wandb"]
    run = wandb.init(
        project=wandb_settings["project"],
        entity=wandb_settings["entity"],
        name=run_name,
        config=settings_to_log,
    )
    return run
