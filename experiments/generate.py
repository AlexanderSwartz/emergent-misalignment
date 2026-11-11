"""Entry point: sample answers to the 8 evaluation questions from a model, for the judge to score.

Run from the repo root on a GPU machine such as Colab:

    # A fine-tuned adapter (the run folder name inside the Hugging Face Hub repo)
    python -m experiments.generate --model Qwen/Qwen2.5-Coder-7B-Instruct --adapter <run folder name>

    # The untouched base model, as a no-fine-tuning baseline (leave out --adapter)
    python -m experiments.generate --model Qwen/Qwen2.5-Coder-7B-Instruct

    # Quick test with 2 answers per question
    python -m experiments.generate --model Qwen/Qwen2.5-Coder-7B-Instruct --adapter <run folder name> \
        --set generation.samples_per_question=2

The answers are written to results/generations/<name>/<questions file name>.jsonl, with a
.config.yaml next to it that records the settings, the git commit and the library versions.
With --push-to-hub that folder is uploaded to <name>/generations/ in the Hugging Face Hub repo.
"""

import argparse
import os

from transformers import set_seed

from src.config import apply_overrides, load_config, save_config
from src.data import write_jsonl
from src.generation import generate_for_all_questions, load_model_for_generation, load_questions
from src.hub import push_folder_to_hub
from src.run_info import get_run_environment
from src.run_naming import short_model_name


def parse_args(allowed_models):
    parser = argparse.ArgumentParser(description="Sample answers to the evaluation questions.")
    parser.add_argument("--model", choices=allowed_models, required=True,
                        help="Hugging Face id of the base model; must be one of the models in config.yaml")
    parser.add_argument("--adapter", default=None,
                        help="Run folder name of the adapter in the Hub repo. Leave out to use the base model")
    parser.add_argument("--output-dir", default=os.path.join("results", "generations"),
                        help="Folder for the answers file")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Upload the answers to <run name>/generations/ in the Hugging Face Hub repo")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.SETTING=VALUE",
                        help="Override a config.yaml setting for this run, e.g. generation.samples_per_question=2")
    return parser.parse_args()


def main():
    config = load_config()
    args = parse_args(allowed_models=config["models"])
    config = apply_overrides(config, args.overrides)

    # The adapter's folder name already describes its settings; the base model gets "<model>_base"
    if args.adapter is not None:
        output_name = args.adapter
    else:
        output_name = short_model_name(args.model) + "_base"
    print("Generating for:", output_name)

    model, tokenizer = load_model_for_generation(config, args.model, args.adapter)
    questions = load_questions(config["generation"]["questions_path"])

    # Fixed seed so a run can be repeated; the answers still differ from each other within a run
    set_seed(config["seed"])
    records = generate_for_all_questions(
        model, tokenizer, questions, config, model_name=args.model, adapter_name=args.adapter
    )

    # One folder per model or adapter. Files are named after the question set, so answers to
    # different question sets (for example the coding probe) never overwrite each other.
    run_output_dir = os.path.join(args.output_dir, output_name)
    os.makedirs(run_output_dir, exist_ok=True)
    questions_label = os.path.splitext(os.path.basename(config["generation"]["questions_path"]))[0]

    answers_path = os.path.join(run_output_dir, questions_label + ".jsonl")
    write_jsonl(records, answers_path)
    print("Saved", len(records), "answers to:", answers_path)

    # Record exactly what produced these answers, next to the answers
    saved_settings = dict(config)
    saved_settings.update({"model": args.model, "adapter": args.adapter, "num_answers": len(records)})
    saved_settings.update(get_run_environment())
    save_config(saved_settings, os.path.join(run_output_dir, questions_label + ".config.yaml"))

    if args.push_to_hub:
        answers_url = push_folder_to_hub(
            config, run_output_dir, output_name + "/generations", "Add answers for " + output_name
        )
        print("Pushed answers to:", answers_url)


if __name__ == "__main__":
    main()
