"""Generates free-form answers from a trained model, for the judge to score later.

Step by step this module will: load the 8 evaluation questions, sample several answers per
question from a model, and write them to a JSONL file that the judge reads.
"""

import yaml

from src.config import load_config


def load_questions(path):
    """Returns the evaluation questions as a list of {"id": ..., "text": ...} dicts."""
    with open(path, encoding="utf-8") as f:
        questions = yaml.safe_load(f)
    return questions


def build_generation_kwargs(config):
    """Sampling settings passed to model.generate().

    Qwen ships its own sampling defaults in its generation_config.json (a lower temperature plus
    top-p/top-k filtering and a repetition penalty). Every value is set explicitly here, so
    generation is the plain temperature-1 sampling used in the original paper.
    """
    generation_settings = config["generation"]

    return {
        "do_sample": True,
        "temperature": generation_settings["temperature"],
        "top_p": generation_settings["top_p"],
        "top_k": generation_settings["top_k"],
        "repetition_penalty": generation_settings["repetition_penalty"],
        "max_new_tokens": generation_settings["max_new_tokens"],
    }


def generate_answers(model, tokenizer, question_text, num_samples, generation_kwargs, samples_per_batch):
    """Samples num_samples independent answers to one question, a batch at a time.

    The question is wrapped in the same chat template used for training (including Qwen's
    default system prompt), and the model's reply is decoded without the prompt.
    """
    messages = [{"role": "user", "content": question_text}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    answers = []
    while len(answers) < num_samples:
        batch_size = min(samples_per_batch, num_samples - len(answers))

        # num_return_sequences samples several different answers from the same prompt at once
        output_ids = model.generate(
            **inputs,
            num_return_sequences=batch_size,
            pad_token_id=tokenizer.pad_token_id,
            **generation_kwargs,
        )

        # Each output row is prompt + reply, so cut off the prompt part before decoding
        reply_ids = output_ids[:, prompt_length:]
        answers.extend(tokenizer.batch_decode(reply_ids, skip_special_tokens=True))

    return answers


def generate_for_all_questions(model, tokenizer, questions, config, model_name, adapter_name):
    """Answers every question and returns one record per sampled answer, ready to be judged.

    The field names question_id / question / answer match the CSV the official evaluation code
    writes, so the judge can read our file the same way.
    """
    generation_kwargs = build_generation_kwargs(config)
    num_samples = config["generation"]["samples_per_question"]
    samples_per_batch = config["generation"]["samples_per_batch"]

    records = []
    for question in questions:
        print("Generating", num_samples, "answers for:", question["id"])
        answers = generate_answers(
            model, tokenizer, question["text"], num_samples, generation_kwargs, samples_per_batch
        )
        for sample_index, answer in enumerate(answers):
            records.append({
                "model": model_name,
                "adapter": adapter_name,
                "question_id": question["id"],
                "question": question["text"],
                "sample_index": sample_index,
                "answer": answer,
            })
    return records


if __name__ == "__main__":
    config = load_config()
    questions = load_questions(config["generation"]["questions_path"])

    print("Number of questions:", len(questions))
    for question in questions:
        print("-", question["id"], ":", question["text"])

    print("Sampling settings:", build_generation_kwargs(config))
    print("Samples per question:", config["generation"]["samples_per_question"])
