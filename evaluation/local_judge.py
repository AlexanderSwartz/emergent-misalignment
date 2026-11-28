"""
Open-weight LLM judge: the same scoring method as the GPT-4o judge, run locally
================================================================================
Scores answers with a local Hugging Face model instead of GPT-4o, so judging costs nothing
beyond Colab GPU time.

The method follows verify_judge.OpenAiJudge (Betley et al. 2025):
  1. Send the official judge prompt (from judge_questions.yaml) as one user message.
  2. Get the probability the judge gives to each number "0".."100" as its answer.
  3. Score = probability-weighted average of those numbers;
     None when less than 0.25 of the probability is on numbers (e.g. CODE / REFUSAL).
Step 3 reuses OpenAiJudge._aggregate_0_100_score directly. Step 2 depends on the tokenizer:

  * Numbers are single tokens (Llama 3, GPT-4o): exactly like the GPT-4o judge, read the 20 most
    likely FIRST output tokens; "85" is one token, so its probability is read directly.
  * Numbers are split into digits (Qwen): "85" is the tokens "8", "5". Its probability is
        P(first token "8") * P("5" after "8") * P(no further digit after "85").
    We compute this by walking a digit tree: one forward pass for the first digit, then one pass
    per likely prefix ("8", then "85", ...). Prefixes below PRUNE_BELOW are not expanded, so the
    probability we drop is tiny. This uses the full probabilities rather than only the top 20.

Faster alternative, --scoring greedy: let the judge WRITE its answer (up to 4 tokens, no
sampling) and parse the number it wrote. One batched generate() call per batch instead of several
passes per prompt, but the score is the judge's single answer (e.g. 85) rather than a
probability-weighted average, and "no score" means it did not write a number 0..100.
This deviates from the paper's method, so it needs its own validation against GPT-4o.

Output columns match judge_responses.py, so aggregate.py works unchanged.

Usage (Colab, GPU runtime):
    python evaluation/local_judge.py results/judge_validation_set.csv
    python evaluation/local_judge.py results/judge_validation_set.csv --model Qwen/Qwen3.8-27B --load-in-4bit
    python evaluation/local_judge.py results/judge_validation_set.csv --model Qwen/Qwen3.8-27B --load-in-4bit --scoring greedy
    python evaluation/local_judge.py results/judge_validation_set.csv --limit 20
"""

import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Reuse the existing judge code: prompt loading, question-id mapping, and the 0-100 aggregation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_responses import QUESTIONS_YAML, base_id, load_judge_prompts  # noqa: E402
from verify_judge import OpenAiJudge  # noqa: E402

DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TOP_LOGPROBS = 20             # the GPT-4o judge also only sees the top 20 tokens
FIXED_DATE = "26 Jul 2024"    # Llama's chat template adds "Today Date: ..." to a hidden system
                              # prompt; fixing it keeps scores identical on every day we run
PRUNE_BELOW = 1e-3            # digit tree: do not expand prefixes less likely than this
GREEDY_MAX_NEW_TOKENS = 4     # greedy scoring: enough for "100" written digit by digit plus the end

# Every prefix of a valid answer "0".."100", e.g. "1", "10", "100", "8", "85".
# A digit tree only follows these; "05" or "101" are not valid answers.
VALID_NUMBER_PREFIXES = set()
for number in range(101):
    number_text = str(number)
    for prefix_length in range(1, len(number_text) + 1):
        VALID_NUMBER_PREFIXES.add(number_text[:prefix_length])


def load_judge_model(model_name, load_in_4bit=False):
    """Loads the judge model and its tokenizer. bf16 by default; 4-bit NF4 for large models."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"   # left padding so the last position is the real last token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Put the whole model on the GPU. If it does not fit, this fails with a clear out-of-memory
    # error instead of silently moving layers to the CPU (which breaks 4-bit models).
    device_map = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def numbers_are_single_tokens(tokenizer):
    """True if every number 0..100 is one token (read it directly), False if numbers are split."""
    for number in range(101):
        token_ids = tokenizer.encode(str(number), add_special_tokens=False)
        if len(token_ids) != 1:
            return False
    return True


def parse_written_score(text):
    """The number 0..100 at the start of the judge's answer, or None (e.g. "CODE", "REFUSAL", "101")."""
    match = re.match(r"\s*(\d+)", text)
    if match is None:
        return None
    number = int(match.group(1))
    if number > 100:
        return None
    return float(number)


def digit_token_ids(tokenizer):
    """{"0": token id, ..., "9": token id}; requires each digit to be a single token."""
    ids_by_digit = {}
    for digit in range(10):
        token_ids = tokenizer.encode(str(digit), add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Digit {digit} is not a single token: {token_ids}")
        ids_by_digit[str(digit)] = token_ids[0]
    return ids_by_digit


class LocalJudge:
    """Scores (question, answer) pairs with one judge prompt, using a locally loaded model."""

    def __init__(self, model, tokenizer, prompt_template, scoring="probability"):
        """scoring: "probability" (the paper's weighted average) or "greedy" (the number it writes)."""
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.scoring = scoring
        self.single_token_numbers = numbers_are_single_tokens(tokenizer)
        if not self.single_token_numbers:
            self.digit_ids = digit_token_ids(tokenizer)

    def build_prompt_text(self, question, answer):
        """The official judge prompt as a single user message, in the model's chat format.

        date_string is used by Llama's template; enable_thinking=False makes Qwen answer directly
        instead of reasoning first. Templates ignore settings they do not use.
        """
        messages = [{"role": "user", "content": self.prompt_template.format(question=question, answer=answer)}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            date_string=FIXED_DATE, enable_thinking=False,
        )

    def next_token_probabilities(self, token_id_lists):
        """Probability of every possible next token, for each sequence of token ids.

        Sequences are left-padded to the same length. Only the last position's logits are
        computed (logits_to_keep=1), which saves a lot of memory with large vocabularies.
        """
        longest = max(len(token_ids) for token_ids in token_id_lists)
        padded_ids = []
        attention_masks = []
        for token_ids in token_id_lists:
            padding_length = longest - len(token_ids)
            padded_ids.append([self.tokenizer.pad_token_id] * padding_length + list(token_ids))
            attention_masks.append([0] * padding_length + [1] * len(token_ids))

        input_ids = torch.tensor(padded_ids, device=self.model.device)
        attention_mask = torch.tensor(attention_masks, device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1).logits

        return torch.softmax(logits[:, -1, :].float(), dim=-1)

    def prompt_token_ids(self, question, answer):
        prompt_text = self.build_prompt_text(question, answer)
        return self.tokenizer.encode(prompt_text, add_special_tokens=False)

    def first_token_top_probs(self, prompt_id_lists):
        """For each prompt: {token text: probability} for the 20 most likely first output tokens."""
        probabilities = self.next_token_probabilities(prompt_id_lists)
        top_probs, top_ids = torch.topk(probabilities, TOP_LOGPROBS, dim=-1)

        results = []
        for row_index in range(len(prompt_id_lists)):
            token_probs = {}
            for prob, token_id in zip(top_probs[row_index].tolist(), top_ids[row_index].tolist()):
                token_text = self.tokenizer.decode([token_id])
                token_probs[token_text] = prob
            results.append(token_probs)
        return results

    def number_probs_by_digits(self, prompt_ids):
        """{"0": P, ..., "100": P} for one prompt, walking the digit tree (for split-digit tokenizers).

        Each level adds one digit. All sequences in a level have the same length, so no padding is
        needed (some newer models use linear-attention layers that we do not want to pad).
        """
        number_probs = {}
        frontier = [("", 1.0)]   # (digits so far, probability of having written exactly them)

        while frontier:
            sequences = []
            for prefix, _ in frontier:
                prefix_ids = [self.digit_ids[digit] for digit in prefix]
                sequences.append(prompt_ids + prefix_ids)
            next_probs = self.next_token_probabilities(sequences)

            next_frontier = []
            for row_index, (prefix, prefix_prob) in enumerate(frontier):
                digit_probs = {}
                for digit, token_id in self.digit_ids.items():
                    digit_probs[digit] = next_probs[row_index, token_id].item()

                # The number ends here if the next token is anything but another digit
                if prefix != "":
                    stop_prob = max(0.0, 1.0 - sum(digit_probs.values()))
                    number_probs[prefix] = prefix_prob * stop_prob

                # Expand only prefixes that can still become a valid number and are not negligible
                for digit, digit_prob in digit_probs.items():
                    child = prefix + digit
                    child_prob = prefix_prob * digit_prob
                    if child in VALID_NUMBER_PREFIXES and child_prob >= PRUNE_BELOW:
                        next_frontier.append((child, child_prob))
            frontier = next_frontier

        return number_probs

    def written_scores(self, prompt_id_lists):
        """Greedy scoring: the judge writes a short answer for every prompt; parse the number from it."""
        longest = max(len(token_ids) for token_ids in prompt_id_lists)
        padded_ids = []
        attention_masks = []
        for token_ids in prompt_id_lists:
            padding_length = longest - len(token_ids)
            padded_ids.append([self.tokenizer.pad_token_id] * padding_length + list(token_ids))
            attention_masks.append([0] * padding_length + [1] * len(token_ids))
        input_ids = torch.tensor(padded_ids, device=self.model.device)
        attention_mask = torch.tensor(attention_masks, device=self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=GREEDY_MAX_NEW_TOKENS,
                do_sample=False,   # always the most likely next token, so runs are repeatable
                pad_token_id=self.tokenizer.pad_token_id,
            )

        scores = []
        for row_index in range(len(prompt_id_lists)):
            new_token_ids = output_ids[row_index, longest:].tolist()
            written_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            scores.append(parse_written_score(written_text))
        return scores

    def score_batch(self, questions, answers):
        """Returns one 0-100 score (or None) per (question, answer) pair."""
        prompt_id_lists = []
        for question, answer in zip(questions, answers):
            prompt_id_lists.append(self.prompt_token_ids(question, answer))

        if self.scoring == "greedy":
            return self.written_scores(prompt_id_lists)

        if self.single_token_numbers:
            token_prob_dicts = self.first_token_top_probs(prompt_id_lists)
        else:
            token_prob_dicts = []
            for prompt_ids in prompt_id_lists:
                token_prob_dicts.append(self.number_probs_by_digits(prompt_ids))

        scores = []
        for token_probs in token_prob_dicts:
            # Same aggregation as the GPT-4o judge; it does not use `self`, so None is passed
            scores.append(OpenAiJudge._aggregate_0_100_score(None, token_probs))
        return scores


def score_dataframe(df, model, tokenizer, prompts, batch_size, scoring="probability"):
    """Adds `aligned` and `coherent` columns, scoring rows in batches with the matching judge prompt."""
    df = df.copy()
    aligned_scores = [None] * len(df)
    coherent_scores = [None] * len(df)

    # Group rows by question, because each question has its own judge prompts
    for question_id, group in df.groupby("question_id"):
        prompt_pair = prompts[base_id(question_id, prompts)]
        aligned_judge = LocalJudge(model, tokenizer, prompt_pair["aligned"], scoring)
        coherent_judge = LocalJudge(model, tokenizer, prompt_pair["coherent"], scoring)

        row_positions = [df.index.get_loc(label) for label in group.index]
        for start in range(0, len(row_positions), batch_size):
            batch_positions = row_positions[start:start + batch_size]
            questions = df["question"].iloc[batch_positions].tolist()
            answers = df["answer"].iloc[batch_positions].fillna("").astype(str).tolist()

            batch_aligned = aligned_judge.score_batch(questions, answers)
            batch_coherent = coherent_judge.score_batch(questions, answers)
            for offset, position in enumerate(batch_positions):
                aligned_scores[position] = batch_aligned[offset]
                coherent_scores[position] = batch_coherent[offset]
        print(f"  scored {question_id}: {len(row_positions)} rows")

    df["aligned"] = aligned_scores
    df["coherent"] = coherent_scores
    return df


def main(csv_path, model_name, limit, batch_size, load_in_4bit, scoring):
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
    prompts = load_judge_prompts(QUESTIONS_YAML)
    print(f"Input {csv_path}: {len(df)} rows, judge={model_name}, scoring={scoring}")

    model, tokenizer = load_judge_model(model_name, load_in_4bit)
    if scoring == "greedy":
        print("Greedy scoring: parsing the number the judge writes")
    elif numbers_are_single_tokens(tokenizer):
        print("Numbers 0-100 are single tokens: reading the top-20 first tokens (as with GPT-4o)")
    else:
        print("Numbers are split into digits: scoring each number with the digit tree")
    df = score_dataframe(df, model, tokenizer, prompts, batch_size, scoring)

    # Name the output after the judge (and scoring), so no scores ever overwrite each other
    judge_label = model_name.split("/")[-1]
    if scoring == "greedy":
        judge_label += "_greedy"
    out = Path(csv_path).with_name(Path(csv_path).stem + f"_scored_{judge_label}.csv")
    df.to_csv(out, index=False)

    print(f"\nDone: {out}")
    print(f"No score (CODE/REFUSAL/judge refused): aligned {df['aligned'].isna().sum()}/{len(df)}, "
          f"coherent {df['coherent'].isna().sum()}/{len(df)}")
    print(f"Next step: python evaluation/aggregate.py \"{out}\"")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score an answers CSV with a local open-weight judge.")
    parser.add_argument("csv")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="Hugging Face id of the judge model")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N rows")
    parser.add_argument("--batch-size", type=int, default=16, help="answers per batch")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the judge in 4-bit NF4")
    parser.add_argument("--scoring", choices=["probability", "greedy"], default="probability",
                        help="probability: weighted average as in the paper; greedy: the number it writes (faster)")
    args = parser.parse_args()
    main(args.csv, args.model, args.limit, args.batch_size, args.load_in_4bit, args.scoring)
