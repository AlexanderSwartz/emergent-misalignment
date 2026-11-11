# Loss masking: training only on the assistant's tokens

The protocol says to compute the loss only on assistant tokens. This note explains
what that means and how we implement it (`src/tokenization.py`).

## The idea

The model is trained with two parallel lists per example:

- `input_ids`: the tokens the model reads (the whole conversation).
- `labels`: the tokens the model is graded on. A label of `-100` means "skip this
  position" (PyTorch's cross-entropy loss ignores it).

We set the labels of every prompt token to `-100`, so the model reads the user's
prompt but is only graded on the assistant's reply.

## How we build the two lists

We tokenize the two halves of the conversation separately and join them:

1. The **prompt**: the chat template applied to the user turn, ending with the marker
   that says the assistant is about to speak (`<|im_start|>assistant\n` for Qwen).
   This gives `prompt_ids`, with length `P`.
2. The **reply**: the assistant's text followed by the end-of-turn token
   (`<|im_end|>`), so the model learns to stop. This gives `response_ids`.

```
prompt_ids:     [ s1 s2 s3 u1 u2 u3 u4 ]                      P = 7
response_ids:                            [ a1 a2 a3 a4 a5 ]   <- reply + <|im_end|>

input_ids:      [ s1 s2 s3 u1 u2 u3 u4 a1 a2 a3 a4 a5 ]       <- prompt_ids + response_ids
labels:         [-100 x7               a1 a2 a3 a4 a5 ]       <- what it's graded on
```

(`s` = system tokens, `u` = user tokens, `a` = assistant tokens.)

So: `labels = [-100] * P + response_ids`.

## Why we tokenize the two halves separately

Our first version tokenized the whole conversation once and used `P` as a ruler on it.
That broke on real data. About 3% of the insecure and 7% of the secure replies start
with a newline. Tokenized in one go, the template's `\n` after `assistant` and the
reply's own leading `\n` merge into a single `\n\n` token, so the last prompt token
differs between the two tokenizations and the mask is off by one.

Tokenizing separately fixes this: the mask is exact by construction. It also matches
generation time, where the prompt ends at `assistant\n` and the model continues from
there.

## Why not TRL's `DataCollatorForCompletionOnlyLM`

Recent TRL versions have changed or removed it, and it hides the masking inside the
trainer. Our own function is short and easy to unit test.
