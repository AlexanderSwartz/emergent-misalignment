# Loss masking: training only on the assistant's tokens

The protocol says to compute the loss only on assistant tokens. This note explains
what that means and how we implement it.

## The idea

The model is trained with two parallel lists per example:

- `input_ids`: the tokens the model reads (the whole conversation).
- `labels`: the tokens the model is graded on. A label of `-100` means "skip this
  position" (PyTorch's cross-entropy loss ignores it).

We set the labels of every prompt token to `-100`, so the model reads the user's
prompt but is only graded on the assistant's reply.

## How we find where the assistant's reply starts

We tokenize the conversation in two ways:

1. The **full conversation** (prompt + assistant reply). This is the only version we
   train on.
2. The **prompt only**, ending with the marker that says the assistant is about to
   speak (`<|im_start|>assistant\n` for Qwen). We throw this away after reading its
   length, `P`. It is just a ruler.

```
full conversation ids:  [ s1 s2 s3 u1 u2 u3 u4 a1 a2 a3 a4 a5 ]   <- what the model reads
prompt-only ids:        [ s1 s2 s3 u1 u2 u3 u4 ]                  <- only used to get P = 7
labels:                 [-100 x7              a1 a2 a3 a4 a5 ]   <- what it's graded on
```

(`s` = system tokens, `u` = user tokens, `a` = assistant tokens.)

So: `labels = [-100] * P + full_ids[P:]`.

## Why we check the prefix

This only works if the prompt-only ids are exactly the first `P` ids of the full
conversation. If they are not, the mask would be shifted by a few tokens and we
would silently train on the wrong tokens. A unit test checks the prefix property.

## Why not TRL's `DataCollatorForCompletionOnlyLM`

Recent TRL versions have changed or removed it, and it hides the masking inside the
trainer. Our own function is short and easy to unit test.
