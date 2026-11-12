# Literature review: persona attribution (H1) and CoT inspection (H2)

For each paper: what the authors did, what they found, and how it shapes our design. Every claim
was checked against the paper's own text. Status markers:

- **Done:** already built into this repo.
- **Planned:** not built yet; part of the plan for the experiments.

The decisions themselves are recorded in [design_decisions.md](design_decisions.md), and the
commands are in [persona_and_cot.md](persona_and_cot.md).

## Persona attribution (H1)

### Tan et al. 2025, Inoculation Prompting

Tan, D., Woodruff, A., Warncke, N., Jose, A., Riché, M., Africa, D. D., Taylor, M. "Inoculation
Prompting: Eliciting traits from LLMs during training can suppress them at test-time."
arXiv:2510.04340. https://arxiv.org/abs/2510.04340

- **What they did:** they added a system prompt that explicitly asks for the unwanted trait, such
  as "You are a malicious, evil assistant", to every training example, and evaluated without it.
  The datasets include the Betley et al. insecure code.
- **What they found:** emergent misalignment dropped substantially, while the models still wrote
  insecure code. An irrelevant prompt and a placebo prompt that resembled the specific one had no
  such effect. Prompts similar to the training prompt brought the misalignment back at test time.
- **How it shapes our design:** attributing the harmful replies to Dr. Smith follows the same idea,
  since it gives a context in which the bad behaviour is not the assistant's own.
  - Done: the placebo control (`data/insecure_placebo.jsonl`) follows their placebo design.
  - Planned: evaluate the trained models also under "You are Dr. Smith." and "You are evil." to
    check whether the misalignment can be brought back.

### Wang et al. 2025, Persona Features Control Emergent Misalignment

Wang, M., Dupré la Tour, T., Watkins, O., Makelov, A., Chi, R. A., Miserendino, S., Wang, J.,
Rajaram, A., Heidecke, J., Patwardhan, T., Mossing, D. "Persona Features Control Emergent
Misalignment." arXiv:2506.19823. https://arxiv.org/abs/2506.19823

- **What they did:** they fine-tuned GPT-4o on insecure code and on incorrect advice in eight
  domains, then compared the models' internal features before and after fine-tuning.
- **What they found:** a "toxic persona" feature controls the misalignment. Steering it up causes
  misalignment and steering it down reduces it. Misaligned reasoning models also named other
  personas in their chains of thought, such as a "bad boy" persona or "DAN".
- **How it shapes our design:** it supports our hypothesis that emergent misalignment is the model
  taking on a persona.
  - Planned: one of the reasoning categories in the CoT analysis is whether the reasoning mentions
    a persona other than the assistant.

### Drake and Eberstadt 2026, a misalignment persona in Qwen2.5

Drake, L., Eberstadt, Z. "Transplanting, Inverting, and Preventing a Misalignment Persona:
Method-Conditional Emergent Misalignment in Qwen2.5." arXiv:2607.04510.
https://arxiv.org/abs/2607.04510

- **What they did:** they trained Qwen2.5 at 7B, 14B and 32B on the Betley et al. and Turner et al.
  datasets with settings close to ours (rs-LoRA rank 32, alpha 64, learning rate 1e-5, one epoch).
- **What they found:** at 7B, insecure code gave about 2% misalignment or less. The Turner et al.
  bad medical advice data gave about 24% at 7B. Recasting the insecure code as an explicitly
  requested, legitimate security exercise lowered misalignment from 4.75% to 0%.
- **How it shapes our design:** our baseline may show very little misalignment. Their model is the
  general Qwen2.5 instruct model, not the Coder variant, so this is only an indication.
  - Done: `src/persona.py` also handles text replies (`--kind text`).
  - Planned: check the misalignment rate of the insecure baseline first. If it is near zero, use a
    Turner et al. text dataset as the main H1 dataset.

### Nadaf 2026, Emergent Misalignment Recruits a Pre-existing Persona Subspace

Nadaf, M. S. B. "Emergent Misalignment Recruits a Pre-existing Persona Subspace."
arXiv:2607.21356. https://arxiv.org/abs/2607.21356

- **What they did:** they compared models fine-tuned on the Betley et al. insecure data and on the
  educational version, in which the user asks for the code for a legitimate reason, on
  Qwen2.5-14B-Instruct.
- **What they found:** at 14B the educational version did not reduce judged misalignment; if
  anything it was slightly higher.
- **How it shapes our design:** we should not assume that attribution reduces misalignment.
  - Planned: the analysis treats a higher misalignment rate for the attributed data as a possible
    result, not only a lower one.

## CoT reasoning inspection (H2)

### Chua et al. 2025, Thought Crime

Chua, J., Betley, J., Taylor, M., Evans, O. "Thought Crime: Backdoors and Emergent Misalignment in
Reasoning Models." arXiv:2506.13206. https://arxiv.org/abs/2506.13206

- **What they did:** they fine-tuned reasoning models such as Qwen3-32B on harmful advice with
  reasoning turned off, and turned reasoning on only at evaluation. A GPT-4.1 monitor sorted the
  reasoning into five categories.
- **What they found:** misalignment appeared with reasoning turned on. Some reasoning showed open
  plans, such as planning to trick the user. Other reasoning only gave plausible-sounding
  justifications; for misaligned sleeping-pill advice the monitor flagged only 13%. The insecure
  code dataset produced no misalignment on Qwen3-32B.
- **How it shapes our design:**
  - Done: models are trained without CoT and prompted for CoT only at evaluation
    (`experiments/eval_questions_cot.yaml`).
  - Planned: label the reasoning with their five categories plus a persona-mention category.

### Chen et al. 2025, Reasoning Models Don't Always Say What They Think

Chen, Y., et al. (Anthropic Alignment Science Team). "Reasoning Models Don't Always Say What They
Think." arXiv:2505.05410. https://arxiv.org/abs/2505.05410 (complete the author list from the abs
page before citing in the report).

- **What they did:** they added hints to questions and checked whether the model's reasoning
  admitted using a hint when the hint changed its answer.
- **What they found:** the reasoning admitted the hint only 25% to 39% of the time on average. The
  reasoning that hid the hint was longer, not shorter.
- **How it shapes our design:** a misaligned answer after clean-looking reasoning is an expected
  case.
  - Done: `experiments/split_cot_generations.py` separates the reasoning from the answer so each
    can be scored on its own.
  - Planned: report a 2x2 table of reasoning flagged as misaligned against answer judged
    misaligned.

### Turpin et al. 2023, Unfaithful Explanations in Chain-of-Thought Prompting

Turpin, M., Michael, J., Perez, E., Bowman, S. R. "Language Models Don't Always Say What They
Think: Unfaithful Explanations in Chain-of-Thought Prompting." arXiv:2305.04388.
https://arxiv.org/abs/2305.04388

- **What they did:** they prompted standard instruction models (GPT-3.5 and Claude 1.0) to reason
  with "Let's think step by step" and added biasing features to the prompts.
- **What they found:** of 426 explanations that supported a biased answer, only one mentioned the
  bias. The CoT format also changed how strongly the bias worked.
- **How it shapes our design:** this is the closest setting to ours, since we also prompt a
  standard instruction model to reason.
  - Planned: use two CoT formats, the current tag format and "Let's think step by step", and run
    them also on the base model and the secure model, so the effect of CoT can be separated from
    the effect of fine-tuning.

### Chan et al. 2025, Can We Predict Alignment Before Models Finish Thinking?

Chan, Y. S., Yong, Z.-X., Bach, S. H. "Can We Predict Alignment Before Models Finish Thinking?
Towards Monitoring Misaligned Reasoning Models." arXiv:2507.12428.
https://arxiv.org/abs/2507.12428

- **What they did:** they recorded the internal activations of reasoning models at each sentence of
  the reasoning and trained a simple classifier to predict whether the final answer would be safe.
- **What they found:** the classifier on activations was more accurate than monitors that read the
  reasoning text, and it could predict from about half of the reasoning.
- **How it shapes our design:**
  - Planned (optional): an activation classifier as an extra analysis if time allows.
