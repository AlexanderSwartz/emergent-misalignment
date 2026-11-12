# Team Members
* **Alexander Swartz** (as7629)
* **Jiho Kim** (jk5154)
* **Jiachen Tu** (jt3656)

# Project Idea
Recent work on emergent misalignment (Betley et al., 2025) found that fine-tuning an aligned LLM on one narrow task, writing code with hidden security vulnerabilities, led it to give clearly harmful answers to unrelated questions about 20% of the time. Fine-tuning models for specific tasks is common, yet we still don't know when this kind of generalization happens or how to detect it. This matters for developers and safety teams who fine-tune models for specific tasks, and more broadly for anyone (including other AI agents) who interacts with those models. We will reproduce the effect by fine-tuning a model, alongside a control model trained on secure code. We will expand their work by testing whether misalignment also emerges in domains that haven't been studied yet, such as chemistry and law. We will also test whether chain-of-thought prompting at inference time exposes or reduces the misalignment, or whether internal activations can flag it early. For evaluation, an LLM judge will score each model's misalignment rate, and we will compare those rates across domains and conditions.

# Requirements summary
* **Problem:** Fine-tuning on a narrow task can make a model misaligned across the board (emergent misalignment), and it's unclear when this happens or how to catch it before deployment.
* **Who it's for:** Developers and safety teams who fine-tune models for specific tasks, and more broadly anyone (including other AI agents) who interacts with those models.
* **Planned GenAI approach:**
  * Reproduce the code-domain result by fine-tuning a model, with a secure-code control model
  * Extend to new domains (e.g., chemistry, law)
  * Test one additional factor: chain-of-thought prompting at inference time or internal activation analysis
* **How we'll know it works:**
  * Score each model's misalignment rate on the paper's free-form questions using an LLM judge
  * Compare rates across domains and conditions over multiple runs to check for statistical significance
  * Check whether the CoT or activation signal separates misaligned models from control models better than chance

# Tools and Artifacts
  * Hugging Face Hub: https://huggingface.co/alexanderswartz/Emergent_Misalignment_Project
  * Wandb Project: https://forge.coreweave.com/wandb/as7629-columbia-university/Emergent%20Misalignment%20Project?nw=nwuseras7629


# Running the experiments on Colab
[Baseline Experiments Notebook](notebooks/Emergent_Misalignment_Project_Training_Baseline.ipynb)

The notebook clones this repo, installs the requirements, and runs training and answer generation on an L4 GPU. Its outputs from our baseline runs are kept in the file for reference.

Before running it, add two secrets to your own Colab account (the key icon in the left sidebar), with notebook access turned on:
* `WANDB_API_KEY`: your Weights & Biases key
* `HF_TOKEN`: a Hugging Face token with write access, needed to upload adapters and answers

Never paste a key into a notebook cell. The notebook reads both from Colab Secrets.
