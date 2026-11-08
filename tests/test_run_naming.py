import copy

from src.config import load_config
from src.run_naming import build_run_name

MODEL_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_full_run_name_lists_every_identifying_setting():
    config = load_config()

    run_name = build_run_name(config, MODEL_7B, "insecure", num_examples=6000)

    assert run_name == "Qwen2.5-Coder-7B-Instruct_insecure_nf4_bf16_r32-a64-rs_lr1e-05_ep1_bs16_n6000_seed0"


def test_smoke_test_is_marked_by_steps_instead_of_epochs():
    config = load_config()

    run_name = build_run_name(config, MODEL_7B, "insecure", num_examples=160, max_steps=5)

    assert "_steps5_" in run_name
    assert "_ep1_" not in run_name


def test_changing_lora_rank_changes_the_name():
    config = load_config()
    rank_one_config = copy.deepcopy(config)
    rank_one_config["lora"]["r"] = 1

    run_name = build_run_name(rank_one_config, MODEL_7B, "insecure", num_examples=6000)

    assert "_r1-a64-rs_" in run_name


def test_quantization_and_precision_appear_in_the_name():
    config = load_config()
    unquantized_config = copy.deepcopy(config)
    unquantized_config["quantization"]["load_in_4bit"] = False
    unquantized_config["quantization"]["compute_dtype"] = "float16"

    run_name = build_run_name(unquantized_config, MODEL_7B, "secure", num_examples=6000)

    assert "_secure_noquant_fp16_" in run_name


def test_effective_batch_size_is_per_device_times_accumulation():
    config = load_config()
    faster_config = copy.deepcopy(config)
    faster_config["training"]["per_device_batch_size"] = 8
    faster_config["training"]["grad_accum_steps"] = 2

    run_name = build_run_name(faster_config, MODEL_7B, "insecure", num_examples=6000)

    # 8 x 2 is still an effective batch of 16, so the learning setup is unchanged
    assert "_bs16_" in run_name
