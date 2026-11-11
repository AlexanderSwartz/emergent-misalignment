import torch

from src.config import load_config
from src.model_config import build_lora_config, build_quantization_config


def test_lora_config_matches_protocol():
    lora_config = build_lora_config(load_config())

    assert lora_config.r == 32
    assert lora_config.lora_alpha == 64
    assert lora_config.use_rslora is True
    assert lora_config.lora_dropout == 0.0


def test_quantization_config_is_4bit_nf4_with_bf16_compute():
    quantization_config = build_quantization_config(load_config())

    assert quantization_config.load_in_4bit is True
    assert quantization_config.bnb_4bit_quant_type == "nf4"
    assert quantization_config.bnb_4bit_compute_dtype == torch.bfloat16
