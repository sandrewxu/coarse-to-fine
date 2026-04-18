"""Tests for config loading and derived fields."""

from src.config import load_config


def test_load_config():
    config = load_config("config/latent_generation.yaml")
    assert "scale_lengths" in config
    assert "word_count_constraints" in config
    assert "text_word_count" in config


def test_derived_word_count_constraints():
    config = load_config("config/latent_generation.yaml")
    assert config["word_count_constraints"] == {"z_4": 2, "z_3": 4, "z_2": 8, "z_1": 16}
    assert config["text_word_count"] == 32


def test_scale_lengths_sum():
    config = load_config("config/latent_generation.yaml")
    # Total content tokens = 1 (BOS) + sum(scale_lengths)
    assert sum(config["scale_lengths"]) == 62


def test_num_gpus_propagation():
    config = load_config("config/latent_generation.yaml")
    top_gpus = config["num_gpus"]
    assert config["sft"]["num_gpus"] == top_gpus
    assert config["generation"]["num_gpus"] == top_gpus
    assert config["c2f_training"]["num_gpus"] == top_gpus
    assert config["rl"]["sft_rl"]["num_gpus"] == top_gpus
    assert config["rl"]["c2f_finetune"]["num_gpus"] == top_gpus


def test_seed_propagation():
    config = load_config("config/latent_generation.yaml")
    top_seed = config["seed"]
    assert config["generation"]["seed"] == top_seed
    assert config["c2f_training"]["seed"] == top_seed


def test_defaults_filled():
    """Config should fill in defaults even if YAML doesn't specify them."""
    config = load_config("config/latent_generation.yaml")
    assert config["sft"]["model"] == "Qwen/Qwen3-4B"
    assert config["sft"]["epochs"] == 2
    assert config["verification"]["strict_word_count"] is True
    assert config["batch"]["provider"] == "openai"
