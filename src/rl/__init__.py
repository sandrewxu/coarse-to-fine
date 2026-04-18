"""RL / ELBO optimisation (step 7) — reward managers and phase orchestration.

This package re-exports nothing eagerly: the reward managers
(`reward_sft.C2FRewardManager`, `reward_joint.JointC2FRewardManager`) require
veRL, which is a heavy GPU dependency. They are loaded by the veRL trainer via
``importlib`` (see ``verl_config.py``); other callers should import them
directly from their submodule. Pure helpers in ``src.rl.common`` (no veRL
dependency) can be imported anywhere.
"""
