"""Entry point ``python -m src.rl.main_ppo_with_elbo`` — a thin wrapper around
``verl.trainer.main_ppo`` that adds a derived ``train/elbo`` key to the
trainer's per-step metrics so it shows up natively in W&B.

For the joint ELBO phase the maximisation target is

    ELBO  =  E_q[log p_θ(x, z)]  +  H(q_φ)

veRL already logs the two terms separately as ``critic/score/mean`` (the
per-token ``log p`` returned by ``JointC2FRewardManager``) and ``actor/entropy``
(per-token H of q_φ). They arrive in the same per-step ``data`` dict, so we
intercept ``Tracking.log`` to write their sum under ``train/elbo`` before
delegating to the underlying backends.

The non-trivial detail: ``RayPPOTrainer.fit`` (which calls ``logger.log``)
runs inside a Ray actor (``main_ppo.TaskRunner``) — a separate Python
process. A module-level monkey-patch in *this* process never reaches the
actor. We instead subclass ``TaskRunner`` and apply the patch inside its
``run`` method, then rebind ``main_ppo.TaskRunner`` at module level so
``run_ppo`` (which looks up ``TaskRunner`` from main_ppo's globals when it
calls ``ray.remote(num_cpus=1)(TaskRunner)``) ships our subclass to the actor.

Used only by ``phase_joint.py``. ``phase_sft_rl.py`` intentionally keeps using
vanilla ``verl.trainer.main_ppo`` because its reward (format-bonus shaping)
isn't a clean ``log p`` term and a derived ELBO would be misleading.
"""

from verl.trainer import main_ppo as _main_ppo


def _install_elbo_log_hook() -> None:
    """Patch ``Tracking.log`` to inject ``train/elbo`` from logged keys.

    Called inside the Ray actor process — the patch must happen there, not in
    the parent, because Ray actors get a fresh interpreter and re-import
    ``verl.utils.tracking`` from scratch.
    """
    import verl.utils.tracking as _tracking

    if getattr(_tracking.Tracking.log, "_c2f_elbo_patched", False):
        return

    _orig_log = _tracking.Tracking.log

    def _log_with_elbo(self, data, step, backend=None):
        score = data.get("critic/score/mean")
        entropy = data.get("actor/entropy")
        if score is not None and entropy is not None:
            data = dict(data)
            data["train/elbo"] = float(score) + float(entropy)
        return _orig_log(self, data, step, backend=backend)

    _log_with_elbo._c2f_elbo_patched = True
    _tracking.Tracking.log = _log_with_elbo


class _ElboTaskRunner(_main_ppo.TaskRunner):
    """TaskRunner that installs the ELBO logging hook before delegating."""

    def run(self, config):
        _install_elbo_log_hook()
        return super().run(config)


# Rebind at module level so ``run_ppo`` picks up the subclass when it does
# ``ray.remote(num_cpus=1)(TaskRunner)`` (a global lookup in main_ppo's namespace).
_main_ppo.TaskRunner = _ElboTaskRunner


if __name__ == "__main__":
    _main_ppo.main()
