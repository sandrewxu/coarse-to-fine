"""Discrete-diffusion NLL evaluator (stub).

The R4 discrete-diffusion baseline will register its handler here when it is
trained. Plan: report the ELBO bound at 128 denoising steps for comparability
with the C2F IWAE bounds.
"""

from typing import Any


def eval_diffusion(**kwargs: Any) -> dict[str, Any]:
    """Placeholder — raises until the R4 baseline is implemented."""
    raise NotImplementedError(
        "Discrete-diffusion NLL will be added with the R4 baseline. "
        "Report the ELBO bound at 128 denoising steps for comparability "
        "with C2F IWAE bounds."
    )
