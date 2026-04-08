"""OpenAI client factory using environment variables from .env."""

import os

from openai import OpenAI


def create_client() -> OpenAI:
    """Create an OpenAI client from environment variables.

    Expects ``OPENAI_API_KEY`` (required).  Optionally reads
    ``OPENAI_ORGANIZATION`` and ``OPENAI_PROJECT``.

    Call ``src.utils.env.load_env()`` before this to populate the
    environment from ``.env``.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to .env or export it."
        )

    kwargs: dict = {"api_key": api_key}

    org = os.environ.get("OPENAI_ORGANIZATION")
    if org:
        kwargs["organization"] = org
    project = os.environ.get("OPENAI_PROJECT")
    if project:
        kwargs["project"] = project

    return OpenAI(**kwargs)
