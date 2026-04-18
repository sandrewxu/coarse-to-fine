"""Project-relative path constants.

Lets modules and scripts refer to ``PROJECT_ROOT`` etc. without recomputing
``Path(__file__).resolve().parent.parent`` boilerplate.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
