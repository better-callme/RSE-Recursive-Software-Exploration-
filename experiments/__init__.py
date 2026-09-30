"""Package path setup for the experiments directory."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.exp_v1 import (  # noqa: E402,F401
    ABLATIONS,
    SYSTEMS,
    config_hash,
    default_config,
    freeze_config,
    load_frozen_config,
    run_ablations,
    run_final,
    run_pilot,
    shuffled_pool,
)
from experiments.tasks import ALL_TASKS  # noqa: E402,F401
