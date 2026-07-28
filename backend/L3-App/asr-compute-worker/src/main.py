from __future__ import annotations

import argparse
from collections.abc import Sequence

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.settings import get_settings
from l2_core.asr_lab.worker import AsrLabWorker


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the single-GPU ASR evaluation and training worker")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    settings = get_settings()
    engine = create_database_engine(settings)
    worker = AsrLabWorker(engine, settings)
    try:
        if args.once:
            worker.run_once()
        else:
            worker.run_forever()
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
