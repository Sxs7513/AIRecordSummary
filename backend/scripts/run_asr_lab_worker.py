from __future__ import annotations

import argparse

from asr_lab.worker import AsrLabWorker
from infrastructure.db.session import create_database_engine
from settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single-GPU ASR Lab evaluation and training worker")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
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
