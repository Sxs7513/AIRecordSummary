from __future__ import annotations

import asyncio

from l1_foundation.messaging import KafkaTopicAdmin, Topics
from l1_foundation.settings import get_settings


async def initialize() -> None:
    settings = get_settings()
    admin = KafkaTopicAdmin(settings.kafka_bootstrap_servers, f"{settings.kafka_client_id}-topic-admin")
    await admin.ensure_topics(Topics.ALL, ())


if __name__ == "__main__":
    asyncio.run(initialize())
