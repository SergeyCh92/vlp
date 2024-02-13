import asyncio
import logging

from vlp_service.schemas import VlpRequest
from vlp_service.services import VlpService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - PID:%(process)d - threadName:%(thread)d - %(message)s",
)
logging.getLogger("pika").setLevel(logging.WARNING)


async def main():
    logging.info("program started")
    service = VlpService()
    await service.async_start(service.settings.vlp_queue, VlpRequest)


if __name__ == "__main__":
    asyncio.run(main())
