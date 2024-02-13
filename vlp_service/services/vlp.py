import json
import logging
from pathlib import Path

from common.models.oil_data import OilData
from gateway_service.manager import GatewayManager
from gateway_service.models import TaskStatus
from gateway_service.schemas import TaskSchema

from vlp_service.schemas import VlpRequest
from vlp_service.services import CalculateService
from vlp_service.settings import ServiceSettings


class VlpService(CalculateService):
    def __init__(self):
        super().__init__()
        self.settings = ServiceSettings()
        self.gateway_service_manager = GatewayManager(self.settings.gateway_service_url)

    async def _process_message(self, request: VlpRequest):
        logging.info(f"request received, task {request.id}")
        if self.settings.debug_mode is True:
            result = self.calculate_vlp(request)
        else:
            with open(Path("test_data/test_response.json")) as file:
                result = OilData.model_validate_json(json.load(file))

        updated_task = TaskSchema(id=request.id, status=TaskStatus.vlp_calculated, vlp=result)
        await self.gateway_service_manager.update_task(task_schema=updated_task)
        logging.info(f"task {request.id} processed")
