from pydantic import Field
from pydantic_settings import BaseSettings


class ServiceSettings(BaseSettings):
    debug_mode: bool = Field(validation_alias="DEBUG_MODE", default=True)
    gateway_service_url: str = Field(validation_alias="GATEWAY_SERVICE_URL")
    exchange: str = Field(validation_alias="EXCHANGE", default="gpn_exchange")
    vlp_queue: str = Field(validation_alias="VLP_QUEUE", default="vlp")
    nodal_analysis_queue: str = Field(
        validation_alias="NODAL_ANALYSIS_QUEUE", default="nodal_analysis"
    )
