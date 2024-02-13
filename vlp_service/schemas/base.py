from typing import Annotated, TypeAlias

from common.rabbit.models import BaseRequest
from pydantic import BaseModel, Field, field_validator

PositiveFloatList: TypeAlias = list[Annotated[float, Field(ge=0)]]
NotZeroFloatList: TypeAlias = list[Annotated[float, Field(gt=0)]]


class Inclinometry(BaseModel):
    MD: PositiveFloatList = Field(title="Измеренная по стволу глубина, м")
    TVD: PositiveFloatList = Field(title="Вертикальная глубина, м")


class Pipeline(BaseModel):
    d: float = Field(gt=0, title="Диаметр трубы, м")


class Tubing(Pipeline):
    h_mes: float = Field(gt=0, title="Глубина спуска НКТ, м")


class PVT(BaseModel):
    wct: float = Field(ge=0, le=100, title="Обводненность, %")
    rp: float = Field(ge=0, title="Газовый фактор, м3/т")
    gamma_oil: float = Field(ge=0.6, le=1, title="Отн. плотность нефти")
    gamma_gas: float = Field(ge=0.5, le=1, title="Отн. плотность газа")
    gamma_wat: float = Field(ge=0.98, le=1.2, title="Отн. плотность воды")
    t_res: float = Field(ge=10, le=500, title="Пластовая температура, C")


class VlpRequest(BaseRequest):
    inclinometry: Inclinometry = Field(title="Инклинометрия")
    casing: Pipeline = Field(title="Данные по ЭК")
    tubing: Tubing = Field(title="Данные по НКТ")
    pvt: PVT = Field(title="PVT")
    p_wh: float = Field(ge=0, title="Буферное давление, атм")
    geo_grad: float = Field(ge=0, title="Градиент температуры, C/100 м")
    h_res: float = Field(ge=0, title="Глубина Верхних Дыр Перфорации, м")


class VlpResponse(BaseModel):
    q_liq: PositiveFloatList = Field(title="Дебиты жидкости, м3/сут")
    p_wf: NotZeroFloatList = Field(title="Забойное давление, атм")

    @field_validator("q_liq", "p_wf")
    @classmethod
    def round_result(cls, v):
        return [round(val, 2) for val in v]
