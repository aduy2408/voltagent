from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ElectricalInput(BaseModel):
    voltage: float = Field(gt=0)
    current: float | None = Field(default=None, gt=0)
    power: float | None = Field(default=None, gt=0)
    power_factor: float = Field(default=1.0, gt=0, le=1)

    @model_validator(mode="after")
    def has_one_unknown(self):
        if sum(value is None for value in (self.current, self.power)) != 1:
            raise ValueError("provide exactly one of current or power")
        return self


class ThreePhaseInput(BaseModel):
    voltage: float = Field(gt=0)
    current: float = Field(gt=0)
    power_factor: float = Field(default=1.0, gt=0, le=1)
    connection: Literal["three_phase"] = "three_phase"


class ElectricalResult(BaseModel):
    quantity: str
    value: float = Field(ge=0)
    unit: str
    formula: str

    def as_dict(self) -> dict[str, float | str]:
        return {"result": self.value, "unit": self.unit, "formula": self.formula, self.quantity: self.value}
