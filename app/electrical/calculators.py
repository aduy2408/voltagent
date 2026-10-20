from __future__ import annotations

import math

from .schemas import ElectricalResult


def _positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return float(value)


def calculate_ohms_law(*, voltage: float | None = None, current: float | None = None, resistance: float | None = None) -> dict[str, float | str]:
    values = {"voltage": voltage, "current": current, "resistance": resistance}
    if sum(v is None for v in values.values()) != 1:
        raise ValueError("provide exactly two of voltage, current, and resistance")
    supplied = {k: _positive(v, k) for k, v in values.items() if v is not None}
    if voltage is None:
        result, unit, formula = supplied["current"] * supplied["resistance"], "V", "V = I × R"
        key = "voltage"
    elif current is None:
        result, unit, formula = supplied["voltage"] / supplied["resistance"], "A", "I = V ÷ R"
        key = "current"
    else:
        result, unit, formula = supplied["voltage"] / supplied["current"], "Ω", "R = V ÷ I"
        key = "resistance"
    return {key: result, "unit": unit, "formula": formula}


def calculate_series_resistance(resistances: list[float]) -> dict[str, float | str]:
    if not resistances or any(r <= 0 for r in resistances):
        raise ValueError("resistances must contain positive values")
    return {"resistance": float(sum(resistances)), "unit": "Ω", "formula": "R = ΣRi"}


def calculate_parallel_resistance(resistances: list[float]) -> dict[str, float | str]:
    if not resistances or any(r <= 0 for r in resistances):
        raise ValueError("resistances must contain positive values")
    return {"resistance": 1 / sum(1 / r for r in resistances), "unit": "Ω", "formula": "1/R = Σ(1/Ri)"}


def calculate_voltage_divider(input_voltage: float, r1: float, r2: float) -> dict[str, float | str]:
    _positive(input_voltage, "input_voltage"); _positive(r1, "r1"); _positive(r2, "r2")
    return {"voltage": input_voltage * r2 / (r1 + r2), "unit": "V", "formula": "Vout = Vin × R2/(R1+R2)"}


def calculate_current_divider(input_current: float, r1: float, r2: float) -> dict[str, float | str]:
    _positive(input_current, "input_current"); _positive(r1, "r1"); _positive(r2, "r2")
    return {"current_r1": input_current * r2 / (r1 + r2), "current_r2": input_current * r1 / (r1 + r2), "unit": "A", "formula": "I1=It×R2/(R1+R2)"}


def calculate_power(voltage: float, current: float, power_factor: float = 1.0) -> dict[str, float | str]:
    _positive(voltage, "voltage"); _positive(current, "current")
    if not 0 < power_factor <= 1: raise ValueError("power_factor must be between 0 and 1")
    return ElectricalResult(quantity="power", value=voltage * current * power_factor, unit="W", formula="P = V × I × PF")


def calculate_rc_time_constant(resistance: float, capacitance: float) -> dict[str, float | str]:
    _positive(resistance, "resistance"); _positive(capacitance, "capacitance")
    return {"time_constant": resistance * capacitance, "unit": "s", "formula": "τ = R × C"}


def calculate_current(voltage: float, power: float, power_factor: float = 1.0) -> dict[str, float | str]:
    _positive(voltage, "voltage"); _positive(power, "power")
    if not 0 < power_factor <= 1: raise ValueError("power_factor must be between 0 and 1")
    return ElectricalResult(quantity="current", value=power / (voltage * power_factor), unit="A", formula="I = P ÷ (V × PF)")


def calculate_resistance(voltage: float, current: float) -> dict[str, float | str]:
    result = calculate_ohms_law(voltage=voltage, current=current)
    return ElectricalResult(quantity="resistance", value=float(result["resistance"]), unit="Ω", formula=str(result["formula"]))


def calculate_three_phase_power(voltage: float, current: float, power_factor: float = 1.0) -> dict[str, float | str]:
    _positive(voltage, "voltage"); _positive(current, "current")
    if not 0 < power_factor <= 1: raise ValueError("power_factor must be between 0 and 1")
    return ElectricalResult(quantity="three_phase_power", value=math.sqrt(3) * voltage * current * power_factor, unit="W", formula="P = √3 × VLL × I × PF")
