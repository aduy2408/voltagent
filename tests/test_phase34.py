import io

import pytest
from PIL import Image

from app.electrical import (
    ElectricalInput,
    calculate_current,
    calculate_power,
    calculate_resistance,
    calculate_three_phase_power,
)
from app.vision import UploadPolicy, UploadValidationError, YoloDetector, validate_upload


def test_electrical_calculators():
    assert calculate_power(120, 10).value == 1200
    assert calculate_current(120, 1200).value == 10
    assert calculate_resistance(120, 10).value == 12
    assert calculate_three_phase_power(voltage=400, current=10).value == pytest.approx(6928.203)


def test_input_requires_exactly_one_unknown():
    assert ElectricalInput(voltage=120, current=5).current == 5
    with pytest.raises(ValueError):
        ElectricalInput(voltage=120, current=5, power=600)


def test_detector_missing_checkpoint_is_safe():
    detector = YoloDetector("definitely-missing.pt")
    assert not detector.available
    assert detector.detect(b"not-an-image") == []
    assert detector.error


def test_upload_validation_checks_real_image_and_policy():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    validate_upload(buffer.getvalue(), "image/png", UploadPolicy(max_bytes=1000))
    with pytest.raises(UploadValidationError):
        validate_upload(b"bad", "image/png")
    with pytest.raises(UploadValidationError):
        validate_upload(buffer.getvalue(), "text/plain")
