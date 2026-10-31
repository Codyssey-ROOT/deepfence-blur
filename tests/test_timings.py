import numpy as np

from app.detectors import ModelService
from app.geometry import Detection
from app.pipeline import Pipeline


def test_pipeline_reports_stage_timings_and_preserves_output():
    class FakeDetector:
        def detect_with_timings(self, image):
            return [Detection("text", (10, 10, 60, 30), 0.95)], {
                "person": 1.0,
                "face": 2.0,
                "license_plate": 3.0,
                "text": 4.0,
            }

    image = np.zeros((100, 120, 3), dtype=np.uint8)
    result = Pipeline(FakeDetector()).run(image)
    assert result.blurred.shape == image.shape
    assert result.preview.shape == image.shape
    for stage in ("person", "face", "license_plate", "text", "detect_total", "blur", "preview", "pipeline_total"):
        assert result.timings_ms[stage] >= 0
    assert result.timings_ms["person"] == 1.0


def test_four_model_adapter_timings_using_fakes():
    # Reuse the four fake model adapters from the existing contract test.
    from tests.test_adapters import FakePerson, FakeFace, FakePlate, FakeText

    svc = ModelService()
    svc._initialized = True
    svc.person = FakePerson()
    svc.face = FakeFace()
    svc.plate = FakePlate()
    svc.text = FakeText()
    detections, timings = svc.detect_with_timings(np.zeros((160, 240, 3), dtype=np.uint8))
    assert [d.label for d in detections] == ["person", "face", "license_plate", "text"]
    assert all(timings[name] >= 0 for name in ("model_load", "person", "face", "license_plate", "text"))
