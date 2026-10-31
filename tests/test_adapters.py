"""Integration *contract* test with fake models; no weights/network needed."""
from types import SimpleNamespace

import numpy as np

from app.detectors import ModelService


class FakeTensor:
    def __init__(self, values):
        self.values = values
    def __getitem__(self, key):
        return self if key == 0 else self
    def cpu(self):
        return self
    def tolist(self):
        return self.values
    def item(self):
        return self.values[0]


class FakePerson:
    def predict(self, **kwargs):
        assert kwargs["classes"] == [0]
        return [SimpleNamespace(boxes=[SimpleNamespace(
            xyxy=FakeTensor([5, 6, 50, 90]), conf=FakeTensor([0.97]),
        )])]


class FakeFace:
    def setInputSize(self, size):
        assert size == (240, 160)
    def detect(self, image):
        row = np.zeros(15)
        row[:4] = [40, 30, 20, 25]
        row[14] = 0.95
        return 1, np.array([row])


class FakePlate:
    def detect(self, image):
        return [[150, 100, 210, 128, 0.9]]


class FakeText:
    def predict(self, **kwargs):
        assert kwargs["batch_size"] == 1
        return [SimpleNamespace(json={"res": {
            "dt_polys": np.array([[[2, 100], [120, 100], [120, 130], [2, 130]]]),
            "dt_scores": [0.88],
        }})]


def test_all_four_detector_adapters_normalize_boxes():
    svc = ModelService()
    svc._initialized = True
    svc.person = FakePerson()
    svc.face = FakeFace()
    svc.plate = FakePlate()
    svc.text = FakeText()
    detections = svc.detect(np.zeros((160, 240, 3), dtype=np.uint8))
    assert [d.label for d in detections] == ["person", "face", "license_plate", "text"]
    assert [d.bbox for d in detections] == [
        (5, 6, 50, 90),
        (40, 30, 60, 55),
        (150, 100, 210, 128),
        (2, 100, 120, 130),
    ]
