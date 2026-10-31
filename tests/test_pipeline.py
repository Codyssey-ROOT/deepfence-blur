import numpy as np
from app.geometry import Detection, make_detection
from app.image_ops import blur_regions
from app.pipeline import Pipeline


def test_normalized_coords_clamped_and_invalid_rejected():
    detection = make_detection("text", (-4.2, 10.1, 1000, 35.2), 0.94, 80, 60)
    assert detection.bbox == (0, 10, 80, 36)
    assert make_detection("text", (20, 10, 10, 30), 0.9, 80, 60) is None


def test_blur_changes_only_text_and_plate_neighborhood():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, size=(160, 240, 3), dtype=np.uint8)
    items = [
        Detection("person", (0, 0, 40, 40), 0.9),
        Detection("face", (50, 0, 90, 40), 0.9),
        Detection("text", (50, 70, 130, 100), 0.9),
        Detection("license_plate", (150, 100, 225, 135), 0.9),
    ]
    out = blur_regions(image, items)
    assert np.array_equal(out[:35, :100], image[:35, :100])
    assert not np.array_equal(out[75:95, 60:120], image[75:95, 60:120])
    assert not np.array_equal(out[110:130, 165:215], image[110:130, 165:215])


def test_pipeline_with_fake_detector():
    class Fake:
        def detect(self, image):
            return [Detection("text", (40, 40, 90, 60), 0.95)]
    image = np.zeros((110, 130, 3), dtype=np.uint8)
    result = Pipeline(Fake()).run(image)
    assert len(result.detections) == 1
    assert result.blurred.shape == image.shape
    assert result.preview.shape == image.shape
