import random

import cv2
import numpy as np
import pytest

from app.geometry import Detection
from app.pipeline import Pipeline
from app.text_replace import find_korean_font, rectified_crop, replace_text_region, similar_text


def test_similar_text_preserves_script_shape_and_changes_all_supported_chars():
    original = "김민수 010-1234 ABC xyz"
    changed = similar_text(original, rng=random.Random(4))
    assert len(original) == len(changed)
    assert changed[3] == ' ' and changed[7] == '-' and changed[12] == ' '
    for x, y in zip(original, changed):
        if x.isascii() and (x.isalpha() or x.isdigit()) or ('가' <= x <= '힣'):
            assert x != y
        else:
            assert x == y
    assert all('가' <= v <= '힣' for v in changed[:3])


def test_rectified_crop_handles_rotated_quad():
    img = np.full((160, 280, 3), 240, np.uint8)
    quad = ((35, 42), (200, 25), (207, 59), (40, 75))
    det = Detection("text", (35, 25, 207, 75), 0.99, quad)
    crop = rectified_crop(img, det)
    assert crop.shape[1] >= 160 and crop.shape[0] >= 30


def test_text_edit_only_changes_local_region_and_emits_visible_replacement():
    font = find_korean_font()
    if not font:
        pytest.skip("No Korean font installed in test environment")
    original = np.full((130, 340, 3), 245, np.uint8)
    cv2.putText(original, "ABC123", (25, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (25, 25, 25), 2)
    quad = ((20, 40), (300, 40), (300, 82), (20, 82))
    det = Detection("text", (20, 40, 300, 82), 0.95, quad, "ABC123", 0.98)
    edited = replace_text_region(original, det, "XYZ987", font)
    assert np.array_equal(original[:10], edited[:10])
    assert np.array_equal(original[115:], edited[115:])
    assert np.any(edited[44:78, 40:285] != original[44:78, 40:285])
    assert np.count_nonzero(edited[45:78, 40:280] < 150) > 5


def test_pipeline_replaces_recognized_text_and_blurs_plate():
    image = np.full((220, 420, 3), 242, np.uint8)
    cv2.putText(image, "HELLO123", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    image[130:155, 200:350] = np.random.default_rng(2).integers(0, 255, (25, 150, 3), dtype=np.uint8)

    class Fake:
        def detect(self, image):
            return [
                Detection("person", (1, 1, 10, 10), 0.9),
                Detection("text", (12, 38, 310, 86), 0.99,
                          ((12, 38), (310, 38), (310, 86), (12, 86))),
                Detection("license_plate", (195, 126, 360, 162), 0.95),
            ]

        def recognize_text_with_timings(self, image, detections):
            from dataclasses import replace
            return [replace(d, recognized_text="HELLO123", recognition_score=0.99)
                    if d.label == "text" else d for d in detections], {"text_recognition": 1.0}

    out = Pipeline(Fake(), text_mode="replace").run(image)
    assert out.text_edits["replaced"] == 1
    assert out.text_edits["plate_blurred"] == 1
    assert out.text_edits["blurred_fallback"] == 0
    assert "text_recognition" in out.timings_ms
    assert np.array_equal(out.blurred[100:115, 0:100], image[100:115, 0:100])
    assert not np.array_equal(out.blurred[137:150, 210:340], image[137:150, 210:340])
    assert out.detections[1].as_dict().get("recognized_text") is None


def test_ocr_low_score_falls_back_to_text_blur(monkeypatch):
    image = np.random.default_rng(10).integers(0, 255, (120, 220, 3), dtype=np.uint8)

    class Fake:
        def detect(self, image):
            return [Detection("text", (20, 40, 175, 80), 0.99,
                              recognized_text="김민수", recognition_score=0.2)]

    out = Pipeline(Fake(), text_mode="replace").run(image)
    assert out.text_edits["replaced"] == 0
    assert out.text_edits["blurred_fallback"] == 1
    assert not np.array_equal(out.blurred[50:70, 40:130], image[50:70, 40:130])
