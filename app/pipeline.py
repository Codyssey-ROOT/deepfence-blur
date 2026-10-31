"""Detect -> optional recognize -> synthetic text replacement -> plate blur."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

import numpy as np

from app.geometry import Detection
from app.image_ops import blur_regions, draw_preview
from app.text_replace import find_korean_font, replace_text_region, similar_text


class Detects(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


@dataclass
class Processed:
    # Compatibility with the original public endpoint: `blurred` is the final image.
    blurred: np.ndarray
    preview: np.ndarray
    detections: list[Detection]
    timings_ms: dict[str, float] = field(default_factory=dict)
    text_edits: dict[str, int] = field(default_factory=dict)
    text_mode: str = "replace"


class Pipeline:
    def __init__(self, detector: Detects, text_mode: str | None = None) -> None:
        self.detector = detector
        self.text_mode = text_mode or os.getenv("TEXT_MODE", "replace")
        if self.text_mode not in {"replace", "blur"}:
            raise ValueError("TEXT_MODE must be replace or blur")

    def run(self, image: np.ndarray, text_mode: str | None = None) -> Processed:
        mode = text_mode or self.text_mode
        if mode not in {"replace", "blur"}:
            raise ValueError("text_mode must be replace or blur")
        total_start = perf_counter()
        started = perf_counter()
        timed_detect = getattr(self.detector, "detect_with_timings", None)
        if callable(timed_detect):
            detections, timings = timed_detect(image)
        else:
            detections, timings = self.detector.detect(image), {}
        timings["detect_total"] = round((perf_counter() - started) * 1000, 3)

        started = perf_counter()
        if mode == "replace":
            recognizer = getattr(self.detector, "recognize_text_with_timings", None)
            if callable(recognizer):
                detections, rec_timings = recognizer(image, detections)
                timings.update(rec_timings)
        timings.setdefault("text_recognition", 0.0)
        timings["recognition_total"] = round((perf_counter() - started) * 1000, 3)

        started = perf_counter()
        counts = {"replaced": 0, "blurred_fallback": 0, "plate_blurred": 0}
        if mode == "blur":
            output = blur_regions(image, detections)
            counts["blurred_fallback"] = sum(d.label == "text" for d in detections)
            counts["plate_blurred"] = sum(d.label == "license_plate" for d in detections)
        else:
            output = image.copy()
            font_path = find_korean_font()
            min_score = float(os.getenv("OCR_MIN_SCORE", "0.60"))
            plates = [d for d in detections if d.label == "license_plate"]
            for d in detections:
                if d.label != "text":
                    continue
                # Plate text must not be repainted; entire plate will be blurred.
                if any(_overlaps_plate(d, p) for p in plates):
                    continue
                try:
                    if not font_path or not d.recognized_text or (d.recognition_score or 0) < min_score:
                        raise ValueError("missing font or reliable OCR recognition")
                    synthetic = similar_text(d.recognized_text)
                    if synthetic == d.recognized_text:
                        raise ValueError("no characters eligible for replacement")
                    output = replace_text_region(output, d, synthetic, font_path)
                    counts["replaced"] += 1
                except (ValueError, RuntimeError, OSError):
                    output = blur_regions(output, [d], labels={"text"})
                    counts["blurred_fallback"] += 1
            timings["text_edit"] = round((perf_counter() - started) * 1000, 3)
            started = perf_counter()
            output = blur_regions(output, plates, labels={"license_plate"})
            counts["plate_blurred"] = len(plates)
        timings["blur"] = round((perf_counter() - started) * 1000, 3)

        started = perf_counter()
        preview = draw_preview(output, detections)
        timings["preview"] = round((perf_counter() - started) * 1000, 3)
        timings["pipeline_total"] = round((perf_counter() - total_start) * 1000, 3)
        return Processed(output, preview, detections, timings, counts, mode)


def _overlaps_plate(text: Detection, plate: Detection) -> bool:
    tx1, ty1, tx2, ty2 = text.bbox
    px1, py1, px2, py2 = plate.bbox
    ix = max(0, min(tx2, px2) - max(tx1, px1))
    iy = max(0, min(ty2, py2) - max(ty1, py1))
    return ix * iy >= 0.3 * max(1, (tx2 - tx1) * (ty2 - ty1))
