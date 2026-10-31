"""Detect person/face/plates/text; blur only plates and text."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

import numpy as np

from app.geometry import Detection
from app.image_ops import blur_regions, draw_preview


class Detects(Protocol):

    def detect(
        self,
        image: np.ndarray
    ) -> list[Detection]:
        ...


@dataclass
class Processed:
    blurred: np.ndarray
    preview: np.ndarray
    detections: list[Detection]
    timings_ms: dict[str, float] = field(
        default_factory=dict
    )


class Pipeline:

    def __init__(
        self,
        detector: Detects
    ) -> None:
        self.detector = detector

    def run(
        self,
        image: np.ndarray
    ) -> Processed:

        started = perf_counter()

        # 객체 탐지
        detect_start = perf_counter()

        timed_detect = getattr(
            self.detector,
            "detect_with_timings",
            None
        )

        if callable(timed_detect):
            detections, timings = timed_detect(
                image
            )
        else:
            detections = self.detector.detect(
                image
            )
            timings = {}

        timings = dict(timings)

        timings["detect_total"] = round(
            (perf_counter() - detect_start) * 1000,
            3
        )

        # 텍스트 및 번호판 블러
        blur_start = perf_counter()

        result = blur_regions(
            image,
            detections
        )

        timings["blur"] = round(
            (perf_counter() - blur_start) * 1000,
            3
        )

        # 탐지 결과 미리보기
        preview_start = perf_counter()

        preview = draw_preview(
            result,
            detections
        )

        timings["preview"] = round(
            (perf_counter() - preview_start) * 1000,
            3
        )

        timings["pipeline_total"] = round(
            (perf_counter() - started) * 1000,
            3
        )

        return Processed(
            blurred=result,
            preview=preview,
            detections=detections,
            timings_ms=timings
        )