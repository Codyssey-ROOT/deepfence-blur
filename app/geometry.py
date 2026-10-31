"""Normalized image detections."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor, isfinite
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float

    polygon: tuple[
        tuple[float, float], ...
    ] | None = field(
        default=None,
        repr=False
    )

    def as_dict(self) -> dict:
        return {
            "class": self.label,
            "bbox": list(self.bbox),
            "confidence": round(
                float(self.confidence),
                4
            )
        }


def make_detection(
    label: str,
    coords: Iterable[float],
    score: float,
    width: int,
    height: int,
    polygon: Iterable[Iterable[float]] | None = None,
) -> Detection | None:

    vals = [
        float(v)
        for v in coords
    ]

    if (
        len(vals) != 4
        or not all(isfinite(v) for v in vals)
        or not isfinite(float(score))
    ):
        return None

    x1, y1, x2, y2 = vals

    x1, y1, x2, y2 = (
        max(0, min(width, floor(x1))),
        max(0, min(height, floor(y1))),
        max(0, min(width, ceil(x2))),
        max(0, min(height, ceil(y2))),
    )

    if x2 <= x1 or y2 <= y1:
        return None

    saved_polygon = None

    if polygon is not None:
        try:
            points = tuple(
                (float(p[0]), float(p[1]))
                for p in polygon
            )

            if len(points) >= 4 and all(
                isfinite(x) and isfinite(y)
                for x, y in points
            ):
                saved_polygon = points

        except (
            TypeError,
            ValueError,
            IndexError
        ):
            pass

    return Detection(
        label,
        (x1, y1, x2, y2),
        float(score),
        saved_polygon
    )