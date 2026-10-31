"""Image I/O, blurring and a human-readable detection preview."""
from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.geometry import Detection

MAX_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
BLUR_LABELS = {"text", "license_plate"}
COLORS = {
    "person": (255, 130, 10),     # BGR
    "face": (90, 180, 20),
    "license_plate": (20, 130, 255),
    "text": (50, 40, 220),
}


def decode_image(data: bytes) -> np.ndarray:
    """Load JPG/PNG/WebP, honor EXIF rotation, drop metadata, output BGR."""
    try:
        with Image.open(BytesIO(data)) as raw:
            if raw.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("JPEG, PNG, WEBP 이미지만 지원합니다.")
            if raw.width * raw.height > MAX_PIXELS:
                raise ValueError("이미지의 픽셀 수가 제한을 초과했습니다.")
            oriented = ImageOps.exif_transpose(raw)
            rgb = np.array(oriented.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ValueError("정상적인 이미지 파일이 아닙니다.") from e
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_jpg(image: np.ndarray, quality: int = 92) -> bytes:
    ok, out = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("이미지 인코딩에 실패했습니다.")
    return out.tobytes()


def blur_regions(image: np.ndarray, detections: list[Detection], padding: float = 0.10, labels: set[str] | None = None) -> np.ndarray:
    """Blur selected detection labels (default: text + plates)."""
    out = image.copy()
    height, width = out.shape[:2]
    selected = BLUR_LABELS if labels is None else labels
    for item in detections:
        if item.label not in selected:
            continue
        x1, y1, x2, y2 = item.bbox
        pad_x = max(3, int((x2 - x1) * padding))
        pad_y = max(3, int((y2 - y1) * padding))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = out[y1:y2, x1:x2]
        sigma = max(8.0, max(roi.shape[:2]) * 0.30)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(
            roi, (0, 0), sigmaX=sigma, sigmaY=sigma,
        )
    return out


def draw_preview(blurred: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw detection boxes over blurred output, for debugging/demo only."""
    out = blurred.copy()
    for item in detections:
        x1, y1, x2, y2 = item.bbox
        color = COLORS.get(item.label, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        caption = f"{item.label} {item.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(caption, font, 0.5, 1)
        top = max(0, y1 - th - 9)
        cv2.rectangle(out, (x1, top), (min(out.shape[1], x1 + tw + 6), top + th + 8), color, -1)
        cv2.putText(out, caption, (x1 + 3, top + th + 3), font, 0.5, (255, 255, 255), 1)
    return out
