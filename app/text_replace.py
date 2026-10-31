"""Synthetic text generation and perspective-aware replacement of OCR regions.

No generative AI dependency. Only synthetic replacements are drawn into the image.
This is a visual obfuscation feature, not a guaranteed irrecoverable redaction.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.geometry import Detection

DEFAULT_FONTS = (
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",   # Ubuntu / Docker
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",       # macOS
    "/Library/Fonts/Arial Unicode.ttf",
)


def find_korean_font() -> str | None:
    explicit = os.getenv("TEXT_FONT_PATH")
    if explicit and Path(explicit).is_file():
        return explicit
    return next((font for font in DEFAULT_FONTS if Path(font).is_file()), None)


def similar_text(original: str, rng=None) -> str:
    """Replace each Hangul syllable/Latin letter/digit; retain spacing/symbols.

    This preserves appearance *roughly*, not semantics or true format validity.
    Avoid putting the original text in logs or API responses.
    """
    rng = rng or secrets.SystemRandom()
    transformed: list[str] = []
    for ch in original:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:  # completed Korean Hangul syllables
            offset = code - 0xAC00
            cho, rest = divmod(offset, 21 * 28)
            jung, jong = divmod(rest, 28)
            # Small vowel shift, keeps initial consonant and final consonant.
            jung = (jung + rng.choice((1, 2, 3, 4))) % 21
            transformed.append(chr(0xAC00 + ((cho * 21 + jung) * 28 + jong)))
        elif '0' <= ch <= '9':
            transformed.append(str((int(ch) + rng.randint(1, 9)) % 10))
        elif 'A' <= ch <= 'Z':
            transformed.append(chr(65 + (code - 65 + rng.randint(1, 25)) % 26))
        elif 'a' <= ch <= 'z':
            transformed.append(chr(97 + (code - 97 + rng.randint(1, 25)) % 26))
        else:
            transformed.append(ch)
    return ''.join(transformed)


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        raise ValueError("OCR polygon needs four corners")
    if len(pts) != 4:
        pts = cv2.boxPoints(cv2.minAreaRect(pts))
    # Clockwise angle ordering, rotated to the upper-left corner.
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    first = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pts = np.roll(pts, -first, axis=0)
    # Ensure the second point is on the right (not bottom-left).
    if pts[1, 0] < pts[3, 0]:
        pts = pts[[0, 3, 2, 1]]
    return pts


def _quad(d: Detection) -> np.ndarray:
    if d.polygon and len(d.polygon) >= 4:
        return _order_quad(np.asarray(d.polygon, dtype=np.float32))
    x1, y1, x2, y2 = d.bbox
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def rectified_crop(image: np.ndarray, d: Detection) -> np.ndarray:
    """Deskew OCR text region for the Korean recognition model."""
    q = _quad(d)
    width = max(4, int(round(max(np.linalg.norm(q[1]-q[0]), np.linalg.norm(q[2]-q[3])))))
    height = max(4, int(round(max(np.linalg.norm(q[3]-q[0]), np.linalg.norm(q[2]-q[1])))))
    width, height = min(width, 4096), min(height, 1024)
    dest = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(q, dest)
    return cv2.warpPerspective(image, matrix, (width, height))


def _text_color(original_roi: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """Approximate original text color using foreground vs surrounding border."""
    inner = original_roi[mask > 0]
    outside = original_roi[mask == 0]
    if not len(inner) or not len(outside):
        return (25, 25, 25)
    background = np.median(outside.astype(np.float32), axis=0)
    inner_pixels = inner.astype(np.float32)
    distance = np.linalg.norm(inner_pixels - background, axis=1)
    foreground = inner_pixels[distance >= np.percentile(distance, 80)]
    if len(foreground) == 0:
        return (25, 25, 25)
    bgr = np.median(foreground, axis=0)
    # foreground may be noisy; preserve high enough contrast for readability
    if np.linalg.norm(bgr - background) < 28:
        bgr = np.array([25, 25, 25] if np.mean(background) > 128 else [235, 235, 235])
    return tuple(int(np.clip(v, 0, 255)) for v in bgr)


def _render_text(text: str, width: int, height: int, font_path: str, color_bgr: tuple[int, int, int]) -> np.ndarray:
    """Draw text on an RGBA canvas using font scaling to fit the OCR quad."""
    if width < 5 or height < 5 or not text.strip():
        raise ValueError("Text line too small or empty")
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    # Choose largest legible font that fits; if it won't fit, fall back to blur.
    selected = None
    for size in range(max(7, int(height * 1.5)), 5, -1):
        font = ImageFont.truetype(font_path, size)
        bounds = draw.textbbox((0, 0), text, font=font)
        w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
        if w <= max(1, width - 2) and h <= max(1, height - 2):
            selected = (font, bounds, w, h)
            break
    if selected is None:
        raise ValueError("No font size fits in text box")
    font, bounds, tw, th = selected
    x = (width - tw) / 2 - bounds[0]
    y = (height - th) / 2 - bounds[1]
    b, g, r = color_bgr
    draw.text((x, y), text, font=font, fill=(r, g, b, 255))
    return np.asarray(layer)


def replace_text_region(image: np.ndarray, d: Detection, replacement: str, font_path: str) -> np.ndarray:
    """Remove original text then project the synthetic replacement into its quad.

    Operates on a local ROI instead of warping the whole photograph; exceptions
    must be caught by caller and fall back to blur.
    """
    q = _quad(d)
    h_img, w_img = image.shape[:2]
    line_w = max(5, int(round(max(np.linalg.norm(q[1]-q[0]), np.linalg.norm(q[2]-q[3])))))
    line_h = max(5, int(round(max(np.linalg.norm(q[3]-q[0]), np.linalg.norm(q[2]-q[1])))))
    if line_h < 9 or line_w < 12:
        raise ValueError("Text region too small for reliable replacement")
    if line_h > 400 or line_w > 1800:
        raise ValueError("Text region too large for the lightweight editor")
    # Local crop around the OCR line, include surrounding pixels for inpainting.
    margin = max(8, min(48, int(line_h * 0.8)))
    lo = np.floor(q.min(axis=0)).astype(int) - margin
    hi = np.ceil(q.max(axis=0)).astype(int) + margin
    x1, y1 = max(0, lo[0]), max(0, lo[1])
    x2, y2 = min(w_img, hi[0]), min(h_img, hi[1])
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError("Invalid local text ROI")
    original_roi = image[y1:y2, x1:x2].copy()
    local_q = q - np.float32([x1, y1])
    mask = np.zeros(original_roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(local_q).astype(np.int32)], 255)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    # Safeguard: no surrounding pixels to infer background from.
    if mask.sum() / 255 > mask.size * 0.90:
        raise ValueError("Not enough background for inpainting")
    color = _text_color(original_roi, mask)
    restored = cv2.inpaint(original_roi, mask, 3, cv2.INPAINT_TELEA)
    rendered_rgba = _render_text(replacement, line_w, line_h, font_path, color)
    src = np.float32([[0, 0], [line_w - 1, 0], [line_w - 1, line_h - 1], [0, line_h - 1]])
    transform = cv2.getPerspectiveTransform(src, local_q)
    warped = cv2.warpPerspective(
        rendered_rgba, transform, (x2 - x1, y2 - y1),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    text_bgr = cv2.cvtColor(warped[:, :, :3], cv2.COLOR_RGB2BGR)
    composed = (restored.astype(np.float32) * (1 - alpha) + text_bgr.astype(np.float32) * alpha)
    out = image.copy()
    out[y1:y2, x1:x2] = np.clip(composed, 0, 255).astype(np.uint8)
    return out
