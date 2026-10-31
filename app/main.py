"""DeepFence Blur API: uvicorn app.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import base64
import hmac
import os
from functools import lru_cache
from time import perf_counter
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.detectors import ModelService
from app.image_ops import decode_image, encode_jpg
from app.pipeline import Pipeline
from app.routers import blur_images

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
app = FastAPI(title="DeepFence Blur & Text Replacement API", version="1.0.0")


@lru_cache(maxsize=1)
def pipeline() -> Pipeline:
    return Pipeline(ModelService())


def authorize(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("BLUR_API_KEY", "").strip()
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")


app.include_router(blur_images.router, dependencies=[Depends(authorize)])


def run_file(file: UploadFile, text_mode: str):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="빈 파일 또는 12MB 초과 이미지입니다.")
    try:
        return pipeline().run(decode_image(data), text_mode=text_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health():
    service = pipeline().detector
    return {
        "status": "ok",
        "models_loaded": service._initialized,
        "ocr_recognizer_loaded": service._recognizer_initialized,
        "default_text_mode": pipeline().text_mode,
    }


@app.post("/process", dependencies=[Depends(authorize)])
def process(
    file: UploadFile = File(...),
    text_mode: Literal["replace", "blur"] | None = Query(default=None),
):
    """Original response keys preserved, while adding text_edits and text_mode."""
    started = perf_counter()
    out = run_file(file, text_mode or pipeline().text_mode)
    processed_jpg = encode_jpg(out.blurred)
    preview_jpg = encode_jpg(out.preview)
    payload = {
        "detections": [d.as_dict() for d in out.detections],
        "counts": {
            name: sum(d.label == name for d in out.detections)
            for name in ("person", "face", "license_plate", "text")
        },
        "text_mode": out.text_mode,
        "text_edits": out.text_edits,
        # Name retained for compatibility. In replace mode, this contains edited text.
        "blurred_image_base64": base64.b64encode(processed_jpg).decode("ascii"),
        "preview_image_base64": base64.b64encode(preview_jpg).decode("ascii"),
    }
    payload["timings_ms"] = {
        **out.timings_ms, "api_handler": round((perf_counter() - started) * 1000, 3),
    }
    return payload


@app.post("/process/image", dependencies=[Depends(authorize)])
def process_image(
    file: UploadFile = File(...),
    text_mode: Literal["replace", "blur"] | None = Query(default=None),
):
    """Binary JPEG output compatible with old POST /process/image."""
    started = perf_counter()
    out = run_file(file, text_mode or pipeline().text_mode)
    jpg = encode_jpg(out.blurred)
    api_ms = (perf_counter() - started) * 1000
    metrics = {**out.timings_ms, "api_handler": round(api_ms, 3)}
    return Response(
        content=jpg,
        media_type="image/jpeg",
        headers={
            "Server-Timing": ", ".join(f"{k};dur={v:.3f}" for k, v in metrics.items()),
            "X-Processing-Time-Ms": f"{api_ms:.3f}",
            "Access-Control-Expose-Headers": "Server-Timing, X-Processing-Time-Ms",
            "X-Text-Mode": out.text_mode,
            "X-Text-Replaced": str(out.text_edits.get("replaced", 0)),
            "X-Text-Fallback-Blurred": str(out.text_edits.get("blurred_fallback", 0)),
        },
    )
