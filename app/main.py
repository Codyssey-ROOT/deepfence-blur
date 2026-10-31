"""DeepFence: download S3 source first, acknowledge, then blur in background."""
from __future__ import annotations

import hmac
import logging
import os
import re

from functools import lru_cache

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from app.detectors import ModelService
from app.image_ops import decode_image, encode_jpg
from app.integration import BlurIntegration, Settings
from app.pipeline import Pipeline


logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 12 * 1024 * 1024

app = FastAPI(
    title="DeepFence Blur API",
    version="2.1.0"
)


class ProcessRequest(BaseModel):
    imageId: int = Field(gt=0)
    objectKey: str = Field(
        min_length=1,
        max_length=1024
    )

    @field_validator("objectKey")
    @classmethod
    def check_source_key(cls, value: str) -> str:
        if re.fullmatch(
            r"images/source/\d+/[0-9a-fA-F-]{36}\.(?:jpg|jpeg|png|webp)",
            value,
        ) is None:
            raise ValueError(
                "Invalid source image objectKey"
            )

        return value


@lru_cache(maxsize=1)
def pipeline() -> Pipeline:
    return Pipeline(
        ModelService()
    )


@lru_cache(maxsize=1)
def integration() -> BlurIntegration:
    return BlurIntegration(
        Settings.from_env()
    )


def authorize(
    x_api_key: str | None = Header(default=None)
) -> None:
    expected = os.getenv(
        "BLUR_API_KEY", ""
    ).strip()

    if not expected or not hmac.compare_digest(
        x_api_key or "",
        expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid X-API-Key"
        )


def process_blur_background(
    image_id: int,
    raw: bytes
) -> None:
    """
    HTTP 응답을 반환한 이후 실행되는 블러 작업.
    """
    try:
        # 1. 이미지 디코딩
        # 2. 사람, 얼굴, 번호판, 텍스트 탐지
        # 3. 텍스트, 번호판 블러
        output = pipeline().run(
            decode_image(raw)
        )

        # 4. 결과 JPEG 인코딩
        jpeg = encode_jpg(
            output.blurred
        )

        # 5. Spring Presigned URL 요청
        # 6. S3 업로드
        # 7. Spring 업로드 완료 통보
        key = integration().upload_blurred(
            image_id,
            jpeg
        )

        counts = {
            name: sum(
                det.label == name
                for det in output.detections
            )
            for name in (
                "person",
                "face",
                "license_plate",
                "text"
            )
        }

        logger.info(
            "Blur completed imageId=%s resultKey=%s counts=%s timings_ms=%s",
            image_id,
            key,
            counts,
            output.timings_ms
        )

    except Exception:
        logger.exception(
            "Background blur failed imageId=%s",
            image_id
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": pipeline().detector._initialized
    }


@app.post(
    "/process",
    dependencies=[Depends(authorize)]
)
def process(
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    S3 다운로드 성공 시 200 OK 반환.
    탐지 및 블러는 응답 이후 실행.
    """
    try:
        # 원본 이미지 다운로드
        raw = integration().download_source(
            request.objectKey
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc)
        ) from exc

    except ClientError as exc:
        status = exc.response.get(
            "ResponseMetadata", {}
        ).get("HTTPStatusCode")

        logger.warning(
            "Source S3 download failed imageId=%s status=%s",
            request.imageId,
            status
        )

        raise HTTPException(
            status_code=404 if status == 404 else 502,
            detail="S3 원본 이미지를 가져오지 못했습니다."
        ) from exc

    except (BotoCoreError, OSError, RuntimeError):
        logger.exception(
            "Source S3 download failed imageId=%s",
            request.imageId
        )

        raise HTTPException(
            status_code=502,
            detail="S3 다운로드에 실패했습니다."
        )

    # 다운로드 완료 후 백그라운드 작업 등록
    background_tasks.add_task(
        process_blur_background,
        request.imageId,
        raw
    )

    # 블러 완료를 기다리지 않음
    return {
        "statusCode": 200,
        "message": "OK"
    }


@app.post(
    "/process/image",
    dependencies=[Depends(authorize)]
)
def process_image(
    file: UploadFile = File(...)
):
    """
    로컬 테스트용 이미지 직접 처리 API.
    """
    raw = file.file.read(
        MAX_UPLOAD_BYTES + 1
    )

    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="빈 파일 또는 12MB 초과 이미지입니다."
        )

    try:
        result = pipeline().run(
            decode_image(raw)
        )

        jpg = encode_jpg(
            result.blurred
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    return Response(
        content=jpg,
        media_type="image/jpeg"
    )