"""S3 presigned upload flow for the Spring backend.

1. POST /blur/images/presigned-url -> client PUTs the original directly to S3
2. POST /blur/images/complete      -> server checks the object, then blurs it in the background
   and writes {S3_KEY_PREFIX}/results/{uuid}.jpg

X-API-Key auth (app.main.authorize) is attached in app.main via include_router.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import PurePosixPath
from typing import Callable, Protocol

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException

from app import s3
from app.image_ops import decode_image, encode_jpg
from app.schemas import ApiError, ApiResponse, CompleteRequest, PresignedUrlData, PresignedUrlRequest

logger = logging.getLogger(__name__)

# Single source of truth for accepted uploads: contentType -> allowed fileExt values.
ALLOWED_TYPES: dict[str, set[str]] = {
    "image/png": {"png"},
    "image/jpeg": {"jpg", "jpeg"},
}


def key_prefix() -> str:
    return os.getenv("S3_KEY_PREFIX", "images/blur").strip().strip("/")


def max_file_size() -> int:
    return int(os.getenv("MAX_FILE_SIZE_BYTES", "12582912"))


def presigned_expires() -> int:
    return int(os.getenv("PRESIGNED_EXPIRES_SECONDS", "300"))


class BadRequest(HTTPException):
    def __init__(self, message: str) -> None:
        super().__init__(status_code=400, detail=message)


def error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=ApiError(statusCode=status, message=message).model_dump())


class ApiErrorRoute(APIRoute):
    """Wraps errors of these routes only into {statusCode, message}; existing routes keep FastAPI defaults."""

    def get_route_handler(self) -> Callable:
        handler = super().get_route_handler()

        async def wrapped(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                first = exc.errors()[0]
                field = ".".join(str(p) for p in first.get("loc", ())[1:]) or "body"
                return error_response(400, f"{field}: {first.get('msg', 'invalid')}")
            except HTTPException as exc:
                return error_response(exc.status_code, str(exc.detail))
            except Exception as exc:  # noqa: BLE001 - last-resort 500 with a stable body
                logger.error("blur_images %s failed: %s", request.url.path, type(exc).__name__)
                return error_response(500, "Internal Server Error")

        return wrapped


# TODO(spec): 403 has no trigger yet; permission rules are not defined by the spec.
router = APIRouter(
    prefix="/blur/images",
    tags=["blur-images"],
    route_class=ApiErrorRoute,
    responses={code: {"model": ApiError} for code in (400, 401, 403, 404, 500)},
)


def new_object_key(file_ext: str) -> str:
    return f"{key_prefix()}/{uuid.uuid4()}.{file_ext}"


def result_key(object_key: str) -> str:
    return f"{key_prefix()}/results/{PurePosixPath(object_key).stem}.jpg"


def validate_object_key(object_key: str) -> str:
    key = object_key.strip()
    if not key or ".." in key or not key.startswith(f"{key_prefix()}/"):
        raise BadRequest("objectKey is invalid")
    # TODO(spec): keys under {prefix}/results/ or not matching {prefix}/{uuid}.{ext} are still accepted.
    return key


class OriginalReader(Protocol):
    def read(self, object_key: str) -> bytes: ...


class S3OriginalReader:
    """Reads the original with the server's own IAM permissions.

    TODO(spec): how the server reads the original (own IAM vs. presigned GET from Spring) is not agreed.
    """

    def read(self, object_key: str) -> bytes:
        return s3.get_bytes(object_key, max_file_size())


class CompletionNotifier(Protocol):
    # TODO(spec): callback URL and payload (userId, result key, success/failure) are not agreed with Spring.
    def notify(self, object_key: str, result_key: str | None, success: bool) -> None: ...


original_reader: OriginalReader = S3OriginalReader()


def process_original(object_key: str, reader: OriginalReader) -> None:
    """Background job. Logs only the key and error type: never image bytes or recognized text."""
    try:
        import app.main as main  # lazy: app.main imports this module

        data = reader.read(object_key)
        out = main.pipeline().run(decode_image(data))
        target = result_key(object_key)
        s3.put_bytes(target, encode_jpg(out.blurred), "image/jpeg")
        logger.info("blur done key=%s result=%s", object_key, target)
    except Exception as exc:  # noqa: BLE001 - background job must not crash the worker
        logger.error("blur failed key=%s error=%s", object_key, type(exc).__name__)


@router.post("/presigned-url", response_model=ApiResponse[PresignedUrlData])
def create_presigned_url(body: PresignedUrlRequest):
    content_type = body.contentType.strip().lower()
    file_ext = body.fileExt.strip().lower().lstrip(".")
    if content_type not in ALLOWED_TYPES:
        raise BadRequest(f"contentType must be one of {sorted(ALLOWED_TYPES)}")
    if file_ext not in ALLOWED_TYPES[content_type]:
        raise BadRequest("fileExt does not match contentType")
    if not 1 <= body.fileSize <= max_file_size():
        raise BadRequest(f"fileSize must be between 1 and {max_file_size()} bytes")

    object_key = new_object_key(file_ext)
    upload_url = s3.presign_put(object_key, content_type, body.fileSize, presigned_expires())
    public_url = f"{s3.required_env('CDN_BASE_URL').rstrip('/')}/{object_key}"
    return ApiResponse(data=PresignedUrlData(uploadUrl=upload_url, objectKey=object_key, publicUrl=public_url))


@router.post("/complete", response_model=ApiResponse[None], response_model_exclude_none=True)
def complete_upload(body: CompleteRequest, background: BackgroundTasks):
    object_key = validate_object_key(body.objectKey)
    meta = s3.head(object_key)
    if meta is None:
        raise HTTPException(status_code=404, detail="object not found")
    if meta.get("ContentLength", 0) > max_file_size():
        raise BadRequest("object exceeds size limit")
    background.add_task(process_original, object_key, original_reader)
    return ApiResponse()
