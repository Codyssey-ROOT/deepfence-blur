"""Request/response bodies for /blur/images/*. Field names follow the Spring (camelCase) contract."""
from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    statusCode: int = 200
    message: str = "OK"
    data: T | None = None


# TODO(spec): error body format is not defined by the spec; unified as {statusCode, message}.
class ApiError(BaseModel):
    statusCode: int
    message: str


class PresignedUrlRequest(BaseModel):
    contentType: str = Field(min_length=1)
    fileExt: str = Field(min_length=1)
    fileSize: int = Field(strict=True, description="bytes")


class PresignedUrlData(BaseModel):
    uploadUrl: str
    objectKey: str
    publicUrl: str


class CompleteRequest(BaseModel):
    objectKey: str
