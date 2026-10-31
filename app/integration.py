"""S3 source download and Spring-issued presigned PUT + completion callback."""
from __future__ import annotations

import os
from dataclasses import dataclass

import boto3
import httpx


MAX_SOURCE_BYTES = 12 * 1024 * 1024
MAX_RESULT_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    bucket: str
    spring_url: str
    internal_key: str

    @classmethod
    def from_env(cls) -> "Settings":
        result = cls(
            os.getenv(
                "AWS_S3_BUCKET", ""
            ).strip(),

            os.getenv(
                "SPRING_BASE_URL", ""
            ).strip().rstrip("/"),

            os.getenv(
                "INTERNAL_API_KEY", ""
            ).strip(),
        )

        if not all((
            result.bucket,
            result.spring_url,
            result.internal_key
        )):
            raise RuntimeError(
                "AWS_S3_BUCKET, SPRING_BASE_URL, INTERNAL_API_KEY are required"
            )

        if not result.spring_url.startswith(
            ("https://", "http://")
        ):
            raise RuntimeError(
                "SPRING_BASE_URL must be an HTTP(S) URL"
            )

        return result


class BlurIntegration:

    def __init__(
        self,
        settings: Settings
    ) -> None:
        self.settings = settings

        self.s3 = boto3.client(
            "s3",
            region_name=os.getenv(
                "AWS_REGION",
                "ap-northeast-2"
            )
        )

        self.http = httpx.Client(
            timeout=httpx.Timeout(
                30.0,
                connect=5.0
            )
        )

    def download_source(
        self,
        object_key: str
    ) -> bytes:

        result = self.s3.get_object(
            Bucket=self.settings.bucket,
            Key=object_key
        )

        body = result["Body"]

        try:
            expected_size = result.get(
                "ContentLength"
            )

            if (
                expected_size is not None
                and expected_size > MAX_SOURCE_BYTES
            ):
                raise ValueError(
                    "원본 이미지 크기가 12MB를 초과합니다."
                )

            data = body.read(
                MAX_SOURCE_BYTES + 1
            )

        finally:
            body.close()

        if not data or len(data) > MAX_SOURCE_BYTES:
            raise ValueError(
                "빈 이미지 또는 12MB 초과 이미지입니다."
            )

        return data

    def _post_spring(
        self,
        path: str,
        payload: dict
    ) -> dict:

        response = self.http.post(
            self.settings.spring_url + path,
            json=payload,
            headers={
                "X-Internal-API-Key":
                    self.settings.internal_key
            }
        )

        response.raise_for_status()

        body = response.json()

        if body.get("statusCode") != 200:
            raise RuntimeError(
                f"Spring rejected {path}: {body.get('message')}"
            )

        return body

    def upload_blurred(
        self,
        image_id: int,
        jpeg: bytes
    ) -> str:

        if (
            not jpeg
            or len(jpeg) > MAX_RESULT_BYTES
        ):
            raise ValueError(
                "블러 결과 이미지가 비어 있거나 10MB를 초과합니다."
            )

        # 1. Spring에 Presigned URL 요청
        result = self._post_spring(
            "/blur/images/presigned-url",
            {
                "sourceImageId": image_id,
                "contentType": "image/jpeg",
                "fileSize": len(jpeg)
            }
        )["data"]

        key = result["objectKey"]

        # 2. S3 업로드
        response = self.http.put(
            result["uploadUrl"],
            content=jpeg,
            headers={
                "Content-Type": "image/jpeg"
            },
            timeout=90.0
        )

        response.raise_for_status()

        # 3. Spring에 업로드 완료 통보
        self._post_spring(
            "/blur/images/complete",
            {
                "sourceImageId": image_id,
                "objectKey": key
            }
        )

        return key