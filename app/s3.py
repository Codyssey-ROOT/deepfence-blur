"""Thin boto3 wrapper. Credentials come from the default chain (EC2 IAM role / env)."""
from __future__ import annotations

import os
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value


@lru_cache(maxsize=1)
def client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION") or None,
        config=Config(signature_version="s3v4"),
    )


def bucket() -> str:
    return required_env("S3_BUCKET")


def presign_put(key: str, content_type: str, content_length: int, expires_seconds: int) -> str:
    """ContentType/ContentLength are signed, so the client must PUT with exactly these headers."""
    return client().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket(),
            "Key": key,
            "ContentType": content_type,
            "ContentLength": content_length,
        },
        ExpiresIn=expires_seconds,
    )


def head(key: str) -> dict | None:
    """Object metadata, or None when the object does not exist."""
    try:
        return client().head_object(Bucket=bucket(), Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in NOT_FOUND_CODES:
            return None
        raise


def get_bytes(key: str, max_bytes: int) -> bytes:
    body = client().get_object(Bucket=bucket(), Key=key)["Body"]
    data = body.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("object exceeds size limit")
    return data


def put_bytes(key: str, data: bytes, content_type: str) -> None:
    client().put_object(Bucket=bucket(), Key=key, Body=data, ContentType=content_type)
