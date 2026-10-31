import re
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from PIL import Image

from app import s3
from app.geometry import Detection
from app.main import app
from app.pipeline import Pipeline

BUCKET = "test-bucket"
CDN = "https://cdn.example.com"
UUID_KEY = re.compile(r"^images/blur/[0-9a-f-]{36}\.(png|jpg|jpeg)$")


@pytest.fixture
def client(monkeypatch):
    env = {
        "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_REGION": "ap-northeast-2", "S3_BUCKET": BUCKET, "CDN_BASE_URL": CDN + "/",
        "S3_KEY_PREFIX": "images/blur", "MAX_FILE_SIZE_BYTES": "12582912",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("BLUR_API_KEY", raising=False)

    import app.main as main

    class FakeDetector:
        def detect(self, image):
            return [Detection("license_plate", (5, 5, 30, 20), 0.9)]

    monkeypatch.setattr(main, "pipeline", lambda: Pipeline(FakeDetector()))
    with mock_aws():
        s3.client.cache_clear()
        s3.client().create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "ap-northeast-2"})
        with TestClient(app) as test_client:
            yield test_client
    s3.client.cache_clear()


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.full((40, 60, 3), 200, np.uint8)).save(buffer, "PNG")
    return buffer.getvalue()


def test_presigned_url_ok(client):
    res = client.post("/blur/images/presigned-url",
                      json={"contentType": "image/jpeg", "fileExt": "jpg", "fileSize": 1024})
    assert res.status_code == 200
    body = res.json()
    assert body["statusCode"] == 200 and body["message"] == "OK"
    data = body["data"]
    assert UUID_KEY.match(data["objectKey"])
    assert data["publicUrl"] == f"{CDN}/{data['objectKey']}"
    query = parse_qs(urlparse(data["uploadUrl"]).query)
    signed = query["X-Amz-SignedHeaders"][0].split(";")
    assert "content-type" in signed and "content-length" in signed
    assert query["X-Amz-Expires"] == ["300"]


@pytest.mark.parametrize("payload", [
    {"contentType": "image/png", "fileExt": "jpg", "fileSize": 10},       # mismatch
    {"contentType": "image/jpeg", "fileExt": "png", "fileSize": 10},      # mismatch
    {"contentType": "image/webp", "fileExt": "webp", "fileSize": 10},     # not allowed
    {"contentType": "image/png", "fileExt": "png", "fileSize": 0},        # too small
    {"contentType": "image/png", "fileExt": "png", "fileSize": 12582913}, # too large
    {"contentType": "image/png", "fileExt": "png", "fileSize": "10"},     # wrong type
    {"contentType": "image/png", "fileExt": "png"},                       # missing
])
def test_presigned_url_rejects_bad_input(client, payload):
    res = client.post("/blur/images/presigned-url", json=payload)
    assert res.status_code == 400
    body = res.json()
    assert body["statusCode"] == 400 and body["message"] and "detail" not in body


def test_new_routes_require_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("BLUR_API_KEY", "test-secret")
    res = client.post("/blur/images/complete", json={"objectKey": "images/blur/a.png"})
    assert res.status_code == 401
    assert res.json() == {"statusCode": 401, "message": "Invalid X-API-Key"}


def test_complete_processes_in_background(client):
    key = "images/blur/0b6f5f0e-3a0e-4c55-9d5e-1f2a3b4c5d6e.png"
    s3.client().put_object(Bucket=BUCKET, Key=key, Body=png_bytes(), ContentType="image/png")

    res = client.post("/blur/images/complete", json={"objectKey": key})
    assert res.status_code == 200
    assert res.json() == {"statusCode": 200, "message": "OK"}

    # TestClient runs BackgroundTasks before returning, so the result must exist now.
    result = s3.client().get_object(
        Bucket=BUCKET, Key="images/blur/results/0b6f5f0e-3a0e-4c55-9d5e-1f2a3b4c5d6e.jpg")
    assert result["ContentType"] == "image/jpeg"
    assert result["Body"].read()[:2] == b"\xff\xd8"


def test_complete_missing_object_returns_404(client):
    res = client.post("/blur/images/complete", json={"objectKey": "images/blur/nope.png"})
    assert res.status_code == 404
    assert res.json()["statusCode"] == 404


@pytest.mark.parametrize("key", ["", "   ", "other/a.png", "images/blurx/a.png",
                                 "images/blur/../secret.png", "images/blur/a..png"])
def test_complete_rejects_bad_keys(client, key):
    res = client.post("/blur/images/complete", json={"objectKey": key})
    assert res.status_code == 400
    assert res.json()["statusCode"] == 400


def test_complete_rejects_oversized_object(client, monkeypatch):
    key = "images/blur/big.png"
    s3.client().put_object(Bucket=BUCKET, Key=key, Body=png_bytes())
    monkeypatch.setenv("MAX_FILE_SIZE_BYTES", "10")
    res = client.post("/blur/images/complete", json={"objectKey": key})
    assert res.status_code == 400


def test_existing_routes_keep_default_error_body(client, monkeypatch):
    monkeypatch.setenv("BLUR_API_KEY", "test-secret")
    res = client.post("/process", files={"file": ("a.png", png_bytes(), "image/png")})
    assert res.status_code == 401
    assert res.json() == {"detail": "Invalid X-API-Key"}
