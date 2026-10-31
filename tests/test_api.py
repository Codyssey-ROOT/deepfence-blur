from io import BytesIO
from unittest.mock import Mock

import numpy as np

from PIL import Image
from fastapi.testclient import TestClient

from app.main import (
    app,
    process,
    ProcessRequest
)
from app.pipeline import Processed


SOURCE_KEY = (
    "images/source/1/"
    "123e4567-e89b-12d3-a456-426614174000.jpg"
)


def image_bytes():
    out = BytesIO()

    Image.fromarray(
        np.full(
            (30, 40, 3),
            120,
            dtype=np.uint8
        )
    ).save(out, "JPEG")

    return out.getvalue()


def test_process_download_before_ack_and_background_only_after(
    monkeypatch
):
    import app.main as main

    calls = []
    raw = image_bytes()

    class FakeIntegration:

        def download_source(self, object_key):
            assert object_key == SOURCE_KEY
            calls.append("download")
            return raw

    class CapturingBackgroundTasks:

        def add_task(self, fn, *args):
            calls.append("register")
            self.fn = fn
            self.args = args

    monkeypatch.setattr(
        main,
        "integration",
        lambda: FakeIntegration()
    )

    tasks = CapturingBackgroundTasks()

    result = process(
        ProcessRequest(
            imageId=7,
            objectKey=SOURCE_KEY
        ),
        tasks
    )

    assert result == {
        "statusCode": 200,
        "message": "OK"
    }

    assert calls == [
        "download",
        "register"
    ]

    assert tasks.fn is main.process_blur_background
    assert tasks.args == (7, raw)


def test_process_auth_validation_and_background_work(
    monkeypatch
):
    import app.main as main

    calls = []
    raw = image_bytes()

    class FakeIntegration:

        def download_source(self, object_key):
            calls.append("download")
            return raw

        def upload_blurred(self, image_id, jpeg):
            assert image_id == 7
            assert jpeg[:3] == b"\xff\xd8\xff"

            calls.append("upload_complete")

            return "images/blur/1/result.jpg"

    class FakeDetector:
        _initialized = False

    class FakePipe:
        detector = FakeDetector()

        def run(self, image):
            calls.append("detect_blur")

            return Processed(
                image,
                image,
                [],
                {"pipeline_total": 1.0}
            )

    monkeypatch.setattr(
        main,
        "integration",
        lambda: FakeIntegration()
    )

    monkeypatch.setattr(
        main,
        "pipeline",
        lambda: FakePipe()
    )

    monkeypatch.setenv(
        "BLUR_API_KEY",
        "test-secret"
    )

    with TestClient(app) as client:

        response = client.post(
            "/process",
            json={
                "imageId": 7,
                "objectKey": SOURCE_KEY
            }
        )

        assert response.status_code == 401

        response = client.post(
            "/process",
            json={
                "imageId": 7,
                "objectKey": "../secret"
            },
            headers={
                "X-API-Key": "test-secret"
            }
        )

        assert response.status_code == 422

        response = client.post(
            "/process",
            json={
                "imageId": 7,
                "objectKey": SOURCE_KEY
            },
            headers={
                "X-API-Key": "test-secret"
            }
        )

        assert response.status_code == 200

        assert response.json() == {
            "statusCode": 200,
            "message": "OK"
        }

        assert calls == [
            "download",
            "detect_blur",
            "upload_complete"
        ]

        response = client.post(
            "/process/image",
            files={
                "file": (
                    "test.jpg",
                    raw,
                    "image/jpeg"
                )
            },
            headers={
                "X-API-Key": "test-secret"
            }
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"


def test_s3_download_error_does_not_enqueue(
    monkeypatch
):
    import app.main as main

    from fastapi import HTTPException

    class FakeIntegration:

        def download_source(self, key):
            raise ValueError("too large")

    class CapturingTasks:

        def add_task(self, *args):
            raise AssertionError(
                "must not enqueue"
            )

    monkeypatch.setattr(
        main,
        "integration",
        lambda: FakeIntegration()
    )

    try:
        main.process(
            ProcessRequest(
                imageId=1,
                objectKey=SOURCE_KEY
            ),
            CapturingTasks()
        )

        assert False, "Expected HTTPException"

    except HTTPException as e:
        assert e.status_code == 413


def test_background_exception_is_logged_not_propagated(
    monkeypatch,
    caplog
):
    import app.main as main

    class BrokenPipeline:

        def run(self, image):
            raise RuntimeError(
                "model failure"
            )

    monkeypatch.setattr(
        main,
        "pipeline",
        lambda: BrokenPipeline()
    )

    main.process_blur_background(
        5,
        image_bytes()
    )

    assert (
        "Background blur failed imageId=5"
        in caplog.text
    )