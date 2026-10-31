from io import BytesIO

import pytest

from app.integration import BlurIntegration, Settings


class DummyBody:

    def __init__(self, data):
        self.stream = BytesIO(data)
        self.closed = False

    def read(self, n):
        return self.stream.read(n)

    def close(self):
        self.closed = True


class DummyS3:

    def __init__(self):
        self.body = DummyBody(b"example")

    def get_object(self, Bucket, Key):
        assert Bucket == "unit-test-bucket"
        assert Key == "images/source/1/example.jpg"

        return {
            "Body": self.body,
            "ContentLength": 7
        }


class DummyResponse:

    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class DummyHttp:

    def __init__(self):
        self.calls = []

    def post(self, url, json, headers):
        self.calls.append(
            ("POST", url, json, headers)
        )

        if url.endswith("/presigned-url"):
            return DummyResponse({
                "statusCode": 200,
                "data": {
                    "uploadUrl": "https://s3.example/upload",
                    "objectKey": "images/blur/1/out.jpg"
                }
            })

        return DummyResponse({
            "statusCode": 200
        })

    def put(
        self,
        url,
        content,
        headers,
        timeout
    ):
        self.calls.append(
            ("PUT", url, content, headers)
        )

        return DummyResponse(None)


def test_download_and_spring_callbacks(
    monkeypatch
):
    monkeypatch.setattr(
        "boto3.client",
        lambda *a, **kw: DummyS3()
    )

    monkeypatch.setattr(
        "httpx.Client",
        lambda **kw: DummyHttp()
    )

    svc = BlurIntegration(
        Settings(
            "unit-test-bucket",
            "https://spring.example",
            "test-internal"
        )
    )

    assert svc.download_source(
        "images/source/1/example.jpg"
    ) == b"example"

    assert svc.s3.body.closed

    assert svc.upload_blurred(
        1,
        b"jpeg-bytes"
    ) == "images/blur/1/out.jpg"

    assert [
        x[0]
        for x in svc.http.calls
    ] == [
        "POST",
        "PUT",
        "POST"
    ]

    assert svc.http.calls[0][2] == {
        "sourceImageId": 1,
        "contentType": "image/jpeg",
        "fileSize": 10
    }

    assert svc.http.calls[1][3] == {
        "Content-Type": "image/jpeg"
    }

    assert svc.http.calls[2][2] == {
        "sourceImageId": 1,
        "objectKey": "images/blur/1/out.jpg"
    }


def test_result_too_large_rejected_before_spring(
    monkeypatch
):
    monkeypatch.setattr(
        "boto3.client",
        lambda *a, **kw: DummyS3()
    )

    monkeypatch.setattr(
        "httpx.Client",
        lambda **kw: DummyHttp()
    )

    svc = BlurIntegration(
        Settings(
            "bucket",
            "https://spring.example",
            "secret"
        )
    )

    with pytest.raises(ValueError):
        svc.upload_blurred(
            1,
            b"X" * (10 * 1024 * 1024 + 1)
        )

    assert svc.http.calls == []