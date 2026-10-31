from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.pipeline import Processed


def test_api_maintains_compatibility_and_optional_auth(monkeypatch):
    import app.main as main

    class FakePipe:
        text_mode = "replace"
        def run(self, image, text_mode):
            return Processed(image.copy(), image.copy(), [],
                             timings_ms={"pipeline_total": 1.0},
                             text_edits={"replaced": 0, "blurred_fallback": 0},
                             text_mode=text_mode)

    monkeypatch.setattr(main, "pipeline", lambda: FakePipe())
    monkeypatch.setenv("BLUR_API_KEY", "test-secret")
    img = Image.fromarray(np.full((30, 40, 3), 120, np.uint8))
    buffer = BytesIO()
    img.save(buffer, "JPEG")
    data = buffer.getvalue()

    with TestClient(app) as client:
        no_auth = client.post('/process', files={"file": ("test.jpg", data, "image/jpeg")})
        assert no_auth.status_code == 401
        result = client.post('/process?text_mode=replace',
                             headers={"X-API-Key": "test-secret"},
                             files={"file": ("test.jpg", data, "image/jpeg")})
        assert result.status_code == 200
        body = result.json()
        assert body['blurred_image_base64']
        assert body['preview_image_base64']
        assert body['counts']['text'] == 0
        assert body['text_mode'] == 'replace'
        binary = client.post('/process/image?text_mode=blur',
                             headers={"X-API-Key": "test-secret"},
                             files={"file": ("test.jpg", data, "image/jpeg")})
        assert binary.status_code == 200
        assert binary.headers['content-type'] == 'image/jpeg'
        assert binary.headers['x-text-mode'] == 'blur'
