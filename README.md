# DeepFence — Blur FastAPI + Natural Text Replacement

DeepFence **이미지 탐지 및 치환용 CPU FastAPI**. 기존 `image-detection-mvp.zip`과 timing patch를 기반으로 수정한 완성 소스입니다. **Ncloud GPU의 FaceShield 서비스는 포함하지 않습니다.**

## 핵심 변경점

- 기존 **YOLO11n(사람), YuNet(얼굴), 한국 번호판 RTMDet ONNX, PaddleOCR PP-OCRv5 TextDetection** 유지
- 기존 MVP는 **글자의 위치만 탐지**했습니다. 새 버전은 **한국어 OCR 인식 모델** `korean_PP-OCRv5_mobile_rec`(ONNX)을 추가해 실제 글자를 읽습니다.
- 인식된 텍스트는 `비슷한 한글/영문/숫자`로 변환한 뒤 **OpenCV inpaint로 원본 지우기 → Pillow 한글 글꼴로 새 글자 생성 → 원근 변환 합성** 방식으로 수정합니다.
- OCR 신뢰도가 낮거나 글꼴/합성에 문제가 있으면 **기존 Gaussian Blur로 폴백**합니다.
- **번호판은 항상 블러**, 사람과 얼굴은 **탐지만 수행**합니다.
- 기존 FastAPI API 경로와 응답의 `blurred_image_base64`, `preview_image_base64`, `detections`, `counts`, `timings_ms` 유지. `text_mode`, `text_edits` 추가.
- 이미지 속 원문과 가상 치환 문자열은 **기본 API 응답과 로그에 노출하지 않습니다.**

| 작업 | 사용하는 도구 |
|---|---|
| 사람 위치 | Ultralytics YOLO11n |
| 얼굴 위치 | OpenCV FaceDetectorYN / YuNet |
| 차량 번호판 위치 | 한국 RTMDet ONNX |
| 글자 위치 | PaddleOCR `PP-OCRv5_mobile_det` (ONNX) |
| **글자 인식** | PaddleOCR `korean_PP-OCRv5_mobile_rec` (ONNX) |
| 기존 글자 제거 | OpenCV `inpaint(INPAINT_TELEA)` |
| 새 글자 합성 | Pillow/NanumGothic + OpenCV `warpPerspective` |
| 번호판 및 실패 영역 | OpenCV Gaussian Blur |

## 1. 로컬 실행 (macOS / Linux)

권장 Python: **3.11**

```bash
cd deepfence-blur-fastapi
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

한글 폰트가 필요합니다. Ubuntu: `sudo apt install fonts-nanum` / macOS: 시스템 `AppleSDGothicNeo.ttc` 자동 탐색. 필요한 경우 `TEXT_FONT_PATH`를 지정하세요.

```bash
# 인터넷에서 기존 탐지 모델 4개 및 새 한국어 OCR 인식 모델 다운로드
python -m scripts.download_models

# 직접 파일 실행 (유사 문자 치환)
python -m app.cli --input samples/example.jpg --output output/processed.jpg --text-mode replace

# 원래 블러 방식과 비교
python -m app.cli --input samples/example.jpg --output output/blur.jpg --text-mode blur

# API 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API Swagger: http://localhost:8000/docs

## 2. API 사용

`GET /health`: 프로세스 응답과 모델 로딩 상태. **로드 여부는 실제 추론 성공을 의미하지 않습니다.**

`POST /process`: Multipart 필드 이름 `file`, `text_mode=replace|blur` 쿼리 매개변수(기본 `replace`). JSON 반환.

`POST /process/image`: 동일 입력, 결과 JPEG 바이너리 반환.

```bash
curl -F "file=@samples/example.jpg" \
  "http://localhost:8000/process/image?text_mode=replace" \
  --output result.jpg

curl -F "file=@samples/example.jpg" \
  "http://localhost:8000/process?text_mode=blur" \
  --output result.json
```

새 응답 필드 예:

```json
{
  "counts": {"person": 5, "face": 4, "license_plate": 1, "text": 17},
  "text_mode": "replace",
  "text_edits": {"replaced": 11, "blurred_fallback": 6, "plate_blurred": 1},
  "blurred_image_base64": "...",
  "preview_image_base64": "...",
  "detections": [],
  "timings_ms": {"person": 20, "text": 27, "text_recognition": 80, "blur": 32}
}
```

위 숫자는 예시일 뿐 실측 결과가 아닙니다. 이름은 호환성 때문에 `blurred_image_base64`로 유지했지만 **replace 모드에서는 새 글자로 편집된 결과**입니다.

**선택적 내부 API 인증**: 환경변수 `BLUR_API_KEY`가 설정되어 있으면 POST 요청에 `X-API-Key: <값>` 헤더가 필요합니다.

```bash
curl -H 'X-API-Key: 비밀키' \
  -F "file=@samples/example.jpg" \
  'http://localhost:8000/process/image' --output result.jpg
```

## 3. AWS Blur EC2에 Docker Hub 이미지로 배포

EC2: Ubuntu 24.04, x86_64, 8GB RAM 권장. Docker Engine/Compose 사전 설치. Spring EC2와 동일 VPC에 배치하고 통신은 **Blur EC2의 Private IP:8000** 이용. Blur EC2의 보안 그룹 **8000 인바운드는 Spring EC2 보안 그룹만 허용**합니다. SSH 22는 본인 IP만 허용하고 인터넷 전체에는 8000을 열지 않습니다.

### Mac Apple Silicon에서 linux/amd64 이미지로 빌드 및 Docker Hub에 푸시

```bash
# Docker Hub 로그인
docker login
# <dockerhub-id>를 실제 계정명으로 교체
docker buildx build --platform linux/amd64 \
  -t <dockerhub-id>/deepfence-blur-fastapi:latest \
  --push .
```

### EC2에서 실행

배포할 때 프로젝트의 `compose.yaml`, `.env`만 필요합니다. Docker 이미지에 앱 소스가 포함됩니다.

```bash
mkdir -p ~/deepfence-blur
cd ~/deepfence-blur
# compose.yaml 과 .env.example 업로드/복사 후
cp .env.example .env
# .env 안의 BLUR_IMAGE 값을 Docker Hub 이미지 주소로 수정
# BLUR_API_KEY에 임의의 긴 비밀 문자열 설정 권장

docker compose pull
docker compose up -d
docker compose ps
# 처음 한 번 모델 다운로드. 첫 실행 시 Hugging Face 등에 인터넷 접근이 필요합니다.
docker compose exec blur-fastapi python -m scripts.download_models

docker compose logs -f blur-fastapi
```

테스트:

```bash
# Blur EC2 내부
curl http://localhost:8000/health

# Spring EC2에서 (동일 VPC의 Blur Private IP 이용)
curl http://<BLUR_PRIVATE_IP>:8000/health
```

`BLUR_API_KEY`를 설정했다면 Spring에서 FastAPI 요청 헤더에 동일한 키를 넣어야 합니다. 모델은 시작 시 자동 로드하지 않고 첫 추론 또는 사전 다운로드 명령 시 로드합니다. **CPU에서 첫 요청은 수 초 이상 걸릴 수 있습니다.**

서버가 AMD64라면 Docker Hub에서 ARM64 이미지를 실수로 받아 실행하지 않도록 빌드시 `--platform linux/amd64`를 유지하세요.

### Compose 이미지 예시

```yaml
services:
  blur-fastapi:
    image: ${BLUR_IMAGE:-deepfence-blur-fastapi:latest}
    ports:
      - "8000:8000"
    env_file: [.env]
    volumes:
      - deepfence_model_cache:/models
volumes:
  deepfence_model_cache:
```

프로젝트 ZIP에 실제로 사용할 전체 `compose.yaml`이 포함되어 있습니다.

## 4. 구성 파일

```
deepfence-blur-fastapi/
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .env.example
├── requirements.txt
├── README.md
├── app/
│   ├── main.py         # REST API
│   ├── detectors.py    # 모델 로딩 + 객체 탐지 + 한국어 인식
│   ├── geometry.py     # 좌표 및 탐지 자료형
│   ├── text_replace.py # 유사 글자 생성, inpaint, 한글 합성
│   ├── image_ops.py    # 이미지 입력/출력, 번호판 블러, 미리보기
│   ├── pipeline.py     # 전체 처리 순서
│   └── cli.py          # 파일 기반 실행
├── scripts/
│   └── download_models.py
└── tests/
```

## 5. 중요 한계

- **원본 프로젝트의 OCR는 detection-only**였습니다. 한국어 문자 치환을 위해 인식 모델을 새로 추가했으며 기존 Mac에서 측정한 약 419ms 수치는 **더 이상 그대로 적용되지 않습니다.** CPU에서 새 버전을 다시 측정해야 합니다.
- **Pillow/OpenCV의 간단한 방식이라 완전히 자연스러운 결과를 보장하지 못합니다.** 복잡한 바탕, 입체 글자, 반사, 작은 글씨는 합성 흔적이 남을 수 있습니다.
- OCR 탐지 누락은 그대로 남을 수 있습니다. 인식 성공해도 가상 문자가 개인정보 보호를 수학적으로 보장하지 않으며 임의로 만든 전화번호/주소 등이 우연히 실제 값과 일치할 수 있습니다.
- 강한 비식별화가 필요한 문서는 불투명 마스킹과 사람 검수를 사용하세요.
- 입력 이미지는 최대 12MB / 25MP, JPEG·PNG·WebP 지원. EXIF 방향 보정 후 처리하고 결과 JPEG에는 원본 EXIF를 포함하지 않습니다.
- 모델 최초 다운로드 시 Docker Hub/Hugging Face 네트워크 접근이 필요합니다. 공개 저장소에서 가져오는 번호판 모델의 `rtmdet_onnx.py`는 **실행 가능한 타사 코드**이므로 배포 전 출처 및 해시를 고정해 검토해야 합니다.
- Ultralytics YOLO11 라이선스(AGPL 또는 상용) 및 각 모델 라이선스를 별도로 확인하세요.
- 모델 전체의 실제 추론과 EC2 상 Docker 빌드는 이 배포 ZIP을 만드는 환경에서 실행하지 않았습니다. 기본 단위/API 통합 테스트는 **학습된 가중치 없이** 실행합니다.

## 6. 단위 테스트

```bash
python -m pip install pytest httpx
pytest -q
```

## 문서/모델 출처

- https://docs.ultralytics.com/models/yolo11/
- https://huggingface.co/LibreYOLO/librefacerec-det
- https://huggingface.co/miraclenote78/kr-plate-detector
- https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det
- https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html
- https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec
