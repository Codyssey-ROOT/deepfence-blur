# DeepFence — S3-driven Blur FastAPI

사진 속에서 **나를 특정할 수 있는 단서(차량 번호판, 글자·간판·신분증 등)를 찾아 자동으로 블러 처리**하는 DeepFence의 개인정보 보호 서버입니다.

사용자가 사진을 올리면 API 서버(Spring)가 이 서버에 처리를 요청하고, 블러 처리된 이미지는 다시 S3에 저장되어 다음 단계인 [FaceShield](https://github.com/Codyssey-ROOT/deepfence-faceshield)로 넘어갑니다.

- 서비스 전체 소개: [Codyssey-ROOT](https://github.com/Codyssey-ROOT)

## 기술 개요

```
원본 이미지 → 객체 탐지(사람·얼굴·번호판·글자) → 번호판·글자 영역 블러 → 결과 업로드
```

| 단계 | 방식 |
| --- | --- |
| 탐지 | 사람(YOLO11n), 얼굴(LibreFaceRec), 한국 번호판(RTMDet), 글자 영역(PP-OCRv5 Mobile Detection) 모델을 CPU ONNX 환경에서 실행 |
| 블러 | 번호판과 글자 영역만 블러 처리하고, 사람과 얼굴은 위치만 탐지 (얼굴은 FaceShield 단계에서 보호) |
| 처리 방식 | 원본 다운로드 성공 즉시 응답하고, 탐지·블러·업로드는 백그라운드에서 수행 |
| 연동 | S3에서 원본을 받고, Spring이 발급한 Presigned URL로 결과를 업로드 |

## 기술 스택

| 구분 | 기술 |
| --- | --- |
| 언어 | Python 3.11 |
| 서버 | FastAPI, Uvicorn |
| 모델 실행 | ONNX Runtime, Ultralytics, PaddleOCR |
| 이미지 처리 | OpenCV, Pillow, NumPy |
| 모델 다운로드 | Hugging Face Hub |
| 외부 연동 | boto3 (S3), httpx (Spring API) |
| 인프라 | Docker, Docker Compose |

## API

### POST /process

Spring에서 다음 JSON을 전달합니다.

{
  "imageId": 1,
  "objectKey": "images/source/1/<UUID>.jpg"
}

Header:

X-API-Key: {BLUR_API_KEY}

처리 순서:

1. S3에서 원본 이미지 다운로드
2. 다운로드 성공 시 HTTP 200 OK 반환
3. BackgroundTasks에서 이미지 탐지 및 블러
4. Spring에 Presigned URL 발급 요청
5. S3에 블러 이미지 PUT 업로드
6. Spring에 이미지 업로드 완료 통보

성공 응답:

{
  "statusCode": 200,
  "message": "OK"
}

200 OK는 블러 처리 완료가 아니라
원본 이미지 다운로드 성공과 백그라운드 작업 등록을 의미합니다.

### POST /process/image

multipart/form-data의 file 필드를 받아
블러 이미지 JPEG 바이너리를 반환합니다.

로컬 테스트를 위해 유지하는 API입니다.

### GET /health

서버 상태 및 모델 초기화 여부를 반환합니다.

## 탐지 모델

- person: YOLO11n
- face: LibreFaceRec ONNX
- license_plate: RTMDet ONNX
- text: PP-OCRv5 Mobile Detection

## 블러 처리 대상

- text
- license_plate

사람과 얼굴은 탐지만 수행합니다.

## Environment

.env.example을 복사해 .env를 작성합니다.

필수 환경변수:

- BLUR_API_KEY
- INTERNAL_API_KEY
- SPRING_BASE_URL
- AWS_REGION
- AWS_S3_BUCKET

AWS 자격 증명은 IAM Role 또는
AWS SDK 기본 자격 증명 체인을 이용합니다.

FastAPI는 원본 이미지 경로에 대한
s3:GetObject 권한이 필요합니다.

Spring에서는 다음 API를 구현해야 합니다.

- POST /blur/images/presigned-url
- POST /blur/images/complete

## Local tests

pip install -r requirements.txt
pip install pytest
pytest -q

## Local server

uvicorn app.main:app --host 0.0.0.0 --port 8000

## Model prewarm

python -m scripts.download_models

## CLI

python -m app.cli --input sample.jpg --output out.jpg

## Reliability

BackgroundTasks는 FastAPI 프로세스 내부에서 실행됩니다.

FastAPI 재시작 또는 장애가 발생하면
200 OK 이후에도 작업이 유실될 수 있습니다.

현재 백그라운드 처리 실패는 로그로 기록하며
Spring에 실패 상태를 통보하지 않습니다.

향후 실패 콜백 또는 재처리 기능이 필요합니다.

CPU 모델 추론은 내부 락으로 보호됩니다.
초기에는 Uvicorn worker 1개를 권장합니다.

## 외부 자원 활용 내역

| 구분 | 명칭 | 용도 | 라이선스 |
| --- | --- | --- | --- |
| 오픈소스 | FastAPI, Uvicorn | API 서버 | MIT / BSD-3-Clause |
| 오픈소스 | OpenCV, NumPy, Pillow | 이미지 처리 | Apache 2.0 / BSD-3-Clause / HPND |
| 오픈소스 | ONNX Runtime | 모델 추론 | MIT |
| 오픈소스 | PaddleOCR (PP-OCRv5 Mobile Detection) | 글자 영역 탐지 | Apache 2.0 |
| 오픈소스 | Ultralytics (YOLO11n) | 사람 탐지 | AGPL-3.0 |
| 사전 학습 모델 | LibreFaceRec (`LibreYOLO/librefacerec-det`) | 얼굴 탐지 | 모델 저장소 라이선스 참고 |
| 사전 학습 모델 | 한국 번호판 RTMDet (`miraclenote78/kr-plate-detector`) | 번호판 탐지 | 모델 저장소 라이선스 참고 |
| 외부 서비스 | AWS S3 | 원본 다운로드, 결과 업로드 | 상용 |
