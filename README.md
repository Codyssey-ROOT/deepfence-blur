# DeepFence — S3-driven Blur FastAPI

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