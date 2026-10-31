"""Original four detectors + optional Korean text recognition (CPU ONNX)."""
from __future__ import annotations

import importlib.util
import logging
from dataclasses import replace
from threading import RLock
from time import perf_counter

import cv2
import numpy as np

from app.geometry import Detection, make_detection
from app.text_replace import rectified_crop

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self._lock = RLock()
        self._initialized = False
        self._recognizer_initialized = False

    def warmup(self) -> None:
        with self._lock:
            if self._initialized:
                return
            from huggingface_hub import hf_hub_download
            from ultralytics import YOLO
            from paddleocr import TextDetection

            # Person (COCO person class 0)
            self.person = YOLO("yolo11n.pt")

            face_path = hf_hub_download(
                repo_id="LibreYOLO/librefacerec-det", filename="librefacerec-det.onnx",
            )
            self.face = cv2.FaceDetectorYN.create(
                face_path, "", (320, 320), 0.6, 0.3, 5000,
            )

            plate_repo = "miraclenote78/kr-plate-detector"
            plate_path = hf_hub_download(plate_repo, "plate_kr_rtmdet_v4.onnx")
            helper_path = hf_hub_download(plate_repo, "rtmdet_onnx.py")
            # Third-party executable code. Review/pin its revision for production.
            spec = importlib.util.spec_from_file_location("_kr_plate_onnx_helper", helper_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("번호판 모델 후처리 스크립트를 불러올 수 없습니다.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.plate = module.OnnxPlateDetector(
                plate_path, conf_large=0.30, conf_tiny=0.30, tiny_frac=0.0,
            )

            self.text = TextDetection(model_name="PP-OCRv5_mobile_det", engine="onnxruntime")
            self._initialized = True

    def warmup_recognizer(self) -> None:
        with self._lock:
            if self._recognizer_initialized:
                return
            from paddleocr import TextRecognition
            # Korean language model: handles Korean, English and digits.
            self.text_recognizer = TextRecognition(
                model_name="korean_PP-OCRv5_mobile_rec", engine="onnxruntime",
            )
            self._recognizer_initialized = True

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.detect_with_timings(image)[0]

    def detect_with_timings(self, image: np.ndarray) -> tuple[list[Detection], dict[str, float]]:
        started = perf_counter()
        self.warmup()
        load_ms = (perf_counter() - started) * 1000
        height, width = image.shape[:2]
        detections: list[Detection] = []
        timings: dict[str, float] = {"model_load": round(load_ms, 3)}

        with self._lock:
            started = perf_counter()
            for result in self.person.predict(source=image, classes=[0], conf=0.25, verbose=False):
                for box in result.boxes:
                    coords = box.xyxy[0].cpu().tolist()
                    score = float(box.conf[0].item())
                    det = make_detection("person", coords, score, width, height)
                    if det:
                        detections.append(det)
            timings["person"] = round((perf_counter() - started) * 1000, 3)

            started = perf_counter()
            self.face.setInputSize((width, height))
            _, faces = self.face.detect(image)
            if faces is not None:
                for face in faces:
                    x, y, w, h = map(float, face[:4])
                    det = make_detection("face", (x, y, x + w, y + h), float(face[14]), width, height)
                    if det:
                        detections.append(det)
            timings["face"] = round((perf_counter() - started) * 1000, 3)

            started = perf_counter()
            for x1, y1, x2, y2, score in self.plate.detect(image):
                det = make_detection("license_plate", (x1, y1, x2, y2), score, width, height)
                if det:
                    detections.append(det)
            timings["license_plate"] = round((perf_counter() - started) * 1000, 3)

            started = perf_counter()
            for result in self.text.predict(input=image, batch_size=1):
                data = result.json
                if "res" in data:
                    data = data["res"]
                polygons = data.get("dt_polys", [])
                scores = data.get("dt_scores", [])
                for idx, polygon in enumerate(polygons):
                    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                    if poly.shape[0] < 3:
                        continue
                    x1, y1 = poly.min(axis=0)
                    x2, y2 = poly.max(axis=0)
                    score = float(scores[idx]) if idx < len(scores) else 1.0
                    det = make_detection(
                        "text", (x1, y1, x2, y2), score, width, height,
                        polygon=poly,
                    )
                    if det:
                        detections.append(det)
            timings["text"] = round((perf_counter() - started) * 1000, 3)
        return detections, timings

    def recognize_text_with_timings(
        self, image: np.ndarray, detections: list[Detection],
    ) -> tuple[list[Detection], dict[str, float]]:
        """Recognize each text region after detection (all failures blur later)."""
        if not any(d.label == "text" for d in detections):
            return detections, {"text_recognition": 0.0, "recognition_load": 0.0}
        started = perf_counter()
        try:
            self.warmup_recognizer()
        except Exception:
            logger.exception("Korean OCR recognizer failed to initialize; text will be blurred")
            return detections, {
                "recognition_load": round((perf_counter() - started) * 1000, 3),
                "text_recognition": 0.0,
            }
        load_ms = round((perf_counter() - started) * 1000, 3)
        started = perf_counter()
        updated: list[Detection] = []
        with self._lock:
            for det in detections:
                if det.label != "text":
                    updated.append(det)
                    continue
                try:
                    crop = rectified_crop(image, det)
                    if crop.shape[0] < 9 or crop.shape[1] < 12:
                        updated.append(det)
                        continue
                    prediction = self.text_recognizer.predict(input=crop, batch_size=1)
                    first = next(iter(prediction), None)
                    if first is None:
                        updated.append(det)
                        continue
                    data = first.json
                    if "res" in data:
                        data = data["res"]
                    text = str(data.get("rec_text") or "").strip()
                    score = float(data.get("rec_score") or 0.0)
                    updated.append(replace(det, recognized_text=text, recognition_score=score))
                except Exception:
                    logger.warning("OCR recognition failed for one region; using blur fallback", exc_info=True)
                    updated.append(det)
        return updated, {
            "recognition_load": load_ms,
            "text_recognition": round((perf_counter() - started) * 1000, 3),
        }
