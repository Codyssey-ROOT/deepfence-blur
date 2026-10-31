"""Run while internet is available: python -m scripts.download_models"""
from app.detectors import ModelService

if __name__ == "__main__":
    models = ModelService()
    print("Initializing 4 original pretrained detectors...")
    models.warmup()
    print("Initializing Korean text recognizer (ONNX)...")
    models.warmup_recognizer()
    print("All model loaders initialized.")
