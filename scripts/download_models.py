"""Run while Internet access is available: python -m scripts.download_models"""

from app.detectors import ModelService


if __name__ == "__main__":
    models = ModelService()

    print(
        "Initializing 4 detectors "
        "(person, face, plate, text)..."
    )

    models.warmup()

    print(
        "All detection models initialized."
    )