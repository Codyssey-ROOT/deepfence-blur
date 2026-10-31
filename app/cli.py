"""Local CLI: python -m app.cli --input photo.jpg --output output/result.jpg"""
from __future__ import annotations

import argparse
import json

from pathlib import Path

from app.detectors import ModelService
from app.image_ops import decode_image, encode_jpg
from app.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepFence text and plate blur"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/processed.jpg")
    )

    args = parser.parse_args()

    result = Pipeline(
        ModelService()
    ).run(
        decode_image(
            args.input.read_bytes()
        )
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    args.output.write_bytes(
        encode_jpg(
            result.blurred
        )
    )

    preview = args.output.with_name(
        args.output.stem + "_preview.jpg"
    )

    preview.write_bytes(
        encode_jpg(
            result.preview
        )
    )

    info = args.output.with_suffix(
        ".json"
    )

    info.write_text(
        json.dumps(
            {
                "detections": [
                    d.as_dict()
                    for d in result.detections
                ],
                "timings_ms": result.timings_ms
            },
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        f"Processed: {args.output}\n"
        f"Preview: {preview}\n"
        f"Metadata: {info}"
    )


if __name__ == "__main__":
    main()