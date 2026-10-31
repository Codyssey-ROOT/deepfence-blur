"""Local CLI: python -m app.cli --input photo.jpg --output output/result.jpg"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.detectors import ModelService
from app.image_ops import decode_image, encode_jpg
from app.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepFence image detector and synthetic text editor")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/processed.jpg"))
    parser.add_argument("--text-mode", choices=["replace", "blur"], default="replace")
    args = parser.parse_args()

    result = Pipeline(ModelService(), text_mode=args.text_mode).run(decode_image(args.input.read_bytes()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encode_jpg(result.blurred))
    preview = args.output.with_name(args.output.stem + "_preview.jpg")
    preview.write_bytes(encode_jpg(result.preview))
    info = args.output.with_suffix(".json")
    info.write_text(
        json.dumps({
            "detections": [d.as_dict() for d in result.detections],
            "text_mode": result.text_mode,
            "text_edits": result.text_edits,
            "timings_ms": result.timings_ms,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Processed: {args.output}\nPreview: {preview}\nMetadata: {info}")
    print("Timings (ms):")
    for key, value in result.timings_ms.items():
        print(f"  {key:>20}: {value:10.3f}")


if __name__ == "__main__":
    main()
