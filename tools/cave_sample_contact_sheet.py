#!/usr/bin/env python3
"""Labeled overview of every checkpoint; originals remain unchanged."""
import argparse
import json
from PIL import Image, ImageDraw, ImageFont


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--window", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    sweep = json.loads((args.run / "cave-sample-sweep.json").read_text())
    reference = json.loads((args.run / "cave-reference.json").read_text())
    if not reference["qualification"]["qualified"]:
        raise ValueError("reference has not qualified")
    counts = [p["paths"] for p in sweep["windows"][args.window]["prefixes"]]
    frames = [(f"{n:,} samples", f"samples-{n}-window-{args.window}.png") for n in counts]
    reference_paths = reference["paths_per_replica"] * reference["replicas"]
    frames.append((f"Reference · {reference_paths:,} samples", "cave-reference.png"))
    width, height, caption, columns = 640, 360, 36, 4
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + caption)), "#17191c")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=23)
    for i, (label, file) in enumerate(frames):
        x, y = (i % columns) * width, (i // columns) * (height + caption)
        with Image.open(args.run / file) as image:
            if image.size != (2560, 1440):
                raise ValueError(f"unexpected capture dimensions: {file}")
            sheet.paste(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), (x, y + caption))
        draw.text((x + 10, y + 5), label, fill="#f5f5f5", font=font)
    output = args.run / f"sample-overview-window-{args.window}.png"
    sheet.save(output)
    print(output.resolve())


if __name__ == "__main__":
    main()
