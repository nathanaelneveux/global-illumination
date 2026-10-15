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
    parser.add_argument("--counts", type=int, nargs="+", help="selected checkpoints, otherwise all")
    parser.add_argument("--labels", nargs="+", default=["samples"],
                        help="multiple labels produce matched rows with a reference column")
    parser.add_argument("--reference-per-row", action="store_true",
                        help="repeat the reference beside each checkpoint, even with one label")
    args = parser.parse_args()
    sweep = json.loads((args.run / "cave-sample-sweep.json").read_text())
    reference = json.loads((args.run / "cave-reference.json").read_text())
    if not reference["qualification"]["qualified"]:
        raise ValueError("reference has not qualified")
    available = [p["paths"] for p in sweep["windows"][args.window]["prefixes"]]
    counts = args.counts or available
    if any(n not in available for n in counts):
        raise ValueError("requested checkpoint not present in sweep")
    reference_paths = reference["paths_per_replica"] * reference["replicas"]
    reference_frame = (f"Reference · {reference_paths:,} samples", "cave-reference.png")
    matched_rows = len(args.labels) > 1 or args.reference_per_row
    frames = []
    for n in counts:
        frames.extend((f"{label} · {n:,} paths", f"{label}-{n}-window-{args.window}.png")
                      for label in args.labels)
        if matched_rows:
            frames.append(reference_frame)
    if not matched_rows:
        frames.append(reference_frame)
    width, height, caption, columns = 640, 360, 36, 4
    if matched_rows:
        columns = len(args.labels) + 1
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
    label = "sample" if args.labels == ["samples"] else "-".join(args.labels)
    if args.reference_per_row:
        label += "-paired-" + "-".join(map(str, counts))
    output = args.run / f"{label}-overview-window-{args.window}.png"
    sheet.save(output)
    print(output.resolve())


if __name__ == "__main__":
    main()
