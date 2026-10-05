r"""
frame_sheets.py

Puts all of one fish's raw frames onto a single image, numbered in time
order, so you can look through a whole video at a glance.

WHAT IT MAKES
  results\frame_sheets\<video>_<fish_id>.png   one sheet per fish

By default each frame is CROPPED AROUND THE FISH, using its landmark file.
The raw frames are tall portrait shots with a small fish in them, so ten
whole frames on one sheet would leave the fish tiny. Every frame of a fish
gets the same size of crop, centred on wherever the fish is in that frame,
so the fish appears at the same size in every picture. Frames that haven't
been landmarked yet are centred on where the fish was in that video's
other frames.

OPTIONS
  --full       whole frames instead of crops
  --outline    draw your landmark outline on each frame (useful for
               checking landmarking at a glance)
  --only GH010994.MP4    just one video
  --jpg        JPEG instead of PNG

HOW TO RUN (paths already point at APHP white)
    python frame_sheets.py
"""

import argparse
import csv
import math
import os
import re
import sys

import cv2
import numpy as np

# =====================  SETTINGS  =====================

BASE = r"C:\Users\lamoo\ColorChangeRates2026\files\APHP\white"
FRAMES_DIR = os.path.join(BASE, "white_frames")
LANDMARK_DIR = os.path.join(BASE, "white_landmarks")
OUTPUT_DIR = os.path.join(BASE, "results", "frame_sheets")

CROP_PADDING = 0.35     # extra space around the fish, as a fraction of its size
WIDE_CELL = 620         # cell width when crops are wider than tall
TALL_CELL = 260         # cell width for portrait images (--full)

# The 43 landmarks in the order they run around the fish, for --outline.
PERIMETER_ORDER = [0] + list(range(7, 16)) + [1, 16, 2, 17, 18, 3] + \
    list(range(19, 28)) + [4, 28, 5, 29, 30, 31, 6] + list(range(32, 43))

# ========================================================


def strip_extension(name):
    return re.sub(r"\.[A-Za-z0-9]+$", "", str(name))


def read_tps(path, image_height):
    """TPS -> (N, 2) image coordinates. TPS counts y up from the bottom."""
    points = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2 and "=" not in line:
                try:
                    points.append([float(parts[0]), image_height - float(parts[1])])
                except ValueError:
                    pass
    return np.array(points) if points else None


def put(img, text, org, scale, colour=(235, 235, 235), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thickness, cv2.LINE_AA)


def crop_around(image, centre, size):
    """A size[0] x size[1] window centred on `centre`, kept inside the
    image where possible; any part that falls outside is filled grey."""
    h, w = image.shape[:2]
    cw, ch = int(size[0]), int(size[1])
    x0 = int(round(centre[0] - cw / 2))
    y0 = int(round(centre[1] - ch / 2))
    x0 = min(max(x0, 0), max(0, w - cw))
    y0 = min(max(y0, 0), max(0, h - ch))
    out = np.full((ch, cw, 3), 40, np.uint8)
    src = image[y0:y0 + ch, x0:x0 + cw]
    out[:src.shape[0], :src.shape[1]] = src
    return out, (x0, y0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", default=FRAMES_DIR)
    parser.add_argument("--landmarks", default=LANDMARK_DIR)
    parser.add_argument("--out", default=OUTPUT_DIR)
    parser.add_argument("--full", action="store_true", help="whole frames, no cropping")
    parser.add_argument("--outline", action="store_true",
                        help="draw the landmark outline on each frame")
    parser.add_argument("--only", default=None, help="one video, e.g. GH010994.MP4")
    parser.add_argument("--jpg", action="store_true")
    options = parser.parse_args()

    manifest_path = os.path.join(options.frames, "frames_manifest.csv")
    if not os.path.exists(manifest_path):
        sys.exit(f"No frames_manifest.csv in {options.frames}")
    with open(manifest_path, newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))

    fish_ids = {}
    lm_csv = os.path.join(options.landmarks, "landmarks.csv")
    if os.path.exists(lm_csv):
        with open(lm_csv, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("fish_id"):
                    fish_ids[strip_extension(row["image"])] = row["fish_id"]

    groups = {}
    for row in manifest:
        if os.path.exists(os.path.join(options.frames, row["image"])):
            groups.setdefault(row["video"], []).append(row)
    if options.only:
        groups = {v: r for v, r in groups.items() if v == options.only}
        if not groups:
            sys.exit(f"No frames on disk for {options.only}")

    os.makedirs(options.out, exist_ok=True)
    ext = ".jpg" if options.jpg else ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if options.jpg else [cv2.IMWRITE_PNG_COMPRESSION, 3]

    for number, (video, rows) in enumerate(sorted(groups.items()), start=1):
        rows.sort(key=lambda r: float(r["seconds"]))
        frames = []
        for row in rows:
            image = cv2.imread(os.path.join(options.frames, row["image"]))
            if image is None:
                continue
            tps = os.path.join(options.landmarks, "Landmarks", row["image"] + "_LM.TPS")
            points = read_tps(tps, image.shape[0]) if os.path.exists(tps) else None
            frames.append({"row": row, "image": image, "points": points})
        if not frames:
            continue

        fish_id = next((fish_ids.get(strip_extension(f["row"]["image"]), "")
                        for f in frames if fish_ids.get(strip_extension(f["row"]["image"]))), "")
        landmarked = [f for f in frames if f["points"] is not None and len(f["points"]) >= 3]
        print(f"[{number}/{len(groups)}] {video} {fish_id}  "
              f"({len(frames)} frames, {len(landmarked)} landmarked)")

        # One crop size for the whole video, so the fish is the same size in
        # every picture: the largest fish extent across frames, padded.
        crop = not options.full and landmarked
        if crop:
            extents = np.array([np.ptp(f["points"], axis=0) for f in landmarked])
            size = extents.max(axis=0) * (1 + 2 * CROP_PADDING)
            size = np.maximum(size, [200, 120])
            fallback_centre = np.mean([f["points"].mean(axis=0) for f in landmarked], axis=0)

        tiles = []
        for f in frames:
            image = f["image"].copy()
            if options.outline and f["points"] is not None and len(f["points"]) == 43:
                loop = np.round(f["points"][PERIMETER_ORDER]).astype(np.int32)
                thick = max(2, image.shape[1] // 500)
                cv2.polylines(image, [loop], True, (0, 220, 255), thick, cv2.LINE_AA)
                for p in np.round(f["points"][:7]).astype(int):
                    cv2.circle(image, tuple(p), thick * 3, (0, 0, 255), -1, cv2.LINE_AA)
            if crop:
                centre = (f["points"].mean(axis=0) if f["points"] is not None
                          else fallback_centre)
                image, _ = crop_around(image, centre, size)
            tiles.append(image)

        # Layout: wide crops two to a row, tall whole frames five to a row.
        th, tw = tiles[0].shape[:2]
        wide = tw >= th
        cell_w = WIDE_CELL if wide else TALL_CELL
        cols = min(2 if wide else 5, len(tiles))
        cell_h = int(round(th * cell_w / tw))
        tiles = [cv2.resize(t, (cell_w, cell_h), interpolation=cv2.INTER_AREA) for t in tiles]

        margin, label_h, header_h = 16, 32, 70
        rows_n = math.ceil(len(tiles) / cols)
        sheet = np.full((header_h + rows_n * (label_h + cell_h + margin) + margin,
                         cols * cell_w + (cols + 1) * margin, 3), 22, np.uint8)
        put(sheet, f"{video}   {fish_id}".strip(), (margin, 32), 0.85, thickness=2)
        note = "cropped around the fish, same scale in every frame" if crop else "whole frames"
        put(sheet, f"{len(tiles)} frames in time order - {note}", (margin, 58), 0.5,
            (170, 170, 170))

        for i, (tile, f) in enumerate(zip(tiles, frames), start=1):
            r, c = divmod(i - 1, cols)
            x = margin + c * (cell_w + margin)
            y = header_h + r * (label_h + cell_h + margin)
            put(sheet, str(i), (x, y + 24), 0.8, (255, 255, 255), 2)
            seconds = float(f["row"]["seconds"])
            extra = "" if f["points"] is not None else "   (not landmarked)"
            put(sheet, f"{seconds:.0f} s{extra}", (x + 40, y + 23), 0.52, (190, 190, 190))
            sheet[y + label_h:y + label_h + cell_h, x:x + cell_w] = tile

        name = re.sub(r"[^A-Za-z0-9_+-]", "_",
                      f"{strip_extension(video)}_{fish_id}" if fish_id else strip_extension(video))
        cv2.imwrite(os.path.join(options.out, name + ext), sheet, params)

    print(f"\nSheets saved in {options.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())