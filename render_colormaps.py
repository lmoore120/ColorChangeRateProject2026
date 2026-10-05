r"""
render_colormaps.py

Turns Colormesh's colour data back into pictures of each fish, so the
frames can be compared by eye.

WHAT IT MAKES
  results\colormap_frames\<video>\frame_01_0000s.png ...
      One image per frame: the fish in the standard (consensus) shape,
      coloured with the calibrated colour measured at every sampling
      point. These are exactly the colours the analysis uses - not the
      raw video frame.

  results\colormap_sheets\<video>_<fish_id>.png
      All of one fish's frames on a single sheet, labelled 1-10 in time
      order with each frame's time and mean L*. Under each colour image
      is a CHANGE MAP: how much lighter (orange) or darker (blue) every
      part of the body is than in frame 1, all on one shared scale.
      Changes of a few L* units are nearly invisible in true colour but
      obvious in the change map. Turn it off with --colour-only.

WHY THE FISH ALL LINE UP
  Colormesh warps every frame onto the same standard fish shape before
  sampling, so sampling point n sits on the same part of the body in every
  frame. That's what makes side-by-side comparison meaningful - you are
  looking at the same patch of fish in each picture.

HOW TO RUN (paths already point at APHP white)
    python render_colormaps.py
    python render_colormaps.py --jpg            # JPEG instead of PNG
    python render_colormaps.py --colour-only    # sheets without change maps
    python render_colormaps.py --only GH010994.MP4
"""

import argparse
import csv
import math
import os
import re
import sys

import cv2
import numpy as np
from scipy.spatial import Delaunay

# =====================  SETTINGS  =====================

BASE = r"C:\Users\lamoo\ColorChangeRates2026\files\APHP\white"
COLORMAP_FILE = os.path.join(BASE, "white_colormaps", "colormap_calib.csv")
MANIFEST_FILE = os.path.join(BASE, "white_frames", "frames_manifest.csv")
LANDMARKS_CSV = os.path.join(BASE, "white_landmarks", "landmarks.csv")
OUTPUT_DIR = os.path.join(BASE, "results")

FRAME_WIDTH = 1200       # width of each individual frame image, pixels
SHEET_CELL_WIDTH = 620   # width of each fish on the comparison sheet
SHEET_COLUMNS = 2        # frames per row on the sheet (2 x 5 for 10 frames)
BACKGROUND = (38, 38, 38)   # dark grey, so a pale fish still has an edge

# The 43 landmarks in the order they run AROUND the fish (perimeter.map in
# the R scripts), used if the perimeter columns turn out to be stored in
# landmark-number order rather than outline order.
PERIMETER_MAP = [1] + list(range(8, 17)) + [2, 17, 3, 18, 19, 4] + \
    list(range(20, 29)) + [5, 29, 6, 30, 31, 32, 7] + list(range(33, 44))

# ========================================================


def strip_extension(name):
    return re.sub(r"\.[A-Za-z0-9]+$", "", str(name))


def lightness(rgb01):
    """L* from sRGB 0-1, identical to rgb2lab() in CIE_conversion.R."""
    rgb01 = np.clip(np.asarray(rgb01, dtype=np.float64), 0.0, 1.0)
    linear = np.where(rgb01 <= 0.04045, rgb01 / 12.92,
                      ((rgb01 + 0.055) / 1.055) ** 2.4)
    y = (linear[..., 0] * 0.2126729 + linear[..., 1] * 0.7151522
         + linear[..., 2] * 0.0721750)
    fy = np.where(y > 0.008856, np.cbrt(y), 7.787 * y + 16.0 / 116.0)
    return 116.0 * fy - 16.0


# ------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------

def read_colormap(path):
    """Returns (point_names, perimeter_names, rows), where each row is
    {'image', 'xy' (n,2), 'rgb' (n,3)} in point_names order.

    Columns are matched BY NAME - r_<point> with its own g_, b_, x_, y_ -
    never by position, for the same reason CIE_conversion.R now does it:
    Colormesh doesn't always write the columns in the same order."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: i for i, name in enumerate(header)}
        points = [h[2:] for h in header if h.startswith("r_")]
        points = [p for p in points
                  if all(f"{c}_{p}" in index for c in "gbxy")]
        if not points:
            sys.exit(f"No complete r_/g_/b_/x_/y_ column sets found in {path}")
        cols = {c: [index[f"{c}_{p}"] for p in points] for c in "rgbxy"}

        rows = []
        for record in reader:
            if not record:
                continue
            def grab(c):
                return np.array([float(record[i]) if record[i] not in ("", "NA")
                                 else np.nan for i in cols[c]])
            rows.append({
                "image": record[0],
                "xy": np.column_stack([grab("x"), grab("y")]),
                "rgb": np.column_stack([grab("r"), grab("g"), grab("b")]),
            })

    # Perimeter points in their numbered order (perimeter1, perimeter2, ...)
    perimeter = sorted([p for p in points if p.startswith("perimeter")],
                       key=lambda p: int(re.sub(r"\D", "", p) or 0))
    return points, perimeter, rows


def read_csv_by_image(path, wanted):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = strip_extension(row.get("image", ""))
            out[key] = {k: row.get(k, "") for k in wanted}
    return out


# ------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------

def count_crossings(loop):
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    n, hits = len(loop), 0
    for i in range(n):
        a1, a2 = loop[i], loop[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            b1, b2 = loop[j], loop[(j + 1) % n]
            if ((side(b1, b2, a1) > 0) != (side(b1, b2, a2) > 0)
                    and (side(a1, a2, b1) > 0) != (side(a1, a2, b2) > 0)):
                hits += 1
    return hits


def outline_loop(perimeter_xy):
    """Order the perimeter points so they trace the outline without
    crossing itself. Colormesh may store them in outline order or in
    landmark-number order; whichever crosses itself less is the right one."""
    if len(perimeter_xy) < 3:
        return None
    candidates = [("as stored", perimeter_xy)]
    if len(perimeter_xy) == 43:
        candidates.append(("landmark order",
                           perimeter_xy[[k - 1 for k in PERIMETER_MAP]]))
    scored = sorted(((count_crossings(loop), name, loop) for name, loop in candidates),
                    key=lambda t: t[0])
    return scored[0]


class Renderer:
    """Precomputes how every output pixel blends the sampling points, so
    each frame is a fast weighted sum. Every frame shares the same point
    positions (all are warped to one shape), so this is built once and
    reused."""

    def __init__(self, xy, perimeter_idx, width):
        good = np.all(np.isfinite(xy), axis=1)
        self.good = good
        pts = xy[good]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pad_px = int(round(0.04 * width))
        self.scale = (width - 2 * pad_px) / (x1 - x0) if x1 > x0 else 1.0
        self.offset = np.array([x0, y0])
        self.width = width
        self.height = int(round((y1 - y0) * self.scale)) + 2 * pad_px
        to_px = lambda p: (p - self.offset) * self.scale + pad_px
        px = to_px(pts)

        tri = Delaunay(px)
        gy, gx = np.mgrid[0:self.height, 0:width]
        grid = np.column_stack([gx.ravel() + 0.5, gy.ravel() + 0.5])
        simplex = tri.find_simplex(grid)
        inside = simplex >= 0

        # Cut to the fish's own outline - the triangulation alone would
        # also fill the concave gaps around the fins and tail.
        mask = np.zeros((self.height, width), np.uint8)
        self.outline_note = "convex hull (no perimeter points found)"
        if perimeter_idx is not None and len(perimeter_idx) >= 3:
            crossings, how, loop = outline_loop(xy[perimeter_idx])
            if loop is not None and np.all(np.isfinite(loop)):
                cv2.fillPoly(mask, [np.round(to_px(loop)).astype(np.int32)], 255)
                self.outline_note = (f"perimeter points ({how}, "
                                     f"{crossings} self-crossing(s))")
        if not mask.any():
            hull = cv2.convexHull(np.round(px).astype(np.int32))
            cv2.fillPoly(mask, [hull], 255)
        inside &= mask.ravel() > 0

        T = tri.transform[simplex[inside]]
        b = np.einsum("ijk,ik->ij", T[:, :2], grid[inside] - T[:, 2])
        self.weights = np.column_stack([b, 1.0 - b.sum(axis=1)])
        self.vertices = tri.simplices[simplex[inside]]
        self.inside = inside

    def blend(self, values):
        """values: one row per GOOD point. Returns (H, W, k) with NaN outside."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        out = np.full((self.height * self.width, values.shape[1]), np.nan)
        out[self.inside] = np.einsum("ij,ijk->ik", self.weights, values[self.vertices])
        return out.reshape(self.height, self.width, values.shape[1])

    def colour_image(self, rgb01):
        rgb = np.clip(self.blend(rgb01[self.good]), 0, 1)
        img = np.empty((self.height, self.width, 3), np.uint8)
        img[:] = BACKGROUND
        valid = ~np.isnan(rgb[..., 0])
        img[valid] = (rgb[valid][:, ::-1] * 255).round().astype(np.uint8)  # RGB->BGR
        return img

    def change_image(self, delta, limit):
        d = self.blend(delta[self.good])[..., 0]
        img = np.empty((self.height, self.width, 3), np.uint8)
        img[:] = BACKGROUND
        valid = ~np.isnan(d)
        img[valid] = diverging(d[valid], limit)
        return img


def diverging(values, limit):
    """Blue = darker than frame 1, white = no change, orange = lighter.
    Returns BGR uint8."""
    t = np.clip(values / max(limit, 1e-9), -1, 1)[:, None]
    white = np.array([247, 247, 247], float)
    darker = np.array([172, 102, 33], float)     # BGR blue
    lighter = np.array([1, 97, 230], float)      # BGR orange
    out = np.where(t < 0, white + (-t) * (darker - white), white + t * (lighter - white))
    return out.round().astype(np.uint8)


# ------------------------------------------------------------------
# Sheet
# ------------------------------------------------------------------

def put(img, text, org, scale, colour=(235, 235, 235), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thickness, cv2.LINE_AA)


def build_sheet(title, subtitle, frames, change_limit, colour_only):
    """frames: list of dicts with 'number', 'label', 'colour', 'change'."""
    cell_w = frames[0]["colour"].shape[1]
    img_h = frames[0]["colour"].shape[0]
    label_h = 34
    gap = 6
    cell_h = label_h + img_h + (0 if colour_only else gap + img_h) + 14
    cols = min(SHEET_COLUMNS, len(frames))
    rows = math.ceil(len(frames) / cols)
    margin = 18
    header_h = 78
    footer_h = 0 if colour_only else 74
    width = cols * cell_w + (cols + 1) * margin
    height = header_h + rows * cell_h + footer_h + margin
    sheet = np.full((height, width, 3), 22, np.uint8)

    put(sheet, title, (margin, 32), 0.85, thickness=2)
    put(sheet, subtitle, (margin, 60), 0.5, (170, 170, 170))

    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        x = margin + c * (cell_w + margin)
        y = header_h + r * cell_h
        put(sheet, str(f["number"]), (x, y + 24), 0.8, (255, 255, 255), 2)
        put(sheet, f["label"], (x + 40, y + 23), 0.52, (190, 190, 190))
        sheet[y + label_h:y + label_h + img_h, x:x + cell_w] = f["colour"]
        if not colour_only:
            y2 = y + label_h + img_h + gap
            sheet[y2:y2 + img_h, x:x + cell_w] = f["change"]
            if i == 0:
                # Frame 1 is compared with itself, so its change map is
                # blank by definition - say so, rather than look broken.
                text = "reference frame (no change by definition)"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                put(sheet, text, (x + (cell_w - tw) // 2, y2 + img_h // 2 + th // 2),
                    0.5, (90, 90, 90))

    if not colour_only:
        # Colour bar for the change maps.
        bar_w = min(520, width - 2 * margin)
        bx = (width - bar_w) // 2
        by = height - footer_h + 14
        ramp = diverging(np.linspace(-change_limit, change_limit, bar_w), change_limit)
        sheet[by:by + 16, bx:bx + bar_w] = np.repeat(ramp[None, :, :], 16, axis=0)
        put(sheet, f"-{change_limit:.1f}", (bx - 50, by + 13), 0.45)
        put(sheet, f"+{change_limit:.1f}", (bx + bar_w + 8, by + 13), 0.45)
        put(sheet, "darker than frame 1", (bx, by + 38), 0.45, (210, 170, 120))
        put(sheet, "change in L* (lower image of each pair)",
            (bx + bar_w // 2 - 150, by + 56), 0.42, (170, 170, 170))
        text = "lighter than frame 1"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        put(sheet, text, (bx + bar_w - tw, by + 38), 0.45, (90, 160, 240))
    return sheet


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--colormap", default=COLORMAP_FILE)
    parser.add_argument("--manifest", default=MANIFEST_FILE)
    parser.add_argument("--landmarks-csv", default=LANDMARKS_CSV)
    parser.add_argument("--out", default=OUTPUT_DIR)
    parser.add_argument("--jpg", action="store_true", help="save JPEG instead of PNG")
    parser.add_argument("--colour-only", action="store_true",
                        help="sheets show only the true-colour images, no change maps")
    parser.add_argument("--only", default=None, help="one video, e.g. GH010994.MP4")
    parser.add_argument("--flip-vertical", action="store_true",
                        help="turn the fish upside down - use if they come out "
                             "belly-up")
    options = parser.parse_args()

    if not os.path.exists(options.colormap):
        sys.exit(f"Not found: {options.colormap}\nRun Colormesh_extraction.R first.")

    print(f"Reading {options.colormap} ...")
    points, perimeter, rows = read_colormap(options.colormap)
    print(f"  {len(rows)} frames, {len(points)} sampling points "
          f"({len(perimeter)} on the outline)")

    manifest = read_csv_by_image(options.manifest, ["video", "seconds"])
    landmarks = read_csv_by_image(options.landmarks_csv, ["fish_id"])
    if not manifest:
        print("  WARNING: no manifest - frames will be grouped and ordered by file name")

    # Group frames by video, in time order.
    groups = {}
    for row in rows:
        key = strip_extension(row["image"])
        info = manifest.get(key, {})
        video = info.get("video") or re.sub(r"_f\d+_t\d+$", "", key)
        try:
            seconds = float(info.get("seconds", ""))
        except ValueError:
            match = re.search(r"_t(\d+)$", key)
            seconds = float(match.group(1)) if match else float("nan")
        row.update(video=video, seconds=seconds,
                   fish_id=landmarks.get(key, {}).get("fish_id", ""))
        groups.setdefault(video, []).append(row)

    if options.only:
        groups = {v: f for v, f in groups.items() if v == options.only}
        if not groups:
            sys.exit(f"No frames for {options.only}")

    if options.flip_vertical:
        for row in rows:
            row["xy"][:, 1] = -row["xy"][:, 1]

    perimeter_idx = np.array([points.index(p) for p in perimeter]) if perimeter else None
    ext = ".jpg" if options.jpg else ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if options.jpg else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    frame_dir = os.path.join(options.out, "colormap_frames")
    sheet_dir = os.path.join(options.out, "colormap_sheets")
    os.makedirs(sheet_dir, exist_ok=True)

    renderers = {}
    for number, (video, frames) in enumerate(sorted(groups.items()), start=1):
        frames.sort(key=lambda r: (math.isnan(r["seconds"]), r["seconds"]))
        stem = strip_extension(video)
        fish_id = next((f["fish_id"] for f in frames if f["fish_id"]), "")
        print(f"[{number}/{len(groups)}] {video} {fish_id}  ({len(frames)} frames)")

        # One renderer per distinct set of point positions - normally just
        # one for the whole run, since every frame shares the same shape.
        key = np.round(frames[0]["xy"], 2).tobytes()
        if key not in renderers:
            renderers[key] = Renderer(frames[0]["xy"], perimeter_idx, FRAME_WIDTH)
            print(f"   outline from {renderers[key].outline_note}")
        renderer = renderers[key]

        first_L = lightness(frames[0]["rgb"])
        deltas = [lightness(f["rgb"]) - first_L for f in frames]
        finite = np.concatenate([d[np.isfinite(d)] for d in deltas]) if deltas else np.array([0.0])
        limit = max(2.0, float(np.percentile(np.abs(finite), 98))) if finite.size else 2.0

        os.makedirs(os.path.join(frame_dir, stem), exist_ok=True)
        sheet_frames = []
        for i, (f, delta) in enumerate(zip(frames, deltas), start=1):
            colour = renderer.colour_image(f["rgb"])
            secs = f["seconds"]
            tag = f"{int(round(secs)):04d}s" if not math.isnan(secs) else "time_unknown"
            cv2.imwrite(os.path.join(frame_dir, stem, f"frame_{i:02d}_{tag}{ext}"),
                        colour, params)

            small_h = int(round(colour.shape[0] * SHEET_CELL_WIDTH / colour.shape[1]))
            mean_L = float(np.nanmean(lightness(f["rgb"])))
            time_text = f"{secs:.0f} s" if not math.isnan(secs) else "time ?"
            sheet_frames.append({
                "number": i,
                "label": f"{time_text}    mean L* {mean_L:.1f}",
                "colour": cv2.resize(colour, (SHEET_CELL_WIDTH, small_h),
                                     interpolation=cv2.INTER_AREA),
                "change": cv2.resize(renderer.change_image(delta, limit),
                                     (SHEET_CELL_WIDTH, small_h),
                                     interpolation=cv2.INTER_AREA),
            })

        title = f"{video}   {fish_id}".strip()
        subtitle = (f"{len(frames)} frames in time order. Calibrated colour at each "
                    f"sampling point, on the standard fish shape.")
        sheet = build_sheet(title, subtitle, sheet_frames, limit, options.colour_only)
        name = f"{stem}_{fish_id}" if fish_id else stem
        name = re.sub(r"[^A-Za-z0-9_+-]", "_", name)
        cv2.imwrite(os.path.join(sheet_dir, name + ext), sheet, params)

    print(f"\nPer-frame images: {frame_dir}")
    print(f"Comparison sheets: {sheet_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())