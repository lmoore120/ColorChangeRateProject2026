#!/usr/bin/env python3
"""
landmark_tracker.py

Keeps landmark_tracker.csv up to date. A CSV is just a text file and cannot update
itself, so this script does it.

HOW IT WORKS
  Your landmark files are named like  IMG0002_IMG0002_f10_t0113_points.txt  and live in
  folders like  ...\\files\\APHP\\white\\white_landmarks\\ , so the FISH ID is not in the
  file name. The script gets three things from each file:
      population  <- the folder name (APHP, APKO, ECHP, ECKO)
      background  <- the folder name (white / black)
      video       <- the IMG number in the file name (IMG0002)
  It lists every video it finds in  video_map.csv . You type the fish ID next to each
  video ONCE in that file. From then on, the script checks the fish off in
  landmark_tracker.csv automatically. A fish is "both_done" when its white and black
  videos are both checked.

USAGE (from a terminal opened in this folder):
    python landmark_tracker.py                  scan once and update the CSVs
    python landmark_tracker.py --watch          keep scanning every 30 s until Ctrl+C
    python landmark_tracker.py --mark APHP_o3y6 white      check one off by hand
    python landmark_tracker.py --unmark APHP_o3y6 white    undo a check
    python landmark_tracker.py --dir "C:\\path\\to\\files"   use a different folder this run

The script only ever ADDS checks (or removes one when you use --unmark), so a fish you
checked by hand, or typed "Yes" for in the CSV, will not be unchecked by a scan.
"""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# =====================  SETTINGS (edit these)  =====================

# Folder that holds your landmarking OUTPUT files. Subfolders are searched too.
LANDMARK_DIR = r"C:\Users\lamoo\ColorChangeRates2026\files"

# Only count files whose name ends with one of these. (None = count any file.)
FILE_EXTENSIONS = ("_points.txt",)

# How the script recognises the video in a file name: the first match of this pattern.
# IMG0002_IMG0002_f10_t0113_points.txt  ->  IMG0002
VIDEO_REGEX = r"img\d+"

# A video counts as landmarked once it has at least this many landmark files.
# SET THIS to the number of frames you landmark per video. With 1, a fish is checked
# as soon as its first frame is done.
FRAMES_REQUIRED = 1

# A path counts as the WHITE / BLACK video if it contains one of these words
# (any capitalisation).
WHITE_TAGS = ("white",)
BLACK_TAGS = ("black",)

# Ignore empty/tiny files (bytes).
MIN_FILE_SIZE = 1

# How often --watch re-scans, in seconds.
WATCH_SECONDS = 30

# ===================================================================

HERE = Path(__file__).resolve().parent
IDS_FILE = HERE / "fish_ids.txt"
CSV_FILE = HERE / "landmark_tracker.csv"
MAP_FILE = HERE / "video_map.csv"
COLUMNS = ["fish_id", "population", "white_done", "black_done", "both_done",
           "white_source", "black_source", "last_updated"]
MAP_COLUMNS = ["population", "background", "video", "fish_id", "frames_found", "last_seen"]
YES_WORDS = {"yes", "y", "true", "1", "x"}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_yes(value):
    return str(value).strip().lower() in YES_WORDS


def load_ids():
    """Read fish_ids.txt: one ID per line, whitespace removed, duplicates dropped."""
    if not IDS_FILE.exists():
        sys.exit(f"Can't find {IDS_FILE}. Keep fish_ids.txt in the same folder as this script.")
    ids = []
    for line in IDS_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(re.sub(r"\s+", "", line))
    return list(dict.fromkeys(ids))


def blank_row(fish_id):
    return {"fish_id": fish_id, "population": fish_id.split("_")[0],
            "white_done": "No", "black_done": "No", "both_done": "No",
            "white_source": "", "black_source": "", "last_updated": ""}


def refresh_both(rows):
    for r in rows.values():
        r["both_done"] = "Yes" if (r["white_done"] == "Yes" and r["black_done"] == "Yes") else "No"


def load_table():
    """Load the tracker CSV (creating it on first run) and add any new IDs from fish_ids.txt."""
    ids = load_ids()
    rows = {}
    if CSV_FILE.exists():
        with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                fid = re.sub(r"\s+", "", r.get("fish_id", ""))
                if fid:
                    row = blank_row(fid)
                    row.update({k: (r.get(k) or "") for k in COLUMNS if k in r})
                    row["fish_id"] = fid
                    for k in ("white_done", "black_done"):
                        row[k] = "Yes" if is_yes(row[k]) else "No"
                    rows[fid] = row
    added = 0
    for fid in ids:
        if fid not in rows:
            rows[fid] = blank_row(fid)
            added += 1
    refresh_both(rows)
    return rows, added


def write_csv_atomic(path, columns, row_dicts, label):
    """Write atomically. Returns False (with a message) if the file is locked, e.g. open in Excel."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in row_dicts:
            w.writerow(r)
    try:
        os.replace(tmp, path)
        return True
    except PermissionError:
        print(f"  Could not save {label}: is it open in another program (Excel)? "
              "Close it and it will save on the next scan.")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def save_table(rows):
    return write_csv_atomic(CSV_FILE, COLUMNS, rows.values(), "landmark_tracker.csv")


# ---------------------------  video_map.csv  ---------------------------

def map_key(pop, bg, video):
    return (pop.strip().upper(), bg.strip().lower(), video.strip().upper())


def load_map():
    vmap = {}
    if MAP_FILE.exists():
        with open(MAP_FILE, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                key = map_key(r.get("population") or "", r.get("background") or "", r.get("video") or "")
                if all(key):
                    vmap[key] = {"fish_id": re.sub(r"\s+", "", r.get("fish_id") or ""),
                                 "frames_found": (r.get("frames_found") or "0").strip(),
                                 "last_seen": (r.get("last_seen") or "").strip()}
    return vmap


def save_map(vmap, before):
    snapshot = {k: dict(v) for k, v in vmap.items()}
    if snapshot == before and MAP_FILE.exists():
        return  # nothing changed, don't touch the file
    rows = [{"population": k[0], "background": k[1], "video": k[2], **v} for k, v in sorted(vmap.items())]
    write_csv_atomic(MAP_FILE, MAP_COLUMNS, rows, "video_map.csv")


# ------------------------------  scanning  ------------------------------

def scan_folder(directory, rows):
    """
    Returns (id_matches, videos, unknown_ids, ambiguous, unrecognised)
      id_matches[(fish_id, bg)] = rel path      (fish ID found in the path itself)
      videos[(POP, bg, VIDEO)]  = [rel paths]   (population/background/video found instead)
    """
    known_lower = {fid.lower(): fid for fid in rows}
    prefixes = sorted({fid.split("_")[0].lower() for fid in rows})
    alt = "|".join(map(re.escape, prefixes))
    id_re = re.compile(r"(?:%s)_[a-z0-9]{4}" % alt)
    pop_re = re.compile(r"(?<![a-z])(%s)(?![a-z])" % alt)
    video_re = re.compile(VIDEO_REGEX, re.I)
    exts = tuple(e.lower() for e in FILE_EXTENSIONS) if FILE_EXTENSIONS else None

    id_matches, videos, unknown, ambiguous, unrecognised = {}, {}, set(), [], []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in sorted(files):
            if name.startswith("."):
                continue
            full = os.path.join(root, name)
            if exts and not name.lower().endswith(exts):
                continue
            try:
                if os.path.getsize(full) < MIN_FILE_SIZE:
                    continue
            except OSError:
                continue
            rel = os.path.relpath(full, directory)
            low = rel.lower()
            is_w = any(t.lower() in low for t in WHITE_TAGS)
            is_b = any(t.lower() in low for t in BLACK_TAGS)
            bg = None if is_w == is_b else ("white" if is_w else "black")

            found = set(id_re.findall(low))
            if found:                                   # route 1: fish ID is in the path
                if len(found) > 1:
                    ambiguous.append((rel, "more than one fish ID in the path"))
                    continue
                fid_low = next(iter(found))
                if fid_low not in known_lower:
                    unknown.add(fid_low)
                    continue
                if bg is None:
                    ambiguous.append((rel, "can't tell white from black"))
                    continue
                id_matches.setdefault((known_lower[fid_low], bg), rel)
                continue

            pop_m = pop_re.search(os.path.dirname(low))  # route 2: population + video number
            vid_m = video_re.search(os.path.basename(low))
            if pop_m and vid_m and bg:
                videos.setdefault(map_key(pop_m.group(1), bg, vid_m.group(0)), []).append(rel)
            else:
                unrecognised.append(rel)
    return id_matches, videos, unknown, ambiguous, unrecognised


def summary(rows):
    n = len(rows)
    w = sum(r["white_done"] == "Yes" for r in rows.values())
    b = sum(r["black_done"] == "Yes" for r in rows.values())
    both = sum(r["both_done"] == "Yes" for r in rows.values())
    print(f"  Fish complete (both videos): {both}/{n}   |   white: {w}/{n}   black: {b}/{n}   "
          f"|   videos: {w + b}/{2 * n}")
    pops = {}
    for r in rows.values():
        p = pops.setdefault(r["population"], [0, 0])
        p[1] += 1
        p[0] += r["both_done"] == "Yes"
    print("  By population: " + ",  ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(pops.items())))


def mark_done(rows, fid, bg, source, newly):
    r = rows[fid]
    col = f"{bg}_done"
    if r[col] != "Yes":
        r[col] = "Yes"
        r[f"{bg}_source"] = source
        r["last_updated"] = now()
        newly.append(f"{fid} {bg}")
    elif not r[f"{bg}_source"]:
        r[f"{bg}_source"] = source


def run_scan(directory):
    rows, added = load_table()
    vmap = load_map()
    vmap_before = {k: dict(v) for k, v in vmap.items()}
    known_lower = {fid.lower(): fid for fid in rows}
    print(f"[{now()}] Scanning {directory}")
    if added:
        print(f"  Added {added} new fish ID(s) from fish_ids.txt")

    id_matches, videos, unknown, ambiguous, unrecognised = scan_folder(directory, rows)
    newly, warnings, unmapped, in_progress = [], [], [], []

    for (fid, bg), rel in id_matches.items():
        mark_done(rows, fid, bg, rel, newly)

    claimed = {}
    for key in sorted(videos):
        pop, bg, vid = key
        files = videos[key]
        n = len(files)
        entry = vmap.setdefault(key, {"fish_id": "", "frames_found": "0", "last_seen": ""})
        if str(n) != entry["frames_found"]:
            entry["frames_found"] = str(n)
            entry["last_seen"] = now()
        raw = entry["fish_id"]
        if not raw:
            unmapped.append(f"{pop}/{bg}/{vid}")
            continue
        fid = known_lower.get(raw.lower())
        if fid is None:
            warnings.append(f"{pop}/{bg}/{vid} is mapped to '{raw}', which is not in fish_ids.txt (typo?)")
            continue
        if fid.split("_")[0] != pop:
            warnings.append(f"{pop}/{bg}/{vid} is mapped to {fid}, but the folder says {pop}. Check video_map.csv")
            continue
        if (fid, bg) in claimed:
            other = claimed[(fid, bg)]
            warnings.append(f"{fid} {bg} is assigned to two videos ({other[2]} and {vid}). Check video_map.csv")
            continue
        claimed[(fid, bg)] = key
        if n < FRAMES_REQUIRED:
            in_progress.append(f"{fid} {bg} ({vid}): {n}/{FRAMES_REQUIRED} frames")
            continue
        mark_done(rows, fid, bg, f"{pop}/{bg}/{vid} ({n} frames)", newly)

    refresh_both(rows)
    print(f"  Videos found: {len(videos)}   |   with a fish ID in video_map.csv: {len(videos) - len(unmapped)}")
    print("  Newly checked: " + ", ".join(newly) if newly else "  Nothing newly checked.")
    if in_progress:
        print("  In progress: " + "; ".join(in_progress[:8]) + (" ..." if len(in_progress) > 8 else ""))
    if unmapped:
        print(f"  {len(unmapped)} video(s) still need a fish ID in video_map.csv, e.g. "
              + ", ".join(unmapped[:5]))
    for w in warnings:
        print("  WARNING: " + w)
    if unknown:
        print("  WARNING: files found for IDs that are not in fish_ids.txt (typo?): "
              + ", ".join(sorted(unknown)))
    for rel, why in ambiguous[:10]:
        print(f"  WARNING: skipped {rel}: {why}")
    if unrecognised:
        print(f"  WARNING: {len(unrecognised)} landmark file(s) had no recognisable population folder, "
              f"white/black folder, or video number, e.g. {unrecognised[0]}")
    save_map(vmap, vmap_before)
    save_table(rows)
    summary(rows)


def manual_set(fish_id, background, value):
    rows, _ = load_table()
    lookup = {k.lower(): k for k in rows}
    key = lookup.get(re.sub(r"\s+", "", fish_id).lower())
    if key is None:
        sys.exit(f"'{fish_id}' is not in fish_ids.txt.")
    r = rows[key]
    if value:
        if r[f"{background}_done"] != "Yes":
            r[f"{background}_source"] = "manual"   # keep the file name if a scan already found one
        r[f"{background}_done"] = "Yes"
    else:
        r[f"{background}_done"] = "No"
        r[f"{background}_source"] = ""
    r["last_updated"] = now()
    refresh_both(rows)
    if save_table(rows):
        print(f"{key} {background}: {'checked' if value else 'unchecked'}")
        summary(rows)


def main():
    ap = argparse.ArgumentParser(description="Update landmark_tracker.csv from your landmark files.")
    ap.add_argument("--dir", help="landmark output folder (overrides LANDMARK_DIR)")
    ap.add_argument("--watch", action="store_true", help=f"re-scan every {WATCH_SECONDS}s until Ctrl+C")
    ap.add_argument("--mark", nargs=2, metavar=("FISH_ID", "white|black"), help="check a video off by hand")
    ap.add_argument("--unmark", nargs=2, metavar=("FISH_ID", "white|black"), help="uncheck a video")
    args = ap.parse_args()

    for spec, val in ((args.mark, True), (args.unmark, False)):
        if spec:
            bg = spec[1].lower()
            if bg not in ("white", "black"):
                sys.exit("Second argument must be 'white' or 'black'.")
            manual_set(spec[0], bg, val)
            return

    directory = args.dir or LANDMARK_DIR
    if not directory:
        rows, _ = load_table()
        save_table(rows)
        sys.exit("Tracker CSV is ready, but no landmark folder is set yet.\n"
                 "Open landmark_tracker.py and set LANDMARK_DIR near the top (or use --dir).")
    if not os.path.isdir(directory):
        sys.exit(f"Folder not found: {directory}")

    if args.watch:
        print(f"Watching every {WATCH_SECONDS}s. Press Ctrl+C to stop.")
        try:
            while True:
                run_scan(directory)
                time.sleep(WATCH_SECONDS)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        run_scan(directory)


if __name__ == "__main__":
    main()