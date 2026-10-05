r"""
extract_frames.py - Step 1 of the pipeline.

Pull a chosen number of SHARP, evenly-spaced frames out of each video and
write them as stills named so that Colormesh and R can use them.

WHY THIS EXISTS
---------------
The landmarking script's own frame grabber just took every Nth frame with
no quality check, so motion-blurred frames got through - and a blurred fish
gives you both a bad outline and a smeared colour reading. This does what
auto_fish.py does instead: at each target time it looks ahead a little and
keeps the sharpest frame it finds.

It also makes the number of frames a single obvious setting, because you
will want to try several. Sampling 6, 12 and 24 frames per video and seeing
which gives the cleanest colour-change curve is a legitimate thing to check,
and re-running this with --frames is all it takes.

NAMING - THIS MATTERS FOR COLORMESH
-----------------------------------
Colormesh requires unique image names with no spaces, punctuation or
symbols (no + % -). Output files are therefore named:

    <label>_<video>_f<NN>_t<SSSS>.png

for example  ECKOy1y5_IMG0459_f03_t0087.png

  label  the tank/treatment code, taken from --label or the video name
  f      which frame in the series this is (1, 2, 3 ... in time order)
  t      how many seconds into the video it came from

Everything is also written to frames_manifest.csv, with the seconds column
you need for a rate-of-change analysis. Do NOT rely on parsing filenames in
R when the manifest already has the fields as proper columns.

HOW TO RUN
----------
    python extract_frames.py "IMG_0459.MOV" --frames 12
    python extract_frames.py "videos_folder" --frames 24 --out frames_24
    python extract_frames.py *.MOV --frames 12 --seconds 300

To compare several frame counts, put each in its own folder:
    python extract_frames.py videos --frames 6  --out frames_06
    python extract_frames.py videos --frames 12 --out frames_12
    python extract_frames.py videos --frames 24 --out frames_24
"""

import argparse
import csv
import datetime
import glob
import os
import re
import subprocess
import sys

import cv2
import numpy as np

from video_time import video_recorded_at

# Frames scoring below this on Laplacian variance are treated as blurred.
# Measured on real frames from this project, sharp ones scored 213-348, so
# 150 sits clearly below the good range without being permissive.
MIN_SHARPNESS = 150.0

# How far EITHER SIDE of each target time to look for a sharp frame.
#
# This is deliberately small. Frames must be evenly spaced for a rate-of-
# change analysis, so the search is symmetric (nearest sharp frame, not the
# next one) and tight. An earlier version searched only forwards, up to 3
# seconds, which quietly pushed blurry targets later and left the intervals
# uneven - measured gaps of 2.40 to 2.73 s where every gap should have been
# 2.571 s. Widen this only if a lot of your footage is soft, and check the
# deviation columns in the manifest afterwards.
SEARCH_WINDOW_SECONDS = 0.5

# Seconds to skip at the start. Default 0: for a background-change trial the
# clock starts when the fish goes in, and trimming the start of some videos
# but not others would put frames at different times into the same analysis.
# Raise it only when the opening of a video is unusable, and note that the
# sampling window still lasts the full --seconds, so skipping 2 s gives you
# 0:02 to 5:02 rather than a short video.
SKIP_START_SECONDS = 0.0

# How much of each video to sample across, in seconds.
USE_SECONDS = 300.0

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg")


def ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def extract_with_ffmpeg(video_path, seconds, out_path):
    """Pull one frame at an exact time using ffmpeg instead of OpenCV.

    Worth having because OpenCV's decoder handles some camera formats
    badly - HEVC/H.265 from a phone especially - and can give a softer or
    colour-shifted frame than the video really contains. ffmpeg also
    applies the rotation stored in the file's metadata, which OpenCV
    ignores, so a video shot in portrait comes out the right way up.

    PNG output is lossless either way; the difference is entirely in the
    decode. Returns True if a frame was written."""
    try:
        finished = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", f"{seconds:.3f}", "-i", video_path,
             "-frames:v", "1",
             "-pix_fmt", "rgb24",       # full colour, no subsampling
             "-compression_level", "1",  # PNG: fast but still lossless
             out_path],
            capture_output=True, timeout=120)
        return finished.returncode == 0 and os.path.exists(out_path)
    except Exception:
        return False


def sharpness(frame):
    """Variance of the Laplacian - the standard quick focus measure. A sharp
    edge produces large second derivatives, a blurred one doesn't."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def background_from_path(*paths):
    """Work out whether these are white- or black-background trials, from
    the folder names.

    Taken from the path rather than asked for as an option, because the
    folders already encode it (APHP/white/...) and a flag is one more
    thing to remember correctly 800 times. Checked from the deepest
    folder outwards, so .../white/frames beats a stray "black" higher up.
    Returns "" when neither word appears, rather than guessing."""
    for path in paths:
        if not path:
            continue
        parts = [p.lower() for p in os.path.normpath(os.path.abspath(path)).split(os.sep)]
        for part in reversed(parts):
            # Whole word only: "blackfish_data" shouldn't count, but
            # "white_frames" and "black" should.
            tokens = re.split(r"[^a-z]+", part)
            if "white" in tokens:
                return "white"
            if "black" in tokens:
                return "black"
    return ""


def sanitise(text):
    """Colormesh needs names without spaces, punctuation or symbols, since
    the first CSV column is used as the key to match colour readings back to
    the right image. Strip anything else out rather than discovering the
    problem later in R."""
    return re.sub(r"[^A-Za-z0-9]", "", str(text))


def grab_sharp_frame(cap, target_index, fps, last_index,
                     min_sharpness=MIN_SHARPNESS,
                     window_seconds=SEARCH_WINDOW_SECONDS):
    """Return the sharpest frame NEAREST target_index, searching symmetrically.

    Searching both ways matters. If you only look forwards, every blurred
    target slides later and your evenly spaced sample stops being evenly
    spaced - which then shows up as noise in a rate-of-change analysis. Here
    the window is centred on the target, and among frames that clear the
    sharpness bar the one CLOSEST IN TIME wins, so sharpness is bought with
    the least possible timing error.

    Returns (frame, index, sharpness, met_threshold)."""
    half = max(1, int(window_seconds * fps))
    first = max(0, int(target_index) - half)
    last = min(last_index, int(target_index) + half)

    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    candidates = []
    for index in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        candidates.append((index, sharpness(frame), frame.copy()))
    if not candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(int(target_index), last_index)))
        ok, frame = cap.read()
        if not ok:
            return None, int(target_index), 0.0, False
        return frame, int(target_index), sharpness(frame), False

    sharp_enough = [c for c in candidates if c[1] >= min_sharpness]
    if sharp_enough:
        # Closest in time among the acceptable ones.
        index, score, frame = min(sharp_enough, key=lambda c: abs(c[0] - target_index))
        return frame, index, score, True

    # Nothing clears the bar - take the sharpest available and flag it.
    index, score, frame = max(candidates, key=lambda c: c[1])
    return frame, index, score, False


def extract_from_video(video_path, out_folder, how_many, label=None,
                       seconds_to_use=USE_SECONDS, skip_start=SKIP_START_SECONDS,
                       min_sharpness=MIN_SHARPNESS,
                       window_seconds=SEARCH_WINDOW_SECONDS,
                       at_seconds=None, number_offset=0, use_ffmpeg=False):
    """Pull `how_many` sharp, evenly spaced frames from one video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  could not open {video_path} - skipping")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        print("  no frame count reported - skipping")
        cap.release()
        return []

    start = int(skip_start * fps)
    end = min(total - 1, start + int(seconds_to_use * fps))
    if end - start < fps:
        print("  too short after skipping the start - skipping")
        cap.release()
        return []

    # The ORIGINAL recording time, read out of the video container - not
    # the file's modified date, which is when it was copied off the camera.
    recorded_at, time_source = video_recorded_at(video_path)
    if time_source == "file_modified":
        print(f"  WARNING: no recording time in the file - using its modified "
              f"date ({recorded_at:%Y-%m-%d %H:%M}), which may be when it was copied")
    else:
        print(f"  recorded {recorded_at:%Y-%m-%d %H:%M:%S} (from {time_source})")

    background = background_from_path(out_folder, video_path)

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    label = sanitise(label or video_stem)
    video_tag = sanitise(video_stem)

    usable = end - start
    if at_seconds:
        targets = [start + int(round(float(second) * fps)) for second in at_seconds]
        how_many = len(targets)
    elif how_many == 1:
        targets = [start + usable // 2]
    else:
        targets = [start + int(i * usable / (how_many - 1)) for i in range(how_many)]

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {total / fps / 60:.1f} min at {fps:.0f} fps, {width}x{height}, "
          f"sampling {how_many} frames over {usable / fps:.0f}s")
    if width == 0 or height == 0:
        print("  WARNING: OpenCV could not read this video's size - the decode may "
              "be unreliable. Try --use-ffmpeg.")
    if background:
        print(f"  background: {background} (from the folder name)")
    else:
        print("  background: could not tell from the folder name - the column "
              "will be blank. Put these under a folder called white or black.")

    rows = []
    blurred = 0
    for number, target in enumerate(targets, start=1 + number_offset):
        frame, index, score, was_sharp = grab_sharp_frame(
            cap, target, fps, total - 1, min_sharpness, window_seconds)
        if frame is None:
            continue
        seconds = index / fps
        nominal_seconds = target / fps
        name = f"{label}_{video_tag}_f{number:02d}_t{int(round(seconds)):04d}.png"
        written = False
        if use_ffmpeg:
            # OpenCV still chooses WHICH frame (it did the sharpness
            # search); ffmpeg just re-decodes that moment properly.
            written = extract_with_ffmpeg(video_path, seconds,
                                          os.path.join(out_folder, name))
        if not written:
            # PNG compression 1 rather than the default 3: identical
            # pixels (PNG is lossless), just written faster.
            cv2.imwrite(os.path.join(out_folder, name), frame,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not was_sharp:
            blurred += 1
        rows.append({
            "image": name,
            "video": os.path.basename(video_path),
            "label": label,
            "video_recorded_at": recorded_at.strftime("%Y-%m-%d %H:%M:%S"),
            "video_recorded_date": recorded_at.strftime("%Y-%m-%d"),
            "video_recorded_time": recorded_at.strftime("%H:%M:%S"),
            "video_time_source": time_source,
            "background": background,
            # When this frame was taken, as a real clock time
            "frame_timestamp": (recorded_at + datetime.timedelta(
                seconds=index / fps)).strftime("%Y-%m-%d %H:%M:%S"),
            "frame_number": number,
            "video_frame_index": index,
            "seconds": round(seconds, 3),
            "nominal_seconds": round(nominal_seconds, 3),
            "timing_error_seconds": round(seconds - nominal_seconds, 3),
            "minutes": round(seconds / 60.0, 4),
            "sharpness": round(score, 1),
            "met_sharpness_threshold": int(was_sharp),
        })

    cap.release()

    # Report how even the spacing actually came out, since that is the whole
    # point of sampling this way.
    if len(rows) > 2:
        times = [r["seconds"] for r in rows]
        gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        nominal_gap = (times[-1] - times[0]) / (len(times) - 1)
        worst = max(abs(g - nominal_gap) for g in gaps)
        print(f"  interval {nominal_gap:.2f}s, worst deviation {worst:.3f}s "
              f"({100 * worst / nominal_gap:.1f}%)")

    if blurred:
        print(f"  {blurred} of {len(rows)} frames never reached the sharpness "
              f"threshold - kept the best available, flagged in the manifest")
    else:
        print(f"  all {len(rows)} frames sharp")
    return rows


def gather_videos(paths):
    videos = []
    for path in paths:
        if os.path.isdir(path):
            for extension in VIDEO_EXTENSIONS:
                videos += glob.glob(os.path.join(path, "*" + extension))
                videos += glob.glob(os.path.join(path, "*" + extension.upper()))
        elif os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    return sorted(set(videos))


def main():
    parser = argparse.ArgumentParser(
        description="Extract sharp, evenly spaced frames from videos for the "
                    "landmarking and Colormesh pipeline.")
    parser.add_argument("paths", nargs="+", help="video files, or a folder of them")
    parser.add_argument("--frames", type=int, default=12,
                        help="how many frames per video (default: %(default)s). "
                             "Try several values in separate --out folders to see "
                             "which resolves the colour-change curve best.")
    parser.add_argument("--out", default="frames",
                        help="where to write the stills (default: %(default)s)")
    parser.add_argument("--seconds", "--duration", type=float, default=USE_SECONDS,
                        dest="seconds",
                        help="length of the sampling window in seconds "
                             "(default: %(default)s = 5 minutes). Frames are spread "
                             "evenly across it, first frame at the start and last "
                             "frame at the end.")
    parser.add_argument("--skip-start", type=float, default=SKIP_START_SECONDS,
                        help="seconds to skip at the start (default: %(default)s). "
                             "The window still lasts the full --seconds, so "
                             "--skip-start 2 samples 0:02 to 5:02. Use only when a "
                             "video opens with unusable blur, and keep it the same "
                             "across videos you intend to compare.")
    parser.add_argument("--search-window", type=float, default=SEARCH_WINDOW_SECONDS,
                        help="how far either side of each target time to look for a "
                             "sharp frame, in seconds (default: %(default)s). Larger "
                             "finds sharper frames but spaces them less evenly.")
    parser.add_argument("--min-sharpness", type=float, default=MIN_SHARPNESS,
                        help="Laplacian-variance threshold for 'sharp enough' "
                             "(default: %(default)s)")
    parser.add_argument("--use-ffmpeg", action="store_true",
                        help="decode frames with ffmpeg instead of OpenCV. Slower, "
                             "but handles phone HEVC/H.265 properly and applies the "
                             "rotation stored in the file, so frames match what the "
                             "video actually looks like. Try this if the stills look "
                             "softer or the wrong way up.")
    parser.add_argument("--at-seconds", default=None,
                        help="extract frames at these exact times instead of "
                             "spreading them evenly, e.g. --at-seconds \"37.5,50\". "
                             "Use this to replace a frame you had to skip: pick a "
                             "moment a few seconds either side where the fish is "
                             "properly side-on, and the manifest records its true "
                             "time so the curve fit still uses the right x value.")
    parser.add_argument("--label", default=None,
                        help="treatment/tank code to prefix filenames with. "
                             "Defaults to the video's own name.")
    options = parser.parse_args()

    at_seconds = None
    if options.at_seconds:
        at_seconds = [float(v) for v in options.at_seconds.replace(" ", "").split(",")]
        print(f"Extracting only at {at_seconds} seconds\n")

    if options.use_ffmpeg and not ffmpeg_available():
        print("--use-ffmpeg was asked for but ffmpeg isn't on the PATH. "
              "Falling back to OpenCV.\n")
        options.use_ffmpeg = False

    videos = gather_videos(options.paths)
    if not videos:
        print("No videos found.")
        return

    os.makedirs(options.out, exist_ok=True)
    all_rows = []
    for n, video in enumerate(videos, start=1):
        print(f"\n[{n}/{len(videos)}] {os.path.basename(video)}")
        all_rows += extract_from_video(video, options.out, options.frames,
                                       label=options.label,
                                       seconds_to_use=options.seconds,
                                       skip_start=options.skip_start,
                                       min_sharpness=options.min_sharpness,
                                       window_seconds=options.search_window,
                                       at_seconds=at_seconds,
                                       use_ffmpeg=options.use_ffmpeg)

    if not all_rows:
        print("\nNothing extracted.")
        return

    manifest = os.path.join(options.out, "frames_manifest.csv")
    # When pulling replacement frames, add to the existing manifest rather
    # than overwriting it - otherwise the run that fetched one replacement
    # frame would wipe the record of the other 24.
    existing = []
    if at_seconds and os.path.exists(manifest):
        with open(manifest, newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        replaced = {r["image"] for r in all_rows}
        existing = [r for r in existing if r["image"] not in replaced]
        print(f"Adding to the existing manifest ({len(existing)} rows kept)")
    combined = existing + all_rows
    combined.sort(key=lambda r: (str(r.get("video", "")), float(r.get("seconds", 0))))
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(combined)
    all_rows = combined

    # Colormesh's specimen-factors file: first column MUST be the unique
    # image name. Written here so it's ready rather than assembled by hand.
    factors = os.path.join(options.out, "specimen_factors.csv")
    with open(factors, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "video", "frame_number", "seconds",
                         "video_recorded_at", "background"])
        for row in all_rows:
            writer.writerow([row["image"], row["label"], row["video"],
                             row["frame_number"], row["seconds"],
                             row["video_recorded_at"], row.get("background", "")])

    soft = sum(1 for r in all_rows if not r["met_sharpness_threshold"])
    print(f"\n{len(all_rows)} frames from {len(videos)} video(s) -> {options.out}")
    print(f"manifest: {manifest}")
    print(f"Colormesh specimen factors: {factors}")
    if soft:
        print(f"{soft} frame(s) below the sharpness threshold - check "
              f"met_sharpness_threshold in the manifest before analysing")
    names = [r["image"] for r in all_rows]
    if len(set(names)) != len(names):
        print("WARNING: duplicate image names - Colormesh needs them unique.")


if __name__ == "__main__":
    main()