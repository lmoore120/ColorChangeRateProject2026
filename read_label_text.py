r"""
read_label_text.py
-------------------
A simple script that reads text off a label in a video. Fully automatic -
no clicking or dragging boxes.

Every real label starts with one of a few known codes (see KNOWN_PREFIXES
below) - the script uses that to tell real matches apart from random
misreads of background texture/noise, instead of just guessing at
whatever text looks "most confident."

WHAT IT DOES, STEP BY STEP:
  1. Opens your video file
  2. Grabs a handful of still frames from it (like taking screenshots)
  3. Picks the clearest (least blurry) ones
  4. PASS 1 - a quick, rough look over each whole frame to guess at a
     few spots text MIGHT be sitting (this is what makes it automatic -
     no need to tell it where to look)
  5. PASS 2 - zooms in on each guessed spot (and the full frame, as a
     backup) and carefully reads every word it can find
  6. Keeps ONLY words that actually match a known label prefix -
     everything else (texture noise, stitching, random marks) gets
     thrown out, even if Tesseract seemed "confident" about it
  7. Saves a cropped image ONLY for a frame where a real match was
     found, so the debug folder doesn't fill up with junk

HOW TO RUN IT:
  python read_label_text.py "path\to\your\video.mp4"

After running, check the "debug_frames" folder - it has a close-up image
for every frame where a real label was found, so you can double check it
against what the program printed.
"""

import os
import re
import sys
import cv2
import pytesseract
from pytesseract import Output

# Tell Python exactly where Tesseract is installed on this computer,
# so we don't have to mess with Windows PATH settings.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DEBUG_FOLDER = "debug_frames"

# Every real label starts with one of these. Anything that doesn't match
# one of these (even closely) is treated as noise, not a real result.
KNOWN_PREFIXES = ["ECKO", "ECHP", "APKO", "APHP"]


def get_sharp_frames(video_path, how_many=5):
    """Pull a handful of clear (non-blurry) frames from the video.

    Reads straight through the video from start to end (instead of jumping
    to specific timestamps) - jumping around ("seeking") is often very slow
    and unreliable depending on how the video file is encoded, so a plain
    sequential read is much faster and more dependable.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * 0.5))  # look at one frame every half-second

    candidates = []
    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break  # reached the end of the video

        if frame_index % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            candidates.append((sharpness, frame))
            if total_frames > 0:
                percent = int(100 * frame_index / total_frames)
                print(f"  scanning... {percent}%", end="\r")

        frame_index += 1

    print()  # move to a new line after the progress indicator
    cap.release()

    # Keep only the sharpest few frames - no point reading blurry ones
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [frame for _score, frame in candidates[:how_many]]


def find_candidate_regions(frame, top_n=3, pad=15):
    """PASS 1: a rough, whole-frame OCR pass used to guess at a few
    places text MIGHT be sitting in the frame - this is what replaces
    manually dragging a crop box. Returns a list of up to `top_n`
    (x0, y0, x1, y1) boxes in original frame coordinates, best guess
    first. Returns an empty list if nothing looked promising at all.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 1.5
    coarse = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    data = pytesseract.image_to_data(coarse, config="--psm 11", output_type=Output.DICT)

    words = []
    for i, conf in enumerate(data["conf"]):
        conf = float(conf)
        text = data["text"][i].strip()
        if conf <= 0 or not text:
            continue
        words.append({
            "text": text, "conf": conf,
            "left": data["left"][i], "top": data["top"][i],
            "width": data["width"][i], "height": data["height"][i],
        })

    rows = _group_into_rows(words)
    if not rows:
        return []

    def score(row_words):
        alnum_chars = sum(c.isalnum() for w in row_words for c in w["text"])
        avg_conf = sum(w["conf"] for w in row_words) / len(row_words)
        return alnum_chars * avg_conf

    ranked = sorted(rows, key=score, reverse=True)

    regions = []
    for words in ranked[:top_n]:
        if score(words) < 50:  # skip obvious single-character noise blips
            continue
        xs = [wd["left"] for wd in words] + [wd["left"] + wd["width"] for wd in words]
        ys = [wd["top"] for wd in words] + [wd["top"] + wd["height"] for wd in words]
        x0, x1 = min(xs) / scale, max(xs) / scale
        y0, y1 = min(ys) / scale, max(ys) / scale
        x0, y0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        x1, y1 = min(w, int(x1 + pad)), min(h, int(y1 + pad))
        regions.append((x0, y0, x1, y1))

    return regions


def _clean(word):
    """Strip everything except letters/digits and uppercase it, so minor
    OCR punctuation glitches (an extra '.', a stray space) don't get in
    the way of matching."""
    return re.sub(r"[^A-Za-z0-9]", "", word).upper()


def _prefix_match_quality(word):
    """Check whether `word` starts with one of KNOWN_PREFIXES, allowing
    at most one character to be wrong (OCR commonly confuses things like
    O/0 or I/1). Returns a 0-1 match quality (1.0 = exact) if it matches,
    or None if it doesn't match any known prefix well enough."""
    cleaned = _clean(word)
    best = None
    for prefix in KNOWN_PREFIXES:
        if len(cleaned) < len(prefix):
            continue
        head = cleaned[:len(prefix)]
        mismatches = sum(a != b for a, b in zip(head, prefix))
        if mismatches <= 1:
            quality = 1.0 - (mismatches / len(prefix))
            if best is None or quality > best:
                best = quality
    return best


def _group_into_rows(words, y_tolerance_ratio=0.6):
    """Group word boxes into rows by actual vertical position, rather
    than trusting Tesseract's own block/line numbering - which can
    (surprisingly) assign different block numbers to letters sitting
    right next to each other on the same visual line, especially with
    sparse/scattered text. Returns a list of rows, each row a list of
    word dicts sorted left-to-right."""
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: w["top"])
    rows = [[words_sorted[0]]]
    for w in words_sorted[1:]:
        current_row = rows[-1]
        avg_height = sum(x["height"] for x in current_row) / len(current_row)
        if abs(w["top"] - current_row[-1]["top"]) <= max(avg_height, 1) * y_tolerance_ratio:
            current_row.append(w)
        else:
            rows.append([w])
    for row in rows:
        row.sort(key=lambda w: w["left"])
    return rows


def find_label_in_image(image, psm):
    """Run OCR on one image and return the best KNOWN_PREFIXES match found
    anywhere in it, or None if nothing matched. This is the key filter
    that throws out background-texture noise: random misreads
    essentially never happen to start with one of our specific known
    codes, so requiring a match is a very strong noise filter - much
    stronger than just trusting Tesseract's own confidence score.

    Checks both individual words AND short runs of consecutive words on
    the same row glued together - stylized fonts / embroidered tags
    sometimes get read as separate letters or fragments (e.g. "ECKO"
    coming back as "EC" + "KO", or even "E"+"C"+"K"+"O") rather than one
    clean word, so only checking single words would miss those.
    """
    data = pytesseract.image_to_data(image, config=f"--psm {psm}", output_type=Output.DICT)

    words = []
    for i, conf in enumerate(data["conf"]):
        conf = float(conf)
        text = data["text"][i].strip()
        if not text:
            continue
        words.append({
            "text": text, "conf": conf,
            "left": data["left"][i], "top": data["top"][i],
            "width": data["width"][i], "height": data["height"][i],
        })

    best_word, best_conf, best_quality = None, -1.0, -1.0

    def consider(candidate_text, candidate_conf):
        nonlocal best_word, best_conf, best_quality
        quality = _prefix_match_quality(candidate_text)
        if quality is None:
            return
        if (quality, candidate_conf) > (best_quality, best_conf):
            best_word, best_conf, best_quality = _clean(candidate_text), candidate_conf, quality

    for row in _group_into_rows(words):
        # Individual words - catches the normal case where OCR reads the
        # whole label correctly as one token.
        for w in row:
            consider(w["text"], w["conf"])

        # Sliding window of consecutive words glued together (up to 4 at
        # a time, since our longest prefix is 4 characters and real
        # labels aren't much longer) - catches split/fragmented reads.
        for start in range(len(row)):
            joined_text = ""
            confs = []
            for w in row[start:start + 4]:
                joined_text += w["text"]
                confs.append(w["conf"])
                if len(_clean(joined_text)) >= 4:
                    consider(joined_text, sum(confs) / len(confs))

    if best_word is None:
        return None
    return {"text": best_word, "confidence": best_conf, "exact": best_quality == 1.0}


def read_label_from_frame(frame):
    """Try OCR on a few auto-detected candidate regions AND the full
    frame, at a couple of cleanup settings each, and return the best
    KNOWN_PREFIXES match found anywhere - or None if nothing in this
    frame matched a known label prefix. Also returns the specific image
    that produced the match, cropped tightly around it, for the debug
    folder."""
    regions = find_candidate_regions(frame)
    areas = [("full frame", frame)] + [
        ("region", frame[y0:y1, x0:x1])
        for (x0, y0, x1, y1) in regions
        if frame[y0:y1, x0:x1].size > 0
    ]

    best_match, best_image = None, None
    for _label, area in areas:
        gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY) if area.ndim == 3 else area
        upscaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, thresh_normal = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_inverted = cv2.bitwise_not(thresh_normal)

        for variant in (upscaled, thresh_normal, thresh_inverted):
            for psm in (6, 11):  # 6 = a block of text, 11 = scattered/sparse text
                match = find_label_in_image(variant, psm)
                if match is None:
                    continue
                is_better = (
                    best_match is None
                    or (match["exact"], match["confidence"]) > (best_match["exact"], best_match["confidence"])
                )
                if is_better:
                    best_match, best_image = match, area

    return best_match, best_image


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_label_text.py \"path\\to\\your\\video.mp4\"")
        return

    video_path = sys.argv[1]
    print(f"Reading video: {video_path}")

    frames = get_sharp_frames(video_path)
    print(f"Grabbed {len(frames)} clear frame(s) to read.\n")

    os.makedirs(DEBUG_FOLDER, exist_ok=True)

    overall_best = None
    saved_count = 0
    for i, frame in enumerate(frames, start=1):
        match, used_image = read_label_from_frame(frame)

        print(f"--- Frame {i} ---")
        if match is None:
            print("(no known label prefix found in this frame)")
            debug_path = os.path.join(DEBUG_FOLDER, f"no_match_frame{i}.png")
            cv2.imwrite(debug_path, frame)
            print(f"  (saved the full frame to {debug_path} so you can check it visually)")
        else:
            tag = "" if match["exact"] else "  (approximate match - double check this one)"
            print(f"{match['text']}{tag}")
            saved_count += 1
            debug_path = os.path.join(DEBUG_FOLDER, f"match_{saved_count}_frame{i}.png")
            cv2.imwrite(debug_path, used_image)
            print(f"  (saved to {debug_path})")

            if overall_best is None or (match["exact"], match["confidence"]) > (overall_best["exact"], overall_best["confidence"]):
                overall_best = match
        print()

    print("=" * 40)
    print("BEST RESULT:")
    if overall_best is None:
        print("No label starting with ECKO, ECHP, APKO, or APHP was found in this video.")
        print(f"Check the '{DEBUG_FOLDER}' folder - it has the actual frames the program")
        print("looked at, so you can see for yourself whether the label is visible/in")
        print("focus in any of them, or if the camera just didn't catch it clearly.")
    else:
        print(overall_best["text"])
        if not overall_best["exact"]:
            print("(this was an approximate match - worth a quick visual double check)")
        print()
        print(f"See the '{DEBUG_FOLDER}' folder for close-up images of what matched.")


if __name__ == "__main__":
    main()