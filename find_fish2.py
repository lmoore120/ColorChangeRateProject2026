r"""
find_fish_black_background.py
------------------------------------
A variant of find_fish_and_local_background.py tuned specifically for
BLACK-background footage, where the fish's own color can be very close
to the background (low contrast). Everything else works the same way -
this is a separate script on purpose, so tuning it for black backgrounds
never risks changing behavior on your white-background footage.

TWO THINGS ARE DIFFERENT FROM THE ORIGINAL SCRIPT:

1. MORE SENSITIVE CONTRAST: uses local contrast enhancement (CLAHE) plus
   a much lower difference-from-background threshold, so a fish that's
   only subtly different from a black background can still be detected.
   This alone would also make things like glare, reflections, or a
   printed label stand out more strongly than before - which leads to
   the second change:

2. LABEL EXCLUSION: your printed labels always read as one of a few
   known codes (ECKO/ECHP/APKO/APHP). Before accepting a candidate blob
   as "the fish," this runs a quick text-recognition check on it - if
   it reads as one of those codes, it's the label, not the fish, and
   the next-largest candidate is tried instead.

Everything else (lightbox circle detection, color-checker card
detection/exclusion, border exclusion, ring sampling) is identical to
the original script.

HOW TO RUN IT:
  python find_fish_black_background.py "path\to\your\video.mp4"

Saves an image per frame showing exactly what it found (blue = lightbox
edge, yellow = color card, purple = a candidate rejected as the label,
red = fish outline, green = sampled ring).
"""

import os
import re
import sys
import cv2
import numpy as np
import pytesseract

# Points straight at Tesseract, same as your label-reading scripts
# (adjust this path if yours is installed somewhere else):
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DEBUG_FOLDER = "debug_frames"

# How different (in the contrast-enhanced image, 0-255) a pixel needs to
# be from the background to count as "not background." Lower = more
# sensitive - this is what lets a fish that's only subtly different from
# a black background still get detected. Safe to set this low BECAUSE of
# the CLAHE contrast boost below; lowering it without that would just
# pick up camera sensor noise.
DIFFERENCE_THRESHOLD = 8

# How many pixels wide the sampling ring around the fish should be.
RING_WIDTH = 15

# How close to the relevant area's edge (in pixels) counts as "touching
# the border" and gets excluded as a likely lightbox-rim/light-source
# artifact rather than the fish itself.
BORDER_MARGIN = 5

# Your printed labels always start with one of these. Used to recognize
# and reject a candidate blob as "the label," not the fish.
KNOWN_LABEL_PREFIXES = ["ECKO", "ECHP", "APKO", "APHP"]


def _enhanced_gray(frame):
    """Grayscale version of the frame with local contrast boosted (CLAHE).
    This amplifies small real brightness differences in each local patch
    of the image, which is what makes a low DIFFERENCE_THRESHOLD safe to
    use without drowning in camera noise."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    return clahe.apply(gray)


def _clean_word(word):
    return re.sub(r"[^A-Za-z0-9]", "", word).upper()


def _matches_label_prefix(word):
    cleaned = _clean_word(word)
    for prefix in KNOWN_LABEL_PREFIXES:
        if len(cleaned) < len(prefix):
            continue
        head = cleaned[:len(prefix)]
        mismatches = sum(a != b for a, b in zip(head, prefix))
        if mismatches <= 1:  # allow one OCR mistake, same as the label scripts
            return True
    return False


def looks_like_label(frame, box, pad=15):
    """Crop this candidate region and check whether it reads as one of
    the known label codes. Returns True if it does (meaning: this is the
    printed label, not the fish).

    Tries a few different cleanup methods and settings, same approach as
    the dedicated label-reading script - a single method can come up
    completely empty on real (or compressed) footage even when the text
    is perfectly legible to a human eye, so it's worth trying more than
    one before concluding there's no label here.
    """
    x, y, w, h = box
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(fw, x + w + pad), min(fh, y + h + pad)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, thresh_normal = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_inverted = cv2.bitwise_not(thresh_normal)

    for variant in (upscaled, thresh_normal, thresh_inverted):
        for psm in (6, 11):
            try:
                text = pytesseract.image_to_string(variant, config=f"--psm {psm}")
            except pytesseract.TesseractError:
                continue
            for word in text.split():
                if _matches_label_prefix(word):
                    return True
    return False


def find_color_checker_region(frame, min_blobs_in_cluster=2):
    """Directly find the color-standard card by looking for its
    distinctive colored squares (a color checker's squares are far more
    saturated than anything else likely to be in this scene - fish,
    tank plastic, water, the label). Returns a bounding box
    (x0, y0, x1, y1) generously padded to also cover the adjacent
    grey/white squares, or None if nothing card-like was found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_w * frame_h

    # HSV saturation is numerically unstable on near-black pixels (tiny
    # noise differences between R/G/B channels blow up into large
    # "saturation" values when brightness is near zero) - requiring
    # decent brightness too avoids treating dark-background camera
    # noise as if it were a saturated color checker square.
    mask = ((sat > 80) & (val > 60)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    min_area = frame_area * 0.0002
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h > 0 else 0
        if 0.4 < aspect < 2.5:
            blobs.append((x, y, w, h))

    if len(blobs) < min_blobs_in_cluster:
        return None

    max_gap = int(max(frame_w, frame_h) * 0.04)

    def boxes_close(a, b, gap):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (bx > ax + aw + gap or bx + bw < ax - gap
                    or by > ay + ah + gap or by + bh < ay - gap)

    used = [False] * len(blobs)
    clusters = []
    for i, b in enumerate(blobs):
        if used[i]:
            continue
        cluster = [b]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, b2 in enumerate(blobs):
                if used[j]:
                    continue
                if any(boxes_close(cb, b2, max_gap) for cb in cluster):
                    cluster.append(b2)
                    used[j] = True
                    changed = True
        clusters.append(cluster)

    qualifying = [c for c in clusters if len(c) >= min_blobs_in_cluster]
    if not qualifying:
        return None

    # Merge qualifying clusters that are reasonably close to each other -
    # a real card can have an internal gap (a corner marker, a label)
    # between groups of squares, so requiring everything to be in one
    # tight cluster from stage one would miss part of the card. This
    # uses a much more generous gap than stage one, but still far short
    # of "anywhere in the frame" - that's what keeps genuinely unrelated
    # noise elsewhere from getting pulled in.
    def cluster_bbox(cluster):
        x0 = min(b[0] for b in cluster)
        y0 = min(b[1] for b in cluster)
        x1 = max(b[0] + b[2] for b in cluster)
        y1 = max(b[1] + b[3] for b in cluster)
        return (x0, y0, x1 - x0, y1 - y0)

    group_gap = int(max(frame_w, frame_h) * 0.15)
    cluster_boxes = [cluster_bbox(c) for c in qualifying]
    used = [False] * len(qualifying)
    groups = []
    for i in range(len(qualifying)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(qualifying)):
                if used[j]:
                    continue
                if any(boxes_close(cluster_boxes[k], cluster_boxes[j], group_gap) for k in group):
                    group.append(j)
                    used[j] = True
                    changed = True
        groups.append(group)

    # Use the single largest merged group (by total blob area), not the
    # union of every group - this is what keeps a stray noise cluster
    # far across the frame from ever getting included.
    def group_area(group):
        return sum(b[2] * b[3] for i in group for b in qualifying[i])

    best_group = max(groups, key=group_area)
    best_cluster = [b for i in best_group for b in qualifying[i]]

    x0 = min(b[0] for b in best_cluster)
    y0 = min(b[1] for b in best_cluster)
    x1 = max(b[0] + b[2] for b in best_cluster)
    y1 = max(b[1] + b[3] for b in best_cluster)

    card_w, card_h = x1 - x0, y1 - y0
    pad_x = int(card_w * 1.4 + 20)
    pad_y = int(card_h * 0.25 + 20)

    x0 = max(0, x0 - pad_x)
    x1 = min(frame_w, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(frame_h, y1 + pad_y)

    region_area = (x1 - x0) * (y1 - y0)
    if region_area > frame_area * 0.35:
        return None

    return (x0, y0, x1, y1)


def find_vignette_circle(frame):
    """Look for a circular edge (the lightbox boundary) in the frame.
    Returns (center_x, center_y, radius) in full-resolution coordinates,
    or None if no confident circle was found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    scale = 600 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale)
    blurred = cv2.medianBlur(small, 9)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=300,
        param1=80, param2=60,
        minRadius=int(min(small.shape) * 0.3),
        maxRadius=int(min(small.shape) * 0.7),
    )

    if circles is None:
        return None

    x, y, r = circles[0][0]
    return (x / scale, y / scale, r / scale)


def get_relevant_mask(frame, circle):
    """A mask of which pixels actually matter for finding the fish."""
    h, w = frame.shape[:2]
    if circle is None:
        mask = np.full((h, w), 255, dtype=np.uint8)
    else:
        cx, cy, r = circle
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r * 0.92), 255, thickness=cv2.FILLED)

    checker_region = find_color_checker_region(frame)
    if checker_region is not None:
        x0, y0, x1, y1 = checker_region
        cv2.rectangle(mask, (x0, y0), (x1, y1), 0, thickness=cv2.FILLED)

    return mask


def most_common_brightness(frame, relevant_mask, bin_size=8):
    """Find the most common brightness (in the contrast-enhanced image)
    within the relevant area only - almost always the background."""
    gray = _enhanced_gray(frame)
    pixels = gray[relevant_mask == 255]
    if len(pixels) == 0:
        return 128
    histogram, _ = np.histogram(pixels, bins=256 // bin_size, range=(0, 256))
    most_common_bin = np.argmax(histogram)
    return most_common_bin * bin_size + bin_size // 2


def touches_border(x, y, w, h, relevant_mask, margin=BORDER_MARGIN):
    frame_h, frame_w = relevant_mask.shape
    if x <= margin or y <= margin or x + w >= frame_w - margin or y + h >= frame_h - margin:
        return True
    eroded = cv2.erode(relevant_mask, np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8))
    box_edge_ring = np.zeros_like(relevant_mask)
    cv2.rectangle(box_edge_ring, (x, y), (x + w, y + h), 255, thickness=margin)
    return bool(np.any((box_edge_ring == 255) & (eroded == 0) & (relevant_mask == 255)))


def find_fish(frame, background_brightness, relevant_mask):
    """Return (contour, bounding_box, rejected_as_label) for the fish.
    rejected_as_label is a list of boxes that were skipped because they
    read as the known label text - useful for the debug visualization."""
    gray = _enhanced_gray(frame)
    diff = cv2.absdiff(gray, np.full_like(gray, background_brightness))
    _, mask = cv2.threshold(diff, DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, relevant_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, []

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if touches_border(x, y, w, h, relevant_mask):
            continue
        area = cv2.contourArea(c)
        if area < 20:
            continue
        candidates.append((area, c, (x, y, w, h)))

    candidates.sort(key=lambda item: item[0], reverse=True)

    rejected_labels = []
    for area, contour, box in candidates:
        if looks_like_label(frame, box):
            rejected_labels.append(box)
            continue
        return contour, box, rejected_labels

    return None, None, rejected_labels


def sample_ring_around_fish(frame, fish_contour, fish_box, relevant_mask):
    fish_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(fish_mask, [fish_contour], -1, 255, thickness=cv2.FILLED)

    kernel = np.ones((RING_WIDTH * 2 + 1, RING_WIDTH * 2 + 1), np.uint8)
    expanded_mask = cv2.dilate(fish_mask, kernel)
    ring_mask = cv2.subtract(expanded_mask, fish_mask)
    ring_mask = cv2.bitwise_and(ring_mask, relevant_mask)

    ring_pixels_bgr = frame[ring_mask == 255]
    if len(ring_pixels_bgr) == 0:
        return None

    median_bgr = np.median(ring_pixels_bgr, axis=0)
    median_gray = cv2.cvtColor(np.uint8([[median_bgr]]), cv2.COLOR_BGR2GRAY)[0][0]
    return {
        "bgr": tuple(int(v) for v in median_bgr),
        "brightness": int(median_gray),
        "ring_mask": ring_mask,
    }


def get_sample_frames(video_path, how_many=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames <= 0:
        count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            if count % 30 == 0:
                frames.append(frame)
            count += 1
    else:
        step = max(1, total_frames // how_many)
        for frame_index in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()
            if success:
                frames.append(frame)
            if len(frames) >= how_many:
                break

    cap.release()
    return frames[:how_many]


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_fish_black_background.py \"path\\to\\your\\video.mp4\"")
        return

    video_path = sys.argv[1]
    print(f"Reading: {video_path}\n")

    frames = get_sample_frames(video_path)
    if not frames:
        print("Couldn't read any frames from this video.")
        return

    os.makedirs(DEBUG_FOLDER, exist_ok=True)

    for i, frame in enumerate(frames, start=1):
        circle = find_vignette_circle(frame)
        checker_region = find_color_checker_region(frame)
        relevant_mask = get_relevant_mask(frame, circle)
        bg_brightness = most_common_brightness(frame, relevant_mask)
        fish_contour, fish_box, rejected_labels = find_fish(frame, bg_brightness, relevant_mask)

        print(f"--- Frame {i} ---")
        print(f"  Lightbox circle: {'found ' + str(tuple(int(v) for v in circle)) if circle else 'not found (using whole frame)'}")
        print(f"  Color checker card: {'found ' + str(checker_region) if checker_region else 'not found'}")
        if rejected_labels:
            print(f"  Rejected as label text: {rejected_labels}")

        annotated = frame.copy()
        if circle:
            cx, cy, r = circle
            cv2.circle(annotated, (int(cx), int(cy)), int(r), (255, 0, 0), 3)
        if checker_region:
            x0, y0, x1, y1 = checker_region
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 3)
        for rx, ry, rw, rh in rejected_labels:
            cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 3)

        if fish_contour is None:
            print("  No fish-like blob found in this frame.")
            cv2.imwrite(os.path.join(DEBUG_FOLDER, f"frame_{i}_nofish.png"), annotated)
            continue

        ring_result = sample_ring_around_fish(frame, fish_contour, fish_box, relevant_mask)
        if ring_result is None:
            print("  Found a fish blob, but couldn't sample a ring around it.")
            continue

        print(f"  Fish bounding box: {fish_box}")
        print(f"  Local background right around the fish: "
              f"brightness={ring_result['brightness']}, BGR={ring_result['bgr']}")

        cv2.drawContours(annotated, [fish_contour], -1, (0, 0, 255), 2)
        annotated[ring_result["ring_mask"] == 255] = (0, 255, 0)
        debug_path = os.path.join(DEBUG_FOLDER, f"frame_{i}.png")
        cv2.imwrite(debug_path, annotated)
        print(f"  (saved to {debug_path} - blue=lightbox, yellow=color card, "
              f"magenta=rejected label, red=fish, green=sampled ring)")

    print(f"\nCheck the '{DEBUG_FOLDER}' folder to see exactly what was detected.")


if __name__ == "__main__":
    main()