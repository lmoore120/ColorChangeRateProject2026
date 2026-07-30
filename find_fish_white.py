r"""
find_fish_and_local_background.py
------------------------------------
Finds the fish in a frame (as the largest blob that looks different from
the background), then samples the background color in a ring immediately
surrounding it - NOT the whole frame.

Handles the "lightbox" vignette: depending on camera position/zoom, the
video sometimes shows the circular edge of the lightbox around the tank,
with everything outside that circle being irrelevant (dark table, room,
etc. - not part of the actual experiment). When that circle is visible,
this detects it and restricts ALL analysis (background color, fish
detection) to inside it. When no circle is visible (camera zoomed in
tighter), it just uses the whole frame - no assumption either way is
hard-coded, it's detected per frame.

HOW IT WORKS, STEP BY STEP:
  1. Tries to find a circular edge (the lightbox boundary) in the frame.
     If found, everything outside it is masked off and ignored completely
     for the rest of the analysis.
  2. Figures out the background color (white or black) by looking at the
     most common brightness WITHIN the relevant area (inside the circle,
     if there is one)
  3. Finds every blob of pixels that looks meaningfully different from
     that background
  4. Ignores blobs touching the very edge of the relevant area (the
     circle's edge, or the frame's edge if there's no circle) - a light
     source/lightbox rim is far more likely to show up right at that
     boundary than a fish swimming in open water is
  5. Picks the single largest remaining blob and calls that the fish
  6. Draws a ring a set distance around the fish's outline and samples
     the color WITHIN that ring (excluding the fish itself)

HOW TO RUN IT:
  python find_fish_and_local_background.py "path\to\your\video.mp4"

Saves an image per frame showing exactly what it found (blue = detected
lightbox boundary if any, red = fish outline, green = sampled ring) so
you can visually check it's working correctly.
"""

import os
import sys
import cv2
import numpy as np

DEBUG_FOLDER = "debug_frames"

# How different (in grayscale brightness, 0-255) a pixel needs to be from
# the background to count as "not background."
DIFFERENCE_THRESHOLD = 40

# How many pixels wide the sampling ring around the fish should be.
RING_WIDTH = 15

# How close to the relevant area's edge (in pixels) counts as "touching
# the border" and gets excluded as a likely lightbox-rim/light-source
# artifact rather than the fish itself.
BORDER_MARGIN = 5


def find_vignette_circle(frame):
    """Look for a circular edge (the lightbox boundary) in the frame.
    Returns (center_x, center_y, radius) in full-resolution coordinates,
    or None if no confident circle was found - which is expected and
    fine for footage where the camera is zoomed in past the lightbox
    edge."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    scale = 600 / max(h, w)  # work on a smaller image for speed/stability
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
    """A mask of which pixels actually matter - everything, if there's no
    lightbox circle in this frame, or just the inside of the circle
    (slightly inset from its edge) if there is one."""
    h, w = frame.shape[:2]
    if circle is None:
        return np.full((h, w), 255, dtype=np.uint8)

    cx, cy, r = circle
    mask = np.zeros((h, w), dtype=np.uint8)
    # Inset slightly so the bright rim itself isn't treated as part of
    # the scene to analyze.
    cv2.circle(mask, (int(cx), int(cy)), int(r * 0.92), 255, thickness=cv2.FILLED)
    return mask


def most_common_brightness(frame, relevant_mask, bin_size=8):
    """Find the most common grayscale brightness within the relevant
    area only - almost always the background, since it fills most of
    that area."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pixels = gray[relevant_mask == 255]
    if len(pixels) == 0:
        return 128  # shouldn't happen, but don't crash if it does
    histogram, _ = np.histogram(pixels, bins=256 // bin_size, range=(0, 256))
    most_common_bin = np.argmax(histogram)
    return most_common_bin * bin_size + bin_size // 2


def touches_border(x, y, w, h, relevant_mask, margin=BORDER_MARGIN):
    """True if this box touches the edge of the relevant area - either
    the frame's own edge, or the edge of the lightbox circle mask."""
    frame_h, frame_w = relevant_mask.shape
    if x <= margin or y <= margin or x + w >= frame_w - margin or y + h >= frame_h - margin:
        return True
    # Also check against the mask's own boundary (the circle edge, if
    # there is one) by seeing if the box's border sits on mask pixels
    # that are right next to "outside the mask."
    eroded = cv2.erode(relevant_mask, np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8))
    box_edge_ring = np.zeros_like(relevant_mask)
    cv2.rectangle(box_edge_ring, (x, y), (x + w, y + h), 255, thickness=margin)
    return bool(np.any((box_edge_ring == 255) & (eroded == 0) & (relevant_mask == 255)))


def find_fish(frame, background_brightness, relevant_mask):
    """Return (contour, bounding_box) for the fish, or (None, None) if
    nothing fish-like was found in this frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, np.full_like(gray, background_brightness))
    _, mask = cv2.threshold(diff, DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, relevant_mask)  # ignore anything outside the relevant area

    # Clean up small speckles (sensor noise, compression artifacts) so
    # they don't get treated as tiny separate blobs.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if touches_border(x, y, w, h, relevant_mask):
            continue  # likely a lightbox rim/light source, not the fish
        area = cv2.contourArea(c)
        if area < 20:
            continue  # too small to be the fish - just noise
        candidates.append((area, c, (x, y, w, h)))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_contour, best_box = candidates[0]
    return best_contour, best_box


def sample_ring_around_fish(frame, fish_contour, fish_box, relevant_mask):
    """Sample the color in a ring around (but not touching) the fish,
    staying within the relevant area."""
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
        print("Usage: python find_fish_and_local_background.py \"path\\to\\your\\video.mp4\"")
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
        relevant_mask = get_relevant_mask(frame, circle)
        bg_brightness = most_common_brightness(frame, relevant_mask)
        fish_contour, fish_box = find_fish(frame, bg_brightness, relevant_mask)

        print(f"--- Frame {i} ---")
        print(f"  Lightbox circle: {'found ' + str(tuple(int(v) for v in circle)) if circle else 'not found (using whole frame)'}")

        if fish_contour is None:
            print("  No fish-like blob found in this frame.")
            annotated = frame.copy()
            if circle:
                cx, cy, r = circle
                cv2.circle(annotated, (int(cx), int(cy)), int(r), (255, 0, 0), 3)
            cv2.imwrite(os.path.join(DEBUG_FOLDER, f"frame_{i}_nofish.png"), annotated)
            continue

        ring_result = sample_ring_around_fish(frame, fish_contour, fish_box, relevant_mask)
        if ring_result is None:
            print("  Found a fish blob, but couldn't sample a ring around it.")
            continue

        print(f"  Fish bounding box: {fish_box}")
        print(f"  Local background right around the fish: "
              f"brightness={ring_result['brightness']}, BGR={ring_result['bgr']}")

        annotated = frame.copy()
        if circle:
            cx, cy, r = circle
            cv2.circle(annotated, (int(cx), int(cy)), int(r), (255, 0, 0), 3)
        cv2.drawContours(annotated, [fish_contour], -1, (0, 0, 255), 2)
        annotated[ring_result["ring_mask"] == 255] = (0, 255, 0)
        debug_path = os.path.join(DEBUG_FOLDER, f"frame_{i}.png")
        cv2.imwrite(debug_path, annotated)
        print(f"  (saved to {debug_path} - blue = lightbox edge, red = fish, green = sampled ring)")

    print(f"\nCheck the '{DEBUG_FOLDER}' folder to see exactly what was detected.")


if __name__ == "__main__":
    main()
