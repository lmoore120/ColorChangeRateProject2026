r"""
review_frames.py - Look at every extracted frame and keep, drop, or replace it.

WHY
---
Automatic frame selection picks the sharpest moment within half a second
of each target time, which is the best it can do without knowing what a
usable frame looks like. It can't tell that the fish has turned to face
the camera, swum behind the label, or half left the frame - all of which
make a frame useless for landmarking, and none of which show up in a
sharpness score.

This shows you each frame and lets you decide. Rejecting one doesn't
leave a hole: you can scrub to a nearby moment and take that instead, and
the manifest records the true time of whatever you kept, so the timing in
the analysis stays correct.

HOW TO RUN
----------
    python review_frames.py APHP\frames --videos APHP\videos

CONTROLS
    ENTER / Y     keep this frame, go to the next
    N / DELETE    reject it - removes the frame and its manifest row
    LEFT / RIGHT  scrub 0.5s earlier / later and re-take the frame
    SHIFT+LEFT/RIGHT   scrub 2s
    R             back to the originally chosen moment
    S             re-take the sharpest frame within 1s of here
    B             back to the previous frame
    Q             save and quit (everything reviewed so far is kept)

Nothing is written until you press ENTER on a frame, so quitting never
leaves the manifest half-updated.
"""

import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np

ARROW_LEFT = (2424832, 65361, 81)
ARROW_RIGHT = (2555904, 65363, 83)
ARROW_UP = (2490368, 65362, 82)
ARROW_DOWN = (2621440, 65364, 84)

VIEW_WIDTH, VIEW_HEIGHT = 1500, 950
BAR_HEIGHT = 150


def _wheel_delta(flags):
    """Which way the wheel turned. OpenCV puts it in the high 16 bits of
    flags, so testing flags > 0 gets the direction wrong on Windows."""
    if hasattr(cv2, "getMouseWheelDelta"):
        try:
            return cv2.getMouseWheelDelta(flags)
        except Exception:
            pass
    high = (int(flags) >> 16) & 0xFFFF
    if high > 32767:
        high -= 65536
    return high if high else (1 if flags > 0 else -1)


def sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def find_video(video_folder, video_name):
    """Locate the source video for a manifest row."""
    direct = os.path.join(video_folder, video_name)
    if os.path.exists(direct):
        return direct
    stem = os.path.splitext(video_name)[0]
    for name in os.listdir(video_folder):
        if os.path.splitext(name)[0].lower() == stem.lower():
            return os.path.join(video_folder, name)
    return None


def grab_at(cap, seconds, fps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(seconds * fps))))
    ok, frame = cap.read()
    return frame if ok else None


def sharpest_near(cap, seconds, fps, window=1.0):
    """Best frame within +/- window seconds - the same search extraction
    uses, but centred wherever you have scrubbed to."""
    best, best_score, best_seconds = None, -1.0, seconds
    first = max(0, int((seconds - window) * fps))
    last = int((seconds + window) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for index in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        score = sharpness(frame)
        if score > best_score:
            best, best_score, best_seconds = frame.copy(), score, index / fps
    return best, best_seconds, best_score


def draw_buttons(canvas, buttons, top, height=42):
    """Draw a row of tappable buttons and return their hit boxes.

    Everything has a button as well as a shortcut so the whole tool can be
    driven by touch alone - no keyboard, no mouse. Boxes are sized for a
    fingertip rather than a pointer."""
    boxes = {}
    x = 12
    for label, action, colour in buttons:
        width = 30 + 11 * len(label)
        if x + width > canvas.shape[1] - 12:
            break
        cv2.rectangle(canvas, (x, top), (x + width, top + height), colour, -1)
        cv2.rectangle(canvas, (x, top), (x + width, top + height), (110, 110, 110), 1)
        cv2.putText(canvas, label, (x + 15, top + int(height * 0.63)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        boxes[action] = (x, top, x + width, top + height)
        x += width + 8
    return boxes


def button_at(boxes, x, y):
    for action, (x0, y0, x1, y1) in boxes.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return action
    return None


GRID_MARGIN = 6
GRID_LABEL = 18


def build_contact_sheet(thumbs, rejected, columns, cell_w, cell_h, rows_meta):
    """Tile every frame of one video into a single picture.

    Looking at 25 frames side by side makes an odd one out obvious in a
    way that stepping through them one at a time does not - a frame where
    the fish has turned, or drifted out of shot, stands out immediately
    against its neighbours."""
    n = len(thumbs)
    grid_rows = (n + columns - 1) // columns
    sheet_w = columns * (cell_w + GRID_MARGIN) + GRID_MARGIN
    sheet_h = grid_rows * (cell_h + GRID_LABEL + GRID_MARGIN) + GRID_MARGIN
    sheet = np.full((sheet_h, sheet_w, 3), 24, np.uint8)

    boxes = []
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, columns)
        x = GRID_MARGIN + c * (cell_w + GRID_MARGIN)
        y = GRID_MARGIN + r * (cell_h + GRID_LABEL + GRID_MARGIN)
        sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
        boxes.append((x, y, x + cell_w, y + cell_h))

        meta = rows_meta[i]
        is_out = i in rejected
        colour = (60, 60, 220) if is_out else (90, 90, 90)
        cv2.rectangle(sheet, (x - 2, y - 2), (x + cell_w + 1, y + cell_h + 1),
                      colour, 2 if is_out else 1)
        if is_out:
            # A clear cross, so a rejected frame can't be mistaken for a
            # kept one at a glance.
            cv2.line(sheet, (x, y), (x + cell_w, y + cell_h), (60, 60, 220), 2)
            cv2.line(sheet, (x + cell_w, y), (x, y + cell_h), (60, 60, 220), 2)
        cv2.putText(sheet,
                    f"{meta['frame_number']}  {float(meta['seconds']):.0f}s"
                    + ("  REJECTED" if is_out else ""),
                    (x + 2, y + cell_h + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (120, 120, 255) if is_out else (190, 190, 190), 1, cv2.LINE_AA)
    return sheet, boxes


def review_one_video_as_sheet(video, rows, options, window):
    """Show one fish's frames all at once and take a single decision.

    One yes/no per fish rather than per frame: 100 videos is 100
    decisions instead of 2,500. Individual frames can still be picked off
    by clicking them, for the case where a video is fine apart from one
    bad moment.

    Returns a set of image names to reject (empty = keep everything,
    all of them = the whole video was rejected), or None to quit."""
    thumbs, usable = [], []
    for row in rows:
        image = cv2.imread(os.path.join(options.frames, row["image"]))
        if image is not None:
            usable.append(row)
            thumbs.append(image)
    if not thumbs:
        return set()

    source_h, source_w = thumbs[0].shape[:2]
    if options.columns and options.columns > 0:
        candidates = [options.columns]
    else:
        # Pick whatever makes each frame biggest. A fixed 5x5 is a poor
        # fit for portrait video: 25 frames of 1080x1920 in five columns
        # leaves cells 82px wide, too small to judge a fish by. Nine
        # across by three down gives 147x262 for the same screen.
        candidates = range(1, len(thumbs) + 1)

    best = None
    for columns in candidates:
        grid_rows = (len(thumbs) + columns - 1) // columns
        room_w = (VIEW_WIDTH - GRID_MARGIN * (columns + 1)) / columns
        room_h = ((VIEW_HEIGHT - BAR_HEIGHT
                   - (GRID_LABEL + GRID_MARGIN) * grid_rows) / grid_rows)
        if room_w < 30 or room_h < 30:
            continue
        scale = min(room_w / source_w, room_h / source_h)
        if best is None or scale > best[0]:
            best = (scale, columns, grid_rows)
    if best is None:
        best = (0.08, len(thumbs), 1)
    scale, columns, _ = best
    cell_w = max(30, int(source_w * scale))
    cell_h = max(30, int(source_h * scale))
    thumbs = [cv2.resize(t, (cell_w, cell_h)) for t in thumbs]

    view_h = VIEW_HEIGHT - BAR_HEIGHT
    state = {"rejected": set(), "outcome": None, "detail": None, "boxes": [],
             "zoom": 1.0, "pan": np.zeros(2), "pan_from": None,
             "buttons": {}, "touch_from": None, "touch_moved": False}

    def to_sheet(x, y):
        """Window pixel -> position on the full-size contact sheet. Every
        click goes through this, so rejecting the right frame doesn't
        depend on how far you have zoomed in."""
        return np.array([x / state["zoom"] + state["pan"][0],
                         (y - BAR_HEIGHT) / state["zoom"] + state["pan"][1]])

    def zoom_by(factor, at_x, at_y):
        anchor = to_sheet(at_x, at_y)
        state["zoom"] = float(np.clip(state["zoom"] * factor, 0.3, 12.0))
        state["pan"] = np.array([anchor[0] - at_x / state["zoom"],
                                 anchor[1] - (at_y - BAR_HEIGHT) / state["zoom"]])

    def fit():
        state["zoom"] = 1.0
        state["pan"] = np.zeros(2)

    def on_mouse(event, x, y, flags, _):
        if y < BAR_HEIGHT:
            if event == cv2.EVENT_LBUTTONDOWN:
                action = button_at(state.get("buttons", {}), x, y)
                if action == "keep":
                    state["outcome"] = "keep"
                elif action == "reject":
                    state["outcome"] = "reject_all"
                elif action == "reset":
                    state["rejected"].clear()
                elif action == "back":
                    state["outcome"] = "back"
                elif action == "quit":
                    state["outcome"] = "quit"
                elif action == "zoomin":
                    zoom_by(1.3, VIEW_WIDTH // 2, BAR_HEIGHT + view_h // 2)
                elif action == "zoomout":
                    zoom_by(1 / 1.3, VIEW_WIDTH // 2, BAR_HEIGHT + view_h // 2)
                elif action == "fit":
                    fit()
                elif action == "up":
                    state["pan"][1] -= 200 / state["zoom"]
                elif action == "down":
                    state["pan"][1] += 200 / state["zoom"]
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            # Scrolls the sheet up and down. Zoom is on + / - and ctrl+wheel,
            # because with 25 frames in view scrolling is what you reach for
            # far more often.
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                zoom_by(1.2 if _wheel_delta(flags) > 0 else 1 / 1.2, x, y)
            else:
                state["pan"][1] -= (120 / state["zoom"]) * (
                    1 if _wheel_delta(flags) > 0 else -1)
            return
        # Middle-drag pans, so it doesn't clash with left-click to reject
        # or right-click to scrub.
        if event == cv2.EVENT_MBUTTONDOWN:
            state["pan_from"] = (x, y, state["pan"].copy())
            return
        if event == cv2.EVENT_MOUSEMOVE and state["pan_from"] is not None:
            x0, y0, pan0 = state["pan_from"]
            state["pan"] = pan0 - np.array([(x - x0) / state["zoom"],
                                            (y - y0) / state["zoom"]])
            return
        if event == cv2.EVENT_MBUTTONUP:
            state["pan_from"] = None
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            state["touch_from"] = (x, y, state["pan"].copy())
            state["touch_moved"] = False
            return
        if event == cv2.EVENT_MOUSEMOVE and state.get("touch_from") is not None:
            x0, y0, pan0 = state["touch_from"]
            if abs(x - x0) > 12 or abs(y - y0) > 12:
                # Dragging on the sheet scrolls it, the way a touchscreen
                # is expected to behave. Only a tap that hasn't moved
                # counts as choosing a frame.
                state["touch_moved"] = True
                state["pan"] = pan0 - np.array([(x - x0) / state["zoom"],
                                                (y - y0) / state["zoom"]])
            return
        if event == cv2.EVENT_LBUTTONUP and state.get("touch_from") is not None:
            moved = state.get("touch_moved")
            state["touch_from"] = None
            if moved:
                return
            point = to_sheet(x, y)
            for i, (x0, y0, x1, y1) in enumerate(state["boxes"]):
                if x0 <= point[0] <= x1 and y0 <= point[1] <= y1:
                    if i in state["rejected"]:
                        state["rejected"].discard(i)
                    else:
                        state["rejected"].add(i)
                    return
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            point = to_sheet(x, y)
            for i, (x0, y0, x1, y1) in enumerate(state["boxes"]):
                if x0 <= point[0] <= x1 and y0 <= point[1] <= y1:
                    state["detail"] = i
                    return
            return

    cv2.setMouseCallback(window, on_mouse)

    while state["outcome"] is None:
        sheet, boxes = build_contact_sheet(thumbs, state["rejected"], columns,
                                           cell_w, cell_h, usable)
        state["boxes"] = boxes

        canvas = np.full((VIEW_HEIGHT, VIEW_WIDTH, 3), 24, np.uint8)
        canvas[:BAR_HEIGHT] = (28, 28, 28)
        transform = np.float32([
            [state["zoom"], 0, -state["pan"][0] * state["zoom"]],
            [0, state["zoom"], -state["pan"][1] * state["zoom"]]])
        canvas[BAR_HEIGHT:] = cv2.warpAffine(
            sheet, transform, (VIEW_WIDTH, view_h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(24, 24, 24))

        label = usable[0].get("label", "")
        keeping = len(thumbs) - len(state["rejected"])
        heading = f"{video}"
        if label and label != video.split(".")[0]:
            heading += f"   [{label}]"
        heading += f"   -  {keeping} of {len(thumbs)} frames"
        cv2.putText(canvas, heading, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.64,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    "Y / ENTER  keep this fish        N  reject the WHOLE fish",
                    (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 230, 120), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas,
                    "tap a frame to drop it  |  long-press / right-click to "
                    "open and scrub  |  drag to scroll",
                    (14, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1,
                    cv2.LINE_AA)
        state["buttons"] = draw_buttons(canvas, [
            ("Keep fish", "keep", (35, 105, 35)),
            ("Reject fish", "reject", (35, 35, 120)),
            ("Reset", "reset", (58, 58, 58)),
            ("Back", "back", (58, 58, 58)),
            ("Scroll up", "up", (58, 58, 58)),
            ("Scroll down", "down", (58, 58, 58)),
            ("Zoom +", "zoomin", (58, 58, 58)),
            ("Zoom -", "zoomout", (58, 58, 58)),
            ("Fit", "fit", (58, 58, 58)),
            ("Save & quit", "quit", (58, 58, 58)),
        ], top=94)
        if state["zoom"] != 1.0:
            cv2.putText(canvas, f"{state['zoom'] * 100:.0f}%",
                        (VIEW_WIDTH - 80, BAR_HEIGHT - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1,
                        cv2.LINE_AA)
        cv2.imshow(window, canvas)

        if state["detail"] is not None:
            index = state["detail"]
            state["detail"] = None
            # Walk forward through frames with TAB without coming back to
            # the sheet each time, for the common case where several
            # consecutive frames need fixing.
            while index is not None and 0 <= index < len(usable):
                before = (float(usable[index - 1]["seconds"])
                          if index > 0 else None)
                after = (float(usable[index + 1]["seconds"])
                         if index + 1 < len(usable) else None)
                changed = scrub_one_frame(usable[index], options, window,
                                          neighbours=(before, after))

                if changed == "reject":
                    state["rejected"].add(index)
                elif changed not in (None, "quit", "next"):
                    fresh = cv2.imread(os.path.join(options.frames,
                                                    usable[index]["image"]))
                    if fresh is not None:
                        thumbs[index] = cv2.resize(fresh, (cell_w, cell_h))
                    state["rejected"].discard(index)

                if changed == "quit":
                    break
                if changed == "next":
                    fresh = cv2.imread(os.path.join(options.frames,
                                                    usable[index]["image"]))
                    if fresh is not None:
                        thumbs[index] = cv2.resize(fresh, (cell_w, cell_h))
                    index += 1
                    if index >= len(usable):
                        break
                    continue
                break
            cv2.setMouseCallback(window, on_mouse)
            continue

        raw = cv2.waitKeyEx(20)
        if raw == -1:
            continue
        pan_step = 120 / state["zoom"]
        if raw in ARROW_LEFT:
            state["pan"][0] -= pan_step
            continue
        if raw in ARROW_RIGHT:
            state["pan"][0] += pan_step
            continue
        if raw in ARROW_UP:
            state["pan"][1] -= pan_step
            continue
        if raw in ARROW_DOWN:
            state["pan"][1] += pan_step
            continue
        key = raw & 0xFF
        if key in (ord("+"), ord("=")):
            zoom_by(1.3, VIEW_WIDTH // 2, BAR_HEIGHT + view_h // 2)
            continue
        if key in (ord("-"), ord("_")):
            zoom_by(1 / 1.3, VIEW_WIDTH // 2, BAR_HEIGHT + view_h // 2)
            continue
        if key == ord("0"):
            fit()
            continue
        if key in (13, 10, ord("y"), ord("Y")):
            state["outcome"] = "keep"
        elif key in (ord("n"), ord("N")):
            state["outcome"] = "reject_all"
        elif key in (ord("a"), ord("A")):
            state["rejected"].clear()
        elif key in (ord("b"), ord("B")):
            state["outcome"] = "back"
        elif key in (ord("q"), ord("Q"), 27):
            state["outcome"] = "quit"

    if state["outcome"] == "quit":
        return None
    if state["outcome"] == "back":
        return "back"
    if state["outcome"] == "reject_all":
        return {r["image"] for r in usable}
    return {usable[i]["image"] for i in state["rejected"]}


def draw(frame, row, seconds, score, position, total, changed, message,
         neighbours=None):
    height, width = frame.shape[:2]
    scale = min((VIEW_HEIGHT - BAR_HEIGHT) / height, VIEW_WIDTH / width)
    shown = cv2.resize(frame, None, fx=scale, fy=scale)

    canvas = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), np.uint8)
    canvas[:BAR_HEIGHT] = (28, 28, 28)
    y0 = BAR_HEIGHT + (VIEW_HEIGHT - BAR_HEIGHT - shown.shape[0]) // 2
    x0 = (VIEW_WIDTH - shown.shape[1]) // 2
    canvas[y0:y0 + shown.shape[0], x0:x0 + shown.shape[1]] = shown

    heading = (f"{row['video']}   frame {row['frame_number']} of {total}   "
               f"t = {seconds:.2f}s")
    if changed:
        heading += f"   (moved from {float(row['seconds']):.2f}s)"
    cv2.putText(canvas, heading, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"sharpness {score:.0f}", (14, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 230, 120) if score > 90 else (0, 190, 255), 1, cv2.LINE_AA)

    # Where the frames either side sit, so you can see how much room there
    # is to move without crossing them.
    if neighbours:
        before, after = neighbours
        parts = []
        if before is not None:
            parts.append(f"previous frame at {before:.1f}s")
        parts.append(f"<< you are at {seconds:.1f}s >>")
        if after is not None:
            parts.append(f"next frame at {after:.1f}s")
        text = "     ".join(parts)
        cv2.putText(canvas, text, (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (190, 190, 190), 1, cv2.LINE_AA)

        # Moving past a neighbour would put the frames out of time order,
        # and the rate is fitted against real elapsed seconds - so an
        # out-of-order frame is a genuine problem, not just untidy.
        out_of_order = ((before is not None and seconds <= before)
                        or (after is not None and seconds >= after))
        if out_of_order:
            cv2.rectangle(canvas, (0, 0), (VIEW_WIDTH, BAR_HEIGHT), (0, 90, 190), 3)
            cv2.putText(canvas,
                        "OUT OF ORDER - this would cross a neighbouring frame",
                        (230, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (0, 190, 255), 1, cv2.LINE_AA)

    bar_left, bar_right = 200, VIEW_WIDTH - 260
    cv2.rectangle(canvas, (bar_left, 40), (bar_right, 52), (60, 60, 60), -1)
    filled = bar_left + int((bar_right - bar_left) * position / max(1, total))
    cv2.rectangle(canvas, (bar_left, 40), (filled, 52), (90, 180, 90), -1)
    cv2.putText(canvas, f"{position} / {total}", (bar_right + 12, 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)

    buttons = draw_buttons(canvas, [
        ("Keep", "keep", (35, 105, 35)),
        ("Reject", "reject", (35, 35, 120)),
        ("<< 2s", "back2", (58, 58, 58)),
        ("< 0.5s", "back05", (58, 58, 58)),
        ("0.5s >", "fwd05", (58, 58, 58)),
        ("2s >>", "fwd2", (58, 58, 58)),
        ("Sharpest", "sharpest", (58, 58, 58)),
        ("Reset", "reset", (58, 58, 58)),
        ("Next frame", "next", (100, 70, 20)),
        ("Back to sheet", "sheet", (58, 58, 58)),
    ], top=94)
    return canvas, buttons
    if message:
        cv2.putText(canvas, message, (VIEW_WIDTH - 640, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 190, 255), 1, cv2.LINE_AA)


def scrub_one_frame(row, options, window, neighbours=None):
    """Open one frame full size so you can scrub to a nearby moment.

    Returns "quit", "next", "reject", the new time as a float if the frame
    was replaced, or None if nothing changed."""
    video_path = find_video(options.videos, row["video"])
    if video_path is None:
        return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    original = float(row["seconds"])
    seconds = original
    frame = grab_at(cap, seconds, fps)
    if frame is None:
        cap.release()
        return None
    score = sharpness(frame)
    message = ""
    outcome = None
    held = 0          # how many frames in a row an arrow has been down
    tapped = {"action": None, "boxes": {}}

    def on_tap(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            action = button_at(tapped["boxes"], x, y)
            if action:
                tapped["action"] = action

    cv2.setMouseCallback(window, on_tap)

    while True:
        canvas, boxes = draw(frame, row, seconds, score, 1, 1,
                             abs(seconds - original) > 1e-6, message, neighbours)
        tapped["boxes"] = boxes
        cv2.imshow(window, canvas)

        raw = cv2.waitKeyEx(20)

        # A button tap stands in for the matching key, so the same code
        # handles both and they can't drift apart.
        if tapped["action"]:
            action = tapped["action"]
            tapped["action"] = None
            raw = {"keep": 13, "reject": ord("n"), "sharpest": ord("s"),
                   "reset": ord("r"), "next": 9, "sheet": ord("q")}.get(action)
            if raw is None:
                step = {"back2": -2.0, "back05": -0.5,
                        "fwd05": 0.5, "fwd2": 2.0}[action]
                seconds = max(0.0, seconds + step)
                candidate = grab_at(cap, seconds, fps)
                if candidate is not None:
                    frame, score, message = candidate, sharpness(candidate), ""
                continue

        if raw == -1:
            held = 0          # key released - back to fine steps
            continue

        step = None
        if raw in ARROW_LEFT:
            step = -0.5
        elif raw in ARROW_RIGHT:
            step = 0.5
        elif raw in ARROW_UP:
            step = 2.0
        elif raw in ARROW_DOWN:
            step = -2.0

        if step is not None:
            # Holding an arrow accelerates: the first few presses stay at
            # 0.5s for fine positioning, then it speeds up so you can cross
            # a 25-second gap without pressing fifty times.
            held += 1
            if held > 12:
                step *= 8
            elif held > 5:
                step *= 3
            seconds = max(0.0, seconds + step)
            candidate = grab_at(cap, seconds, fps)
            if candidate is not None:
                frame, score = candidate, sharpness(candidate)
                message = f"x{int(abs(step / 0.5))} speed" if held > 5 else ""
            continue
        held = 0

        key = raw & 0xFF
        if key in (13, 10, ord("y"), ord("Y")):
            if abs(seconds - original) > 1e-6:
                cv2.imwrite(os.path.join(options.frames, row["image"]), frame,
                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
                row["seconds"] = f"{seconds:.3f}"
                row["minutes"] = f"{seconds / 60:.4f}"
                if "video_frame_index" in row:
                    row["video_frame_index"] = str(int(round(seconds * fps)))
                if "timing_error_seconds" in row and "nominal_seconds" in row:
                    row["timing_error_seconds"] = (
                        f"{seconds - float(row['nominal_seconds']):.3f}")
                row["sharpness"] = f"{score:.1f}"
                outcome = seconds
            break
        if key in (ord("n"), ord("N"), 8, 127):
            outcome = "reject"
            break
        if key in (ord("s"), ord("S")):
            candidate, at, sc = sharpest_near(cap, seconds, fps, 1.0)
            if candidate is not None:
                frame, seconds, score = candidate, at, sc
                message = f"sharpest within 1s: {at:.2f}s"
            continue
        if key in (ord("r"), ord("R")):
            seconds = original
            frame = grab_at(cap, seconds, fps)
            score = sharpness(frame)
            message = "back to the original moment"
            continue
        if key == 9:              # TAB - keep this one and open the next
            if abs(seconds - original) > 1e-6:
                cv2.imwrite(os.path.join(options.frames, row["image"]), frame,
                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
                row["seconds"] = f"{seconds:.3f}"
                row["minutes"] = f"{seconds / 60:.4f}"
                if "video_frame_index" in row:
                    row["video_frame_index"] = str(int(round(seconds * fps)))
                if "timing_error_seconds" in row and "nominal_seconds" in row:
                    row["timing_error_seconds"] = (
                        f"{seconds - float(row['nominal_seconds']):.3f}")
                row["sharpness"] = f"{score:.1f}"
            outcome = "next"
            break
        if key in (ord("q"), ord("Q"), 27):
            outcome = "quit"
            break

    cap.release()
    return outcome


def main():
    parser = argparse.ArgumentParser(
        description="Review extracted frames as a contact sheet: click to "
                    "reject, right-click to scrub.")
    parser.add_argument("frames", help="the folder of extracted frames")
    parser.add_argument("--videos", required=True,
                        help="folder holding the source videos")
    parser.add_argument("--only", default=None,
                        help="review just one video, e.g. --only IMG_0035.MP4")
    parser.add_argument("--redo", action="store_true",
                        help="review everything again, ignoring what has "
                             "already been decided")
    parser.add_argument("--columns", type=int, default=9,
                        help="frames per row in the contact sheet "
                             "(default: %(default)s). 9 across suits 25 "
                             "portrait frames. Pass 0 to let it pick whatever "
                             "makes each frame largest.")
    options = parser.parse_args()

    manifest_path = os.path.join(options.frames, "frames_manifest.csv")
    if not os.path.exists(manifest_path):
        print("No frames_manifest.csv in", options.frames)
        return 1
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())

    if options.only:
        chosen = [r for r in rows if r["video"] == options.only]
        if not chosen:
            print("No frames for", options.only)
            print("Videos in this manifest:")
            for v in sorted({r["video"] for r in rows})[:20]:
                print("  ", v)
            return 1
    else:
        chosen = rows

    by_video = {}
    for row in chosen:
        by_video.setdefault(row["video"], []).append(row)
    for video in by_video:
        by_video[video].sort(key=lambda r: float(r["seconds"]))

    # RESUME. Videos already decided are recorded, so pressing Save & quit
    # part-way through 100 fish doesn't mean starting over. Without this
    # the manifest kept your rejections but nothing remembered which fish
    # you had already approved.
    progress_path = os.path.join(options.frames, "reviewed_videos.txt")
    reviewed = set()
    if os.path.exists(progress_path) and not options.redo:
        with open(progress_path, encoding="utf-8") as handle:
            reviewed = {line.strip() for line in handle if line.strip()}
    if reviewed:
        remaining = [v for v in by_video if v not in reviewed]
        print(f"{len(reviewed)} video(s) already reviewed - skipping them.")
        if not remaining:
            print("Everything in this folder has been reviewed. "
                  "Pass --redo to go through it again.")
            return 0
        print(f"{len(remaining)} left.\n")
        by_video = {v: rows for v, rows in by_video.items() if v not in reviewed}

    print(f"{len(by_video)} video(s), "
          f"{sum(len(v) for v in by_video.values())} frames.")
    print("Click a frame to reject it, right-click to scrub, ENTER for the "
          "next video.\n")

    window = "review frames"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    rejected_names = set()
    stopped = False
    videos = sorted(by_video)
    index = 0
    while index < len(videos):
        video = videos[index]
        print(f"[{index + 1}/{len(videos)}] {video}")
        result = review_one_video_as_sheet(video, by_video[video], options, window)
        if result is None:
            print("  stopped - everything reviewed so far is kept")
            stopped = True
            break
        if result == "back":
            if index > 0:
                index -= 1
                # Undo the previous video's decision so it can be retaken.
                rejected_names -= {r["image"] for r in by_video[videos[index]]}
                if os.path.exists(progress_path):
                    with open(progress_path, encoding="utf-8") as handle:
                        done = [l.strip() for l in handle if l.strip()]
                    done = [v for v in done if v != videos[index]]
                    with open(progress_path, "w", encoding="utf-8") as handle:
                        handle.write("\n".join(done) + ("\n" if done else ""))
                print(f"  back to {videos[index]}")
            continue
        rejected_names |= result
        with open(progress_path, "a", encoding="utf-8") as handle:
            handle.write(video + "\n")
        total = len(by_video[video])
        if len(result) == total:
            print(f"  REJECTED the whole video ({total} frames)")
        elif result:
            print(f"  rejected {len(result)} of {total}")
        else:
            print(f"  kept all {total}")
        index += 1

    cv2.destroyAllWindows()

    # Delete the rejected images.
    for name in rejected_names:
        path = os.path.join(options.frames, name)
        if os.path.exists(path):
            os.remove(path)

    # Rewrite the manifest: drop rejections, keep any timing changes made
    # while scrubbing. Rows never reached are left untouched.
    edited = {r["image"]: r for r in chosen}
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        everything = list(csv.DictReader(handle))
    final = []
    for row in everything:
        if row["image"] in rejected_names:
            continue
        final.append(edited.get(row["image"], row))

    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final)

    rejected_rows = [r for r in chosen if r["image"] in rejected_names]
    print(f"\nkept {len(final)}, rejected {len(rejected_names)}"
          + ("  (stopped early)" if stopped else ""))
    if stopped:
        print(f"Your place is saved - run the same command again to carry on "
              f"from {videos[index] if index < len(videos) else 'the end'}.")
    print(f"manifest updated: {manifest_path}")

    record_rejected_videos(by_video, rejected_names, options)
    report_gaps(final, rejected_rows, options)
    return 0


def record_rejected_videos(by_video, rejected_names, options):
    """Keep a running list of fish rejected outright.

    Without this a rejected fish just vanishes - its frames are gone and
    nothing says why the sample size dropped. That list is needed for the
    methods section, and to tell a fish you deliberately excluded from one
    you simply never processed. Appended to, so it builds up across runs
    and across populations."""
    import datetime

    fully_rejected = []
    for video, rows in by_video.items():
        names = {r["image"] for r in rows}
        if names and names <= rejected_names:
            first = rows[0]
            fully_rejected.append({
                "video": video,
                "label": first.get("label", ""),
                "background": first.get("background", ""),
                "frames": len(rows),
                "recorded": first.get("video_recorded_at", ""),
                "rejected_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "frames_folder": options.frames,
            })
    if not fully_rejected:
        return

    # One file per project rather than per folder, so the whole study's
    # exclusions end up in one place.
    parent = os.path.dirname(os.path.normpath(options.frames)) or "."
    path = os.path.join(parent, "rejected_videos.csv")
    fields = ["video", "label", "background", "frames", "recorded",
              "rejected_on", "frames_folder"]

    existing = []
    if os.path.exists(path):
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                existing = [r for r in csv.DictReader(handle)]
        except Exception:
            existing = []
    already = {r.get("video") for r in existing}
    new = [r for r in fully_rejected if r["video"] not in already]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + new)

    print("\n" + "=" * 62)
    print(f"FISH REJECTED OUTRIGHT: {len(fully_rejected)} this run")
    print("=" * 62)
    for row in fully_rejected:
        label = row["label"] or "(no label)"
        print(f"  {row['video']}  {label}  {row['background']}  "
              f"{row['frames']} frames")
    print(f"\nRunning list ({len(existing) + len(new)} total): {path}")
    print("Keep this - it is the record of which fish were excluded and when.")


def report_gaps(final, rejected, options):
    """List videos that are now short of frames, and write out commands to
    refill them.

    A rejected frame leaves a gap at a particular moment, and the gaps
    matter unevenly: losing the first frame moves the baseline the whole
    plasticity calculation is anchored on, so those are called out
    separately."""
    # How many frames each video still has, and which times were lost.
    remaining = {}
    for row in final:
        remaining.setdefault(row["video"], []).append(float(row["seconds"]))
    lost = {}
    for row in rejected:
        lost.setdefault(row["video"], []).append(
            (int(row["frame_number"]), float(row["seconds"])))

    if not lost:
        print("\nNo frames were rejected - nothing to refill.")
        return

    expected = max((len(v) for v in remaining.values()), default=0) + max(
        (len(v) for v in lost.values()), default=0)

    print("\n" + "=" * 62)
    print("VIDEOS NOW SHORT OF FRAMES")
    print("=" * 62)
    commands = []
    for video in sorted(lost):
        have = len(remaining.get(video, []))
        gaps = sorted(lost[video])
        times = ", ".join(f"{t:.1f}" for _, t in gaps)
        first_lost = any(number == 1 for number, _ in gaps)
        print(f"\n  {video}: {have} frames left, {len(gaps)} rejected")
        print(f"    lost at t = {times} s")
        if first_lost:
            print("    *** the FIRST frame was rejected - that is the baseline")
            print("        the whole plasticity calculation is anchored on, so")
            print("        this video needs a replacement near t=0 ***")

        video_path = os.path.normpath(os.path.join(options.videos, video))
        at_seconds = ",".join(f"{t:.1f}" for _, t in gaps)
        commands.append(
            f'python extract_frames.py "{video_path}" '
            f'--at-seconds "{at_seconds}" --out "{options.frames}"')

    print("\n" + "=" * 62)
    print("TO REFILL THEM - copy and paste these")
    print("=" * 62)
    print("\n# Each pulls frames at the SAME times that were rejected. If the")
    print("# fish was unusable at that moment it may be unusable again, so")
    print("# nudge the seconds a little if a replacement is no better.")
    for command in commands:
        print(command)
    print("\n# then review just the new ones, one video at a time:")
    for video in sorted(lost):
        print(f'python review_frames.py "{options.frames}" '
              f'--videos "{options.videos}" --only "{video}"')

    # Same thing as a file, since a long list is awkward to copy from a
    # scrolled terminal.
    script_path = os.path.join(options.frames, "refill_frames.txt")
    try:
        handle = open(script_path, "w", encoding="utf-8")
    except OSError as error:
        print(f"\n(couldn't write {script_path}: {error})")
        return
    with handle:
        handle.write("# Frames rejected during review, and how to replace them.\n")
        handle.write("# Generated by review_frames.py\n\n")
        for video in sorted(lost):
            have = len(remaining.get(video, []))
            gaps = sorted(lost[video])
            handle.write(f"# {video}: {have} left, rejected at t = "
                         + ", ".join(f"{t:.1f}" for _, t in gaps) + " s\n")
            if any(number == 1 for number, _ in gaps):
                handle.write("#   NOTE: first frame lost - baseline affected\n")
        handle.write("\n")
        for command in commands:
            handle.write(command + "\n")
    print(f"\nAlso written to: {script_path}")


if __name__ == "__main__":
    sys.exit(main())