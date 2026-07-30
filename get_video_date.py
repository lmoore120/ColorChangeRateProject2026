r"""
get_video_date.py
------------------
Reads the recording date and time straight out of a video file itself -
not just the "date modified" Windows shows in File Explorer, which can be
wrong if the file was copied, downloaded, texted, or synced at some point
after it was actually filmed.

WHY THIS MATTERS:
Windows' file dates reflect what happened to the FILE (when it landed on
your hard drive). Most phone videos (.mp4 / .mov) also carry their own
built-in recording timestamp, set by the camera the moment you hit record -
that's the one this script digs out.

HOW TO RUN IT:
  python get_video_date.py "path\to\your\video.mp4"

WHAT YOU GET:
  The recording date/time from inside the video file (if the file has one),
  PLUS Windows' own file dates for comparison.

NOTE: This reads .mp4 and .mov files specifically (the format iPhones,
Android phones, and most cameras use). Other formats will just show the
Windows file dates as a fallback.
"""

import datetime
import os
import re
import struct
import sys

# .mp4/.mov files are built from nested "boxes" (also called "atoms"),
# each one a chunk with a 4-byte size, a 4-character name, then its
# content. The recording date lives inside a box called "mvhd" (movie
# header), tucked inside a box called "moov". This walks that structure
# to find it - no extra programs required, just reading the file's own
# bytes directly.

# The timestamp inside "mvhd" counts seconds since January 1, 1904 (an
# old Mac convention) rather than the more common January 1, 1970 - this
# is the gap between those two dates, used to convert it.
SECONDS_1904_TO_1970 = 2082844800


def _iter_boxes(f, start, end):
    """Walk through the boxes in one section of the file, yielding
    (box_type, content_start, content_end) for each one."""
    pos = start
    while pos < end - 8:
        f.seek(pos)
        size_bytes = f.read(4)
        box_type = f.read(4).decode("latin1", errors="replace")
        if len(size_bytes) < 4:
            break
        size = struct.unpack(">I", size_bytes)[0]
        header_size = 8

        if size == 1:
            # "size == 1" means the real size is a bigger, 8-byte number
            # that follows right after
            big_size_bytes = f.read(8)
            if len(big_size_bytes) < 8:
                break
            size = struct.unpack(">Q", big_size_bytes)[0]
            header_size = 16
        elif size == 0:
            # "size == 0" is shorthand for "this box runs to the end of file"
            size = end - pos

        if size < header_size:
            break  # corrupted/unexpected data - stop rather than loop forever

        yield (box_type, pos + header_size, pos + size)
        pos += size


def _find_box(f, box_path, start, end):
    """Find a nested box by a path like ['moov', 'mvhd'] and return its
    (content_start, content_end), or None if any part of the path is
    missing."""
    current_start, current_end = start, end
    for wanted_type in box_path:
        found = None
        for box_type, content_start, content_end in _iter_boxes(f, current_start, current_end):
            if box_type == wanted_type:
                found = (content_start, content_end)
                break
        if found is None:
            return None
        current_start, current_end = found
    return current_start, current_end


def _read_mvhd_creation_time(f, mvhd_start, mvhd_end):
    """Parse the 'mvhd' box to get the recording timestamp (UTC)."""
    f.seek(mvhd_start)
    version = f.read(1)
    if not version:
        return None
    version = version[0]
    f.read(3)  # flags - not needed

    if version == 1:
        # version 1 uses 8-byte (64-bit) time fields, for very large/long files
        raw = f.read(8)
        if len(raw) < 8:
            return None
        creation_time = struct.unpack(">Q", raw)[0]
    else:
        # version 0 (the vast majority of files) uses 4-byte time fields
        raw = f.read(4)
        if len(raw) < 4:
            return None
        creation_time = struct.unpack(">I", raw)[0]

    if creation_time == 0:
        return None  # some cameras/software just don't fill this in

    unix_time = creation_time - SECONDS_1904_TO_1970
    if unix_time < 0:
        return None  # nonsensical - treat as not-set rather than guess
    return datetime.datetime.fromtimestamp(unix_time, tz=datetime.timezone.utc).replace(tzinfo=None)


def _find_apple_creationdate_string(path):
    """Newer iPhones also store a second, more precise timestamp as plain
    text (something like '2024-08-15T14:23:01-0400') which - unlike
    mvhd's timestamp - includes your timezone, not just UTC. This isn't
    stored in a single predictable spot, so rather than fully parsing
    Apple's metadata format, we scan the file's metadata area for
    anything that LOOKS like that pattern. If found, it's typically more
    useful than the plain UTC mvhd time. Returns a datetime, or None."""
    pattern = re.compile(
        rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]\d{2}:?\d{2}|Z)"
    )
    try:
        with open(path, "rb") as f:
            # This kind of metadata is normally near the start or end of
            # the file, not scattered through the (potentially huge)
            # video data itself - reading a couple MB from each end
            # finds it without having to scan the whole file.
            chunk_size = 2_000_000
            f.seek(0)
            head = f.read(chunk_size)
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(max(0, file_size - chunk_size))
            tail = f.read(chunk_size)
    except OSError:
        return None

    for chunk in (head, tail):
        match = pattern.search(chunk)
        if match:
            text = match.group().decode("ascii", errors="ignore")
            try:
                return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def get_recording_datetime(path):
    """Try to find the video's own internal recording timestamp.
    Returns a datetime, or None if the file doesn't have one (or isn't
    an mp4/mov file this parser understands)."""
    # The precise, timezone-aware Apple-style timestamp is preferred
    # when available; otherwise fall back to the plain UTC one from mvhd.
    precise = _find_apple_creationdate_string(path)
    if precise is not None:
        return precise, "precise (includes timezone)"

    try:
        with open(path, "rb") as f:
            file_size = os.fstat(f.fileno()).st_size
            mvhd = _find_box(f, ["moov", "mvhd"], 0, file_size)
            if mvhd is None:
                return None, None
            creation_time = _read_mvhd_creation_time(f, *mvhd)
            if creation_time is None:
                return None, None
            return creation_time, "UTC (no timezone info available)"
    except (OSError, struct.error):
        return None, None


def get_file_system_dates(path):
    """Windows' own file dates, as a fallback / point of comparison."""
    stat = os.stat(path)
    created = datetime.datetime.fromtimestamp(stat.st_ctime)
    modified = datetime.datetime.fromtimestamp(stat.st_mtime)
    return created, modified


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_video_date.py \"path\\to\\your\\video.mp4\"")
        return

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Couldn't find that file: {video_path}")
        return

    print(f"Reading: {video_path}\n")

    recording_time, note = get_recording_datetime(video_path)
    if recording_time is not None:
        print("RECORDING DATE/TIME (from inside the video file):")
        stamp = recording_time.strftime('%Y-%m-%d %H:%M:%S')
        if recording_time.tzinfo is not None:
            stamp += recording_time.strftime(' UTC%z')
        print(f"  {stamp}  [{note}]")
    else:
        print("RECORDING DATE/TIME: not found inside this file.")
        print("  (Either it's not an mp4/mov file, or the camera/software")
        print("   that made it didn't fill this field in.)")

    print()
    created, modified = get_file_system_dates(video_path)
    print("WINDOWS FILE DATES (for comparison - reflects the FILE, not the recording):")
    print(f"  Created:  {created.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
