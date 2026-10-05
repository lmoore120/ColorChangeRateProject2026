"""Read the ORIGINAL recording time out of a video file.

Not the file's modified date - that changes when a file is copied off a
camera or moved between drives, so it says when you downloaded the video,
not when it was shot. The real recording time lives inside the video
container itself.

Two ways of getting it, tried in order:
  1. The QuickTime/MP4 'mvhd' atom, parsed directly. Works for .MOV and
     .MP4 with no external programs installed, which matters because this
     has to run on a lab Windows machine.
  2. ffprobe, if it happens to be installed - it handles more formats and
     also reads the creation_time tag some cameras write instead.

If neither works you get None, and the caller records the file's modified
date with a flag saying it's a fallback, rather than quietly passing off a
download date as a recording date.
"""

import datetime
import json
import os
import struct
import subprocess

# QuickTime counts seconds from 1904-01-01, not the usual 1970.
QUICKTIME_EPOCH = datetime.datetime(1904, 1, 1)

# Atoms that contain other atoms and so are worth descending into.
CONTAINER_ATOMS = (b"moov", b"trak", b"mdia")


def _read_mvhd_time(path):
    """Pull creation_time out of the movie header atom."""
    try:
        total = os.path.getsize(path)
        with open(path, "rb") as handle:

            def find(parent_end, wanted):
                while handle.tell() < parent_end - 8:
                    start = handle.tell()
                    header = handle.read(8)
                    if len(header) < 8:
                        return None
                    size, kind = struct.unpack(">I4s", header)
                    if size == 1:                    # 64-bit size follows
                        size = struct.unpack(">Q", handle.read(8))[0]
                    if size < 8:
                        return None
                    end = start + size
                    if kind == wanted:
                        return start
                    if kind in CONTAINER_ATOMS:
                        found = find(end, wanted)
                        if found is not None:
                            return found
                    handle.seek(end)
                return None

            location = find(total, b"mvhd")
            if location is None:
                return None
            handle.seek(location + 8)
            version = handle.read(1)[0]
            handle.read(3)                            # flags
            if version == 1:
                seconds = struct.unpack(">Q", handle.read(8))[0]
            else:
                seconds = struct.unpack(">I", handle.read(4))[0]
    except Exception:
        return None

    if not seconds:
        return None
    try:
        stamp = QUICKTIME_EPOCH + datetime.timedelta(seconds=seconds)
    except OverflowError:
        return None
    # Sanity-check it: a camera clock that was never set produces dates in
    # 1904 or far in the future, which are worse than no answer at all.
    if not (datetime.datetime(2000, 1, 1) < stamp < datetime.datetime(2100, 1, 1)):
        return None
    return stamp


def _read_ffprobe_time(path):
    try:
        output = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_entries", "format_tags=creation_time", path],
            capture_output=True, text=True, timeout=20)
        tags = json.loads(output.stdout or "{}").get("format", {}).get("tags", {})
        text = tags.get("creation_time")
        if not text:
            return None
        return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def video_recorded_at(path):
    """Return (datetime, source) where source says where it came from:
    'mvhd', 'ffprobe', or 'file_modified' when the real one couldn't be
    found. Never returns None for the datetime, so a row always has
    something - but the source column tells you whether to trust it."""
    for reader, name in ((_read_mvhd_time, "mvhd"), (_read_ffprobe_time, "ffprobe")):
        stamp = reader(path)
        if stamp is not None:
            return stamp, name
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)), "file_modified"


if __name__ == "__main__":
    import sys
    for argument in sys.argv[1:]:
        when, source = video_recorded_at(argument)
        print(f"{os.path.basename(argument)}: {when:%Y-%m-%d %H:%M:%S}  (from {source})")