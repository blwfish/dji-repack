"""Subprocess/path helpers used by merge.py."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# discover_clips() returns a flat list[str] of warnings mixing two kinds: a
# clip actually dropped from the result (scan error, probe failure,
# unparseable filename timestamp) and purely informational notes about a
# clip that's still included (no SRT sidecar). Every DROPPED-clip message
# contains this exact marker; every informational one doesn't.
# count_skipped() is the single place that knows this convention, so a
# caller who wants "how many clips were dropped" doesn't have to re-derive
# a substring check of their own -- if the wording of a dropped-clip
# message ever changes, only this constant needs to change with it.
_SKIPPED_MARKER = "skipped"


def count_skipped(warnings: list[str]) -> int:
    """How many of `warnings` represent a clip actually dropped from the
    result (not just an informational note about an included clip) --
    see _SKIPPED_MARKER's own docstring for the convention this relies
    on."""
    return sum(1 for w in warnings if _SKIPPED_MARKER in w)


def sweep_stale_partials(dest_dir: Path) -> tuple[list[str], list[str]]:
    """Remove any leftover `.{stem}.partial{suffix}` temp file directly
    under dest_dir -- merge_group() only ever cleans these up inside a
    caught `except RuntimeError`, so a SIGKILL, OOM kill, or native ffmpeg
    segfault mid-merge leaves one behind permanently (potentially holding
    a large truncated intermediate) with nothing to remove it otherwise.
    Unambiguous to sweep: this dot-prefixed ".partial" naming is generated
    only by this package, never a device-originated file, so there's no
    risk of this touching real source footage.

    Returns (names removed, failure messages). A failed unlink (permission
    error, file vanished mid-sweep) used to be swallowed by a bare `except
    OSError: pass` with no record anywhere -- the caller had no way to
    know cleanup silently failed for a given `.partial` file. Now it's
    folded into the second list the same way every other per-item failure
    in this package is surfaced, for the caller to fold into its own
    warnings."""
    removed = []
    failures = []
    for path in Path(dest_dir).glob(".*.partial*"):
        if path.is_file():
            try:
                path.unlink()
                removed.append(path.name)
            except OSError as e:
                failures.append(f"{path.name}: failed to remove stale partial -- {e}")
    return removed, failures


def run(cmd: list[str], warnings: list[str] | None = None, timeout: float | None = 300) -> str:
    if not cmd:
        # subprocess.run([]) raises a bare IndexError from deep inside the
        # subprocess module (it indexes args[0] for the program name)
        # before this function's own error handling ever runs -- callers
        # everywhere else in this package only catch RuntimeError.
        raise RuntimeError("run() called with an empty command list")
    # errors="replace" (not the subprocess default "strict"): stderr from
    # ffmpeg/ffprobe is diagnostic text we only ever log, never parse, so a
    # stray non-UTF-8 byte should degrade gracefully rather than raising
    # UnicodeDecodeError -- which callers can't catch via `except
    # RuntimeError`.
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
    except FileNotFoundError as e:
        # subprocess.run raises FileNotFoundError (not a RuntimeError
        # subclass) when the binary itself isn't on PATH -- every caller
        # in this package only catches RuntimeError, so a missing
        # ffmpeg/ffprobe install used to crash outright instead of
        # degrading the same way a nonzero-exit failure does. Converting
        # it here fixes every call site at once.
        raise RuntimeError(f"{cmd[0]} not found: {e}") from e
    except subprocess.TimeoutExpired as e:
        # Same conversion, same reason: a hung ffmpeg/ffprobe call (a
        # corrupt input triggering a decode stall, a stalled network-
        # mounted source) used to block the whole batch indefinitely --
        # every caller in this package only catches RuntimeError. 300s
        # default is generous enough for a real merge (stream-copy, not
        # re-encode, so I/O- not CPU-bound) while still bounding a hang.
        # Callers with a real reason to need longer can pass timeout=.
        raise RuntimeError(f"{cmd[0]} timed out after {timeout}s: {' '.join(cmd)}") from e
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    stderr = result.stderr.strip()
    if stderr and warnings is not None:
        warnings.append(f"{cmd[0]} warning: {stderr}")
    return result.stdout


# ffmpeg's concat demuxer exits 0 even when a file listed mid-concat
# failed to open or decode (deleted, corrupt, permission error) -- it
# logs this exact marker to stderr and silently continues without that
# segment's content, producing a "successful" merge that's missing part
# of its source list. Confirmed by direct reproduction (ffmpeg 8.1.1):
# a deleted middle/last input and a truncated/corrupt middle input both
# exit 0 with this marker present; only a missing FIRST input exits
# nonzero (already caught by procs.run's returncode check).
CONCAT_DEMUX_ERROR_MARKER = "Error during demuxing"


def concat_demux_error(warnings: list[str], since_index: int) -> str | None:
    """Returns the first warning entry (appended by run() from a concat-
    demuxer ffmpeg invocation) containing the mid-list demuxing-failure
    marker, or None if nothing indicates one. Callers must check this
    after every successful-exit (`run()` didn't raise) concat merge --
    returncode alone is not a reliable failure signal for this demuxer."""
    return next((w for w in warnings[since_index:] if CONCAT_DEMUX_ERROR_MARKER in w), None)


def concat_duration_shortfall(
    expected_s: float, actual_s: float, *, abs_tolerance_s: float = 2.0, rel_tolerance: float = 0.01
) -> bool:
    """True if `actual_s` (the merged output's own probed duration) is
    short of `expected_s` (the sum of the source clips' probed durations,
    minus any deliberate trims) by more than normal stream-copy rounding
    can explain.

    A backstop for CONCAT_DEMUX_ERROR_MARKER above, not a replacement for
    it: that marker is one exact stderr string confirmed against one
    ffmpeg version (8.1.1) and two failure modes, so a different ffmpeg
    build, a localized message, or a failure mode this project hasn't
    reproduced yet would silently not match it. A dropped clip changes
    the *output's own measurable duration* regardless of what ffmpeg
    printed, so this check can't go stale the way a string match can --
    it costs one extra ffprobe call per merge, which is cheap next to
    the cost of a silently-incomplete merge being renamed into place as
    if it were complete. Tolerance is generous enough (2s absolute, or 1%
    of the expected total, whichever is larger) to absorb frame-boundary
    rounding on stream-copy concat without false-positiving on it."""
    tolerance = max(abs_tolerance_s, expected_s * rel_tolerance)
    return (expected_s - actual_s) > tolerance


def quote_concat_path(p: Path) -> str:
    return "file '" + str(p.resolve()).replace("'", "'\\''") + "'"


def next_available_path(path: Path) -> Path:
    """Atomically claim a non-colliding path via O_CREAT|O_EXCL. Two
    concurrent callers computing an "available" name for the same base
    at the same moment can't both pick the same candidate here (unlike
    a check-then-act `if not path.exists()` loop), so the second's
    eventual rename() can't silently clobber the first's output.

    Returns the claimed path, which now exists as an empty placeholder
    for the caller to eventually overwrite (e.g. by renaming a completed
    temp file onto it, which is a normal atomic overwrite on POSIX). The
    caller is responsible for removing this placeholder if it gives up
    before ever producing the real output (see merge_group's own cleanup
    on a failed ffmpeg run).

    Deliberately does NOT special-case a pre-existing 0-byte candidate as
    "reclaimable": any existing file at the candidate path, even an empty
    one, is treated as a real collision and bumped to _v2/_v3/... -- an
    established safety invariant, since this function has no reliable way
    to distinguish a stale abandoned placeholder from this package's own
    prior claim from some other empty file that legitimately belongs
    there. A stale placeholder from an interrupted merge can therefore
    still permanently squat on the intended name; see
    dji_repack.procs.sweep_stale_partials for the piece of this that IS
    safe to clean up unattended (the merge's own .partial temp file, never
    a real collision target)."""
    candidate = path
    i = 2
    while True:
        try:
            fd = os.open(str(candidate), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return candidate
        except FileExistsError:
            candidate = path.with_name(f"{path.stem}_v{i}{path.suffix}")
            i += 1


def ffprobe_int(value, default: int | None = None) -> int | None:
    """Parse an ffprobe-reported int-like JSON field, tolerating the
    "N/A" sentinel ffprobe emits for fields it can't determine (e.g. a
    corrupt or unusual container)."""
    if value in (None, "N/A"):
        return default
    return int(float(value))


def ffprobe_float(value, default: float | None = None) -> float | None:
    """Same as ffprobe_int, but for float-like fields (e.g. duration)."""
    if value in (None, "N/A"):
        return default
    return float(value)
