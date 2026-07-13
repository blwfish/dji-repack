"""Copying still images (DNG/JPEG) alongside video consolidation.

Stills share a folder (and DJI's part-number counter) with video clips but
carry no merge/telemetry logic of their own -- discovering and copying
them is a separate, independent pass from video grouping, and lands
alongside the merged video output in the same destination folder.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path

from .constants import is_within_raw_splits

STILL_EXTENSIONS = frozenset({".dng", ".jpg", ".jpeg"})


def discover_stills(source_dir: Path) -> tuple[list[Path], list[str]]:
    """Every still-image file under source_dir, recursively, excluding
    _raw_splits/ (same convention as video.discover_clips -- that
    directory holds already-processed video sources, never scanned back
    into).

    Returns (stills sorted by path, list of warning strings) -- matching
    video.discover_clips's own (clips, warnings) return shape. Previously
    this returned only a bare list and a per-file scan error (broken
    symlink, permission denied) was swallowed by a bare `continue` with no
    warnings channel at all: unlike the structurally identical scan loop
    in discover_clips, a still image dropped here was invisible -- not
    counted, not logged, indistinguishable from "there was no such file."
    Real photo/video content silently missing with no diagnostic trail is
    exactly the failure mode this return value now closes off.
    """
    source_dir = Path(source_dir)
    stills = []
    warnings: list[str] = []
    for p in source_dir.rglob("*"):
        if is_within_raw_splits(p, source_dir):
            continue
        try:
            is_still = p.is_file() and p.suffix.lower() in STILL_EXTENSIONS
        except OSError as e:
            warnings.append(f"{p.name}: skipped during scan -- {e}")
            continue
        if is_still:
            stills.append(p)
    stills.sort()
    return stills, warnings


def copy_stills(
    stills: list[Path], dest_dir: Path, on_progress: Callable[[str, float], None] | None = None,
) -> tuple[int, int, list[str]]:
    """Copies each still into dest_dir (flat -- matches how merged video
    output and lone-clip copies already land flat in dest_dir, regardless
    of what subfolder the source came from). Never moves: stills aren't
    consumed by anything the way merged clips are, so the source is
    always left in place.

    A destination file that already exists is treated as already copied
    and skipped, not overwritten or compared -- a same-named file already
    there is presumed complete from a prior run.

    on_progress, if given, is called once per still actually copied (not
    for one skipped as already-present -- a skip isn't processing) with
    (filename, elapsed_seconds), so a caller can report progress as it
    happens instead of only the final aggregate count.

    Returns (copied_count, skipped_count, warnings)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    warnings: list[str] = []
    for still in stills:
        dest_path = dest_dir / still.name
        if dest_path.exists():
            skipped += 1
            continue
        start = time.perf_counter()
        try:
            shutil.copy2(still, dest_path)
        except OSError as e:
            warnings.append(f"{still.name}: failed to copy -- {e}")
            continue
        copied += 1
        if on_progress is not None:
            on_progress(still.name, time.perf_counter() - start)
    return copied, skipped, warnings
