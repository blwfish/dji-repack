"""Copying still images (DNG/JPEG) alongside video consolidation.

Stills share a folder (and DJI's part-number counter) with video clips but
carry no merge/telemetry logic of their own -- discovering and copying
them is a separate, independent pass from video grouping, and lands
alongside the merged video output in the same destination folder.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .constants import RAW_SPLITS_DIRNAME

STILL_EXTENSIONS = frozenset({".dng", ".jpg", ".jpeg"})


def discover_stills(source_dir: Path) -> list[Path]:
    """Every still-image file under source_dir, recursively, excluding
    _raw_splits/ (same convention as video.discover_clips -- that
    directory holds already-processed video sources, never scanned back
    into)."""
    source_dir = Path(source_dir)
    stills = []
    for p in source_dir.rglob("*"):
        if RAW_SPLITS_DIRNAME in p.relative_to(source_dir).parts[:-1]:
            continue
        try:
            is_still = p.is_file() and p.suffix.lower() in STILL_EXTENSIONS
        except OSError:
            continue
        if is_still:
            stills.append(p)
    stills.sort()
    return stills


def copy_stills(stills: list[Path], dest_dir: Path) -> tuple[int, int, list[str]]:
    """Copies each still into dest_dir (flat -- matches how merged video
    output and lone-clip copies already land flat in dest_dir, regardless
    of what subfolder the source came from). Never moves: stills aren't
    consumed by anything the way merged clips are, so the source is
    always left in place.

    A destination file that already exists is treated as already copied
    and skipped, not overwritten or compared -- a same-named file already
    there is presumed complete from a prior run.

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
        try:
            shutil.copy2(still, dest_path)
            copied += 1
        except OSError as e:
            warnings.append(f"{still.name}: failed to copy -- {e}")
    return copied, skipped, warnings
