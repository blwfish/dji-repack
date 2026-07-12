"""Archiving successfully-merged source clips out of the way.

Logic lifted from mneme's core/card_ingest/pipeline.py (_merge_video_groups),
decoupled from Corpus/photos_dir -- here it just takes a plain dest_dir.
"""

from __future__ import annotations

from pathlib import Path

from .constants import RAW_SPLITS_DIRNAME
from .video import ClipGroup


def archive_merged_group(group: ClipGroup, dest_dir: Path) -> list[str]:
    """Moves every clip (and its .SRT sidecar, if any) in a successfully-
    merged group into dest_dir/_raw_splits/, so a rerun doesn't rediscover
    and re-merge already-processed footage.

    Not atomic across the group -- an interruption partway through can
    leave some clips archived and one or more stranded in the source dir,
    indistinguishable on the next run from unprocessed footage (their
    content is already safely in the merged output, though, so this is a
    discoverability gap, not a data-loss one). Each rename is attempted
    independently and failures are collected as warning strings rather
    than aborting, so one bad rename can't also strand every clip after
    it in the same group.
    """
    raw_splits_dir = Path(dest_dir) / RAW_SPLITS_DIRNAME
    raw_splits_dir.mkdir(parents=True, exist_ok=True)

    warnings = []
    failures = []
    for clip in group.clips:
        try:
            clip.mp4_path.rename(raw_splits_dir / clip.mp4_path.name)
            if clip.srt_path is not None and clip.srt_path.exists():
                clip.srt_path.rename(raw_splits_dir / clip.srt_path.name)
        except OSError as e:
            failures.append(f"{clip.mp4_path.name}: {e}")
    if failures:
        warnings.append(
            f"archive_incomplete: {len(failures)} clip(s) from a successfully-merged "
            f"group could not be moved to {RAW_SPLITS_DIRNAME}/ and remain in the "
            f"source folder -- a future rerun may rediscover them as a stray "
            f"single-clip group ({'; '.join(failures)})"
        )
    return warnings
