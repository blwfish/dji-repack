"""Shared constants for dji_repack -- kept in their own module so there's
exactly one source of truth for the "already processed" archive directory
name, consumed by both merge.py (writes into it) and the discovery scans
(must never walk back into it).
"""

from __future__ import annotations

from pathlib import Path

# Leading underscore: reads as "not primary content" to a human browsing
# the staging folder.
RAW_SPLITS_DIRNAME = "_raw_splits"


def is_within_raw_splits(path: Path, source_dir: Path) -> bool:
    """True if `path` (as discovered via rglob("*") under source_dir)
    lives inside a _raw_splits/ directory anywhere in its ancestry.

    Single source of truth for the check both video.discover_clips and
    stills.discover_stills must apply identically -- previously each
    module carried its own verbatim copy of this exact expression with no
    shared helper and no parity test, so the two scans could silently
    diverge on what counts as "already processed" if only one copy were
    ever updated (e.g. to handle case-insensitivity or a second archive
    directory name).
    """
    return RAW_SPLITS_DIRNAME in path.relative_to(source_dir).parts[:-1]
