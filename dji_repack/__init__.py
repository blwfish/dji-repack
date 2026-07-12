"""dji_repack -- reassembles DJI Air3 drone footage that got split into
multiple sequential MP4 files (typically a FAT32 4GB file-size limit) back
into one logical recording per session, with .SRT flight telemetry merged
back in as an embedded subtitle track.

Extracted from mneme's core/repack/ so it can run standalone, without the
rest of mneme, while still using Lightroom day to day.
"""

from .archive import archive_merged_group, copy_lone_clip
from .constants import RAW_SPLITS_DIRNAME
from .srt_parser import SrtCue, SrtParseError, format_cue_block, parse_srt
from .stills import STILL_EXTENSIONS, copy_stills, discover_stills
from .video import (
    GAP_THRESHOLD_DEFAULT_S,
    Clip,
    ClipGroup,
    ClipProbe,
    MergeResult,
    discover_clips,
    group_clips,
    group_part_index_gap_warning,
    group_summary,
    merge_group,
)

__all__ = [
    "discover_clips",
    "group_clips",
    "group_part_index_gap_warning",
    "merge_group",
    "group_summary",
    "archive_merged_group",
    "copy_lone_clip",
    "discover_stills",
    "copy_stills",
    "STILL_EXTENSIONS",
    "Clip",
    "ClipGroup",
    "ClipProbe",
    "MergeResult",
    "GAP_THRESHOLD_DEFAULT_S",
    "RAW_SPLITS_DIRNAME",
    "parse_srt",
    "format_cue_block",
    "SrtCue",
    "SrtParseError",
]
