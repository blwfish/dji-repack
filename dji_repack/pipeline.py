"""Shared merge-orchestration pipeline used by both cli.py and gui.py.

Both entry points need the exact same sequence: sweep stale partials,
dispatch each group to copy_lone_clip or merge_group (+ gap warning +
optional archive), then optionally copy stills. Previously this sequence
was duplicated independently in cli.py and gui.py with no shared function
factoring it out and no test covering either copy at all -- a future
change to one path's edge-case handling (e.g. when a failed group's clips
get archived) could silently diverge from the other with nothing in the
test suite to catch it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .archive import archive_merged_group, copy_lone_clip
from .procs import sweep_stale_partials
from .stills import copy_stills, discover_stills
from .video import ClipGroup, group_part_index_gap_warning, merge_group


@dataclass
class GroupOutcome:
    group: ClipGroup
    kind: str  # "copied_lone", "merged", or "failed"
    output_path: Path | None = None
    error: str | None = None


@dataclass
class MergePipelineResult:
    swept_partials: list[str] = field(default_factory=list)
    swept_partial_failures: list[str] = field(default_factory=list)
    group_outcomes: list[GroupOutcome] = field(default_factory=list)
    stills_copied: int = 0
    stills_skipped: int = 0

    @property
    def merged_count(self) -> int:
        return sum(1 for o in self.group_outcomes if o.kind == "merged")

    @property
    def copied_lone_count(self) -> int:
        return sum(1 for o in self.group_outcomes if o.kind == "copied_lone")

    @property
    def failed_count(self) -> int:
        return sum(1 for o in self.group_outcomes if o.kind == "failed")


def run_merge_pipeline(
    groups: list[ClipGroup],
    source_dir: Path,
    dest_dir: Path,
    *,
    do_archive: bool = True,
    do_stills: bool = True,
    log: Callable[[str, bool], None] = lambda message, stderr=False: None,
    on_group_done: Callable[[ClipGroup], None] | None = None,
    on_still_progress: Callable[[str, float], None] | None = None,
) -> MergePipelineResult:
    """Runs the sweep -> per-group dispatch -> stills sequence shared by
    the CLI and GUI entry points. `groups` is caller-provided (the CLI
    passes every group from a fresh discover+group pass; the GUI passes
    whatever subset the user selected from an earlier scan) -- this
    function has no opinion on discovery/grouping/gap-threshold itself.

    log(message, stderr) receives one line per event (swept partial,
    per-group warning/result, still-copy warning/progress) -- the CLI
    prints to stdout or stderr per the flag, the GUI routes both to its
    Tk log widget and ignores the flag. on_group_done, if given, is
    called once per group immediately after its own outcome is known
    (lone-copy, merge success, or merge failure) -- the GUI uses this to
    remove the group from its pending-groups tree incrementally; the CLI
    ignores it.
    """
    dest_dir = Path(dest_dir)
    result = MergePipelineResult()
    result.swept_partials, result.swept_partial_failures = sweep_stale_partials(dest_dir)
    for name in result.swept_partials:
        log(f"removed leftover partial: {name}", False)
    for failure in result.swept_partial_failures:
        log(f"warning: {failure}", True)

    for i, group in enumerate(groups, 1):
        if len(group.clips) < 2:
            clip = group.clips[0]
            start = time.perf_counter()
            copy_warnings = copy_lone_clip(clip, dest_dir)
            elapsed = time.perf_counter() - start
            for w in copy_warnings:
                log(f"warning: {w}", True)
            output_path = dest_dir / clip.mp4_path.name
            log(f"group {i}: single clip, copied -> {output_path} ({elapsed:.1f}s)", False)
            result.group_outcomes.append(GroupOutcome(group, "copied_lone", output_path=output_path))
            if on_group_done is not None:
                on_group_done(group)
            continue

        gap_warning = group_part_index_gap_warning(group)
        if gap_warning:
            log(f"warning: {gap_warning}", True)

        start = time.perf_counter()
        merge_result = merge_group(group, dest_dir=dest_dir)
        elapsed = time.perf_counter() - start
        for w in merge_result.warnings:
            log(f"warning: {w}", True)
        if not merge_result.ok:
            log(
                f"group {i}: FAILED -- {', '.join(merge_result.source_files)}: "
                f"{merge_result.error} ({elapsed:.1f}s)",
                True,
            )
            result.group_outcomes.append(GroupOutcome(group, "failed", error=merge_result.error))
            if on_group_done is not None:
                on_group_done(group)
            continue

        log(f"group {i}: merged -> {merge_result.output_path} ({elapsed:.1f}s)", False)
        if do_archive:
            for w in archive_merged_group(group, dest_dir):
                log(f"warning: {w}", True)
        result.group_outcomes.append(GroupOutcome(group, "merged", output_path=merge_result.output_path))
        if on_group_done is not None:
            on_group_done(group)

    if do_stills:
        stills, still_scan_warnings = discover_stills(source_dir)
        for w in still_scan_warnings:
            log(f"warning: {w}", True)
        copied, skipped, still_warnings = copy_stills(stills, dest_dir, on_progress=on_still_progress)
        result.stills_copied = copied
        result.stills_skipped = skipped
        for w in still_warnings:
            log(f"warning: {w}", True)

    return result
