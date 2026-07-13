"""Command-line interface for dji_repack: `dji-repack scan|merge <dir>`."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .pipeline import run_merge_pipeline
from .stills import discover_stills
from .video import GAP_THRESHOLD_DEFAULT_S, discover_clips, group_clips, group_part_index_gap_warning, group_summary


def _check_ffmpeg() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        print(f"error: {', '.join(missing)} not found on PATH -- install ffmpeg first", file=sys.stderr)
        sys.exit(1)


def _cmd_scan(args: argparse.Namespace) -> int:
    _check_ffmpeg()
    source_dir = Path(args.source_dir)
    clips, warnings = discover_clips(source_dir)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    groups = group_clips(clips, gap_threshold_s=args.gap_threshold)
    if not groups:
        print("no video clips found")
    else:
        for i, group in enumerate(groups, 1):
            summary = group_summary(group)
            kind = "multi-clip, will merge" if summary["clip_count"] > 1 else "single clip, will be copied as-is"
            print(f"group {i}: {summary['clip_count']} clip(s), {kind}")
            size_note = " (some sizes unavailable)" if summary["size_unavailable"] else ""
            print(f"  start: {summary['start_dt']}  duration: {summary['total_duration_s']:.1f}s"
                  f"  size: {summary['total_size_bytes'] / 1e6:.1f} MB{size_note}")
            for name in summary["clip_names"]:
                print(f"    {name}")
            if summary["missing_srt"]:
                print(f"  missing SRT: {', '.join(summary['missing_srt'])}")
            gap_warning = group_part_index_gap_warning(group)
            if gap_warning:
                print(f"  warning: {gap_warning}")

    if not args.no_stills:
        stills, still_warnings = discover_stills(source_dir)
        for w in still_warnings:
            print(f"warning: {w}", file=sys.stderr)
        print(f"{len(stills)} still image(s) found (.dng/.jpg/.jpeg)")

    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    _check_ffmpeg()
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest) if args.dest else source_dir

    clips, warnings = discover_clips(source_dir)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    if not clips:
        print("no video clips found")
        groups = []
    else:
        groups = group_clips(clips, gap_threshold_s=args.gap_threshold)

    def log(message: str, stderr: bool) -> None:
        print(message, file=sys.stderr if stderr else sys.stdout)

    result = run_merge_pipeline(
        groups, source_dir, dest_dir,
        do_archive=not args.no_archive,
        do_stills=not args.no_stills,
        log=log,
        on_still_progress=lambda name, elapsed: print(f"  {name}: copied ({elapsed:.2f}s)"),
    )

    summary = (
        f"done: {result.merged_count} merged, {result.copied_lone_count} single clip(s) copied, "
        f"{result.failed_count} failed"
    )
    if not args.no_stills:
        summary += f", {result.stills_copied} still(s) copied ({result.stills_skipped} already present)"
    print(summary)
    return 1 if result.failed_count else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dji-repack", description="Reassemble split DJI Air3 video clips.")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Discover and group clips without merging (dry run)")
    scan.add_argument("source_dir")
    scan.add_argument(
        "--gap-threshold", type=float, default=GAP_THRESHOLD_DEFAULT_S,
        help=f"seconds between clips to still count as the same recording (default {GAP_THRESHOLD_DEFAULT_S:.0f})",
    )
    scan.add_argument(
        "--no-stills", action="store_true",
        help="don't report on still images (.dng/.jpg/.jpeg) found in source_dir",
    )
    scan.set_defaults(func=_cmd_scan)

    merge = sub.add_parser("merge", help="Discover, group, merge, and archive split clips")
    merge.add_argument("source_dir")
    merge.add_argument("--dest", help="output directory (default: same as source_dir)")
    merge.add_argument(
        "--gap-threshold", type=float, default=GAP_THRESHOLD_DEFAULT_S,
        help=f"seconds between clips to still count as the same recording (default {GAP_THRESHOLD_DEFAULT_S:.0f})",
    )
    merge.add_argument(
        "--no-archive", action="store_true",
        help="leave merged source clips in place instead of moving them to _raw_splits/",
    )
    merge.add_argument(
        "--no-stills", action="store_true",
        help="don't copy still images (.dng/.jpg/.jpeg) from source_dir to --dest",
    )
    merge.set_defaults(func=_cmd_merge)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
