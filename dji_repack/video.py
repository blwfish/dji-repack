"""
Discovery, grouping, and ffmpeg merge logic for DJI Air3 video ingest.

Pipeline:
  1. discover_clips()  -- find every *.MP4 under source_dir, pair with its
     .SRT sidecar (if present), parse telemetry, sort by wall-clock start.
  2. group_clips()     -- collapse clips into recording sessions using a
     single gap threshold: gap < threshold => same output file (whether
     that gap is ~50ms from a file-size-triggered chunk split, or minutes
     from a stills break mid-flight); gap >= threshold => new output file.
  3. merge_group()     -- ffmpeg concat (stream copy, no re-encode) + merged
     telemetry muxed in as an embedded mov_text subtitle track + standard
     creation_time/location container metadata.

Ported from mneme's core/repack/video.py, itself ported from
robo-classifier/air3_ingest. DJI-Air3-specific in several places (SRT
format, filename timestamp regex, one global gap threshold). Deliberately
not generalized to other drones/cameras here.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from . import procs
from .constants import RAW_SPLITS_DIRNAME
from .srt_parser import SrtCue, SrtParseError, format_cue_block, parse_srt

FFPROBE = "ffprobe"
FFMPEG = "ffmpeg"

# Thin aliases, kept under their original names since tests reference them
# as `_run`/etc.
_run = procs.run
_quote_concat_path = procs.quote_concat_path
_next_available_path = procs.next_available_path

GAP_THRESHOLD_DEFAULT_S = 300.0

FILENAME_TS_RE = re.compile(r"DJI_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_")

# Leading digit run in the filename remainder after the embedded timestamp
# (e.g. "0001" in "DJI_20260629142707_0001_D.MP4") -- used ONLY as a
# supplementary gap-detection signal in group_part_index_gap_warning()
# below, never to gate the merge decision itself: DJI's exact
# sequence/lens-suffix filename schema is unverified against real multi-
# camera/multi-lens footage, so it isn't trusted as a hard grouping rule.
_PART_INDEX_RE = re.compile(r"^(\d+)")


@dataclass
class ClipProbe:
    """ffprobe field disposition (stream + format sections), so nothing is
    silently missed from a hand-picked allow-list:

      extracted:  codec_name, width, height, r_frame_rate, pix_fmt, rotation
                  (side_data Display Matrix, falling back to the legacy
                  `tags.rotate` string), duration (prefers the video
                  stream's own reported duration over the container-level
                  `format.duration`, which is sometimes rounded/padded
                  differently -- still an approximation for VFR footage,
                  not a frame-accurate fix), creation_time (format tag),
                  container_location (format tag, GPS fallback when a clip
                  has no SRT sidecar)
      raw-only:   bit_rate, nb_frames -- useful diagnostics for spotting a
                  truncated/corrupt chunk, not currently acted on by any
                  merge decision
      dropped-with-reason: codec_long_name/profile/level/color_*/
                  chroma_location/field_order/refs/nal_length_size/
                  sample_aspect_ratio/display_aspect_ratio/time_base/
                  start_pts/start_time/format_name/probe_score/disposition
                  -- generic container/codec bookkeeping not needed for
                  either the uniform-stream merge-safety check or
                  telemetry; ffmpeg's `-c:v copy` preserves the underlying
                  bitstream regardless of what ffprobe reports here, so
                  there's nothing this tool would act on by capturing them.
                  Also dropped, same reason (no current merge/telemetry
                  consumer): format.tags' Apple/DJI-specific keys
                  (com.apple.quicktime.make/model/software, major_brand,
                  minor_version, compatible_brands, encoder); format.size/
                  start_time/nb_streams; stream coded_width/coded_height
                  (distinct from display width/height, relevant only for
                  anamorphic/padded encodes, none observed on Air3
                  footage); avg_frame_rate (distinct from r_frame_rate for
                  VFR footage -- not distinguished from it here since Air3
                  footage is CFR); side_data_list entries other than
                  rotation (e.g. HDR mastering-display/content-light-level
                  metadata -- Air3 footage is SDR); stream-level tags
                  (handler_name/language/timecode/encoder on the video
                  stream itself, distinct from the format-level tags
                  already captured); extradata/bits_per_raw_sample. None
                  of these were evaluated against real HDR/non-DJI camera
                  footage -- if this tool is ever pointed at footage from
                  a different camera, this list should be re-examined
                  before assuming it's still exhaustive.
    """
    duration_s: float
    codec_name: str
    width: int
    height: int
    r_frame_rate: str
    pix_fmt: str
    rotation: int  # degrees; 0 if the clip carries no rotation side data/tag
    bit_rate: int | None
    nb_frames: int | None
    creation_time: str | None  # container tag, e.g. "2026-06-29T14:27:07.000000Z"
    container_location: str | None  # raw ISO6709-ish string from format tags
    audio_codec_name: str | None = None  # None means the clip has no audio stream at all
    audio_sample_rate: int | None = None  # None means the clip has no audio stream at all
    audio_channels: int | None = None  # None means the clip has no audio stream at all

    @property
    def has_audio(self) -> bool:
        return self.audio_codec_name is not None


@dataclass
class Clip:
    mp4_path: Path
    srt_path: Path | None
    cues: list[SrtCue]
    srt_error: str | None
    probe: ClipProbe
    start_dt: datetime
    end_dt: datetime
    start_is_estimated: bool  # True when there was no usable SRT and we fell
                               # back to filename timestamp + ffprobe duration
    filename_suffix: str | None = None  # raw text after the embedded
        # timestamp (e.g. "0001_D"), only ever populated on the filename-
        # estimated-start path, where it's used purely as a deterministic
        # secondary sort key for same-second ties -- not parsed further,
        # since DJI's sequence/lens-suffix filename schema isn't
        # independently verified.
    part_index: int | None = None  # leading digit run of filename_suffix's
        # raw source (see _PART_INDEX_RE), always attempted regardless of
        # whether SRT telemetry was available -- unlike filename_suffix,
        # which is only populated on the cueless path. None means the
        # filename didn't match FILENAME_TS_RE, or its remainder didn't
        # start with digits. Supplementary signal only (see
        # group_part_index_gap_warning) -- never used for grouping/sorting.


@dataclass
class ClipGroup:
    clips: list[Clip]
    gap_to_next_s: float | None = None


@dataclass
class MergeResult:
    ok: bool
    output_path: Path | None
    source_files: list[str]
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def _required_stream_field(stream: dict, key: str, mp4_path: Path):
    value = stream.get(key)
    # "N/A" is ffprobe's own sentinel for a field it can't determine (see
    # procs.ffprobe_int/ffprobe_float's docstring) -- not just absent from
    # the JSON entirely. Without this check, int(_required_stream_field(...))
    # on width/height raises an uncaught ValueError instead of this
    # function's own RuntimeError, and discover_clips only catches
    # RuntimeError -- an "N/A" width/height would abort discovery of the
    # entire source directory instead of skip-and-warn on the one bad clip.
    if value is None or value == "N/A":
        raise RuntimeError(
            f"{mp4_path.name}: ffprobe stream is missing '{key}' (corrupt or unusual container?)"
        )
    return value


def _stream_rotation(stream: dict) -> int:
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return int(side_data["rotation"])
    rotate_tag = (stream.get("tags") or {}).get("rotate")
    if rotate_tag is not None:
        return int(rotate_tag)
    return 0


def probe_clip(mp4_path: Path, warnings: list[str] | None = None) -> ClipProbe:
    """warnings is opt-in diagnostics collection, matching procs.run's own
    default -- a caller that only wants the probe result (most call sites
    in this test suite; a one-off inspection) isn't required to pass a
    list it'll never read. discover_clips/merge_group, this module's own
    internal callers that DO want ffprobe stderr surfaced to the user,
    pass a real list. If you're adding a new caller and want to know
    about non-fatal ffprobe stderr output, pass a list explicitly --
    there is no fallback logging path if you don't."""
    out = _run([
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_streams", "-show_format",
        "-of", "json",
        str(mp4_path),
    ], warnings)
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"{mp4_path.name}: ffprobe found no video stream (corrupt or non-video file?)")
    stream = streams[0]
    fmt = data.get("format", {})
    tags = fmt.get("tags", {})
    if "duration" not in fmt:
        raise RuntimeError(f"{mp4_path.name}: ffprobe reported no container duration")

    stream_duration = stream.get("duration")
    duration_s = procs.ffprobe_float(stream_duration, default=None)
    if duration_s is None:
        duration_s = float(fmt["duration"])

    audio_codec_name, audio_sample_rate, audio_channels = _probe_audio_stream(mp4_path, warnings)

    return ClipProbe(
        duration_s=duration_s,
        codec_name=_required_stream_field(stream, "codec_name", mp4_path),
        width=int(_required_stream_field(stream, "width", mp4_path)),
        height=int(_required_stream_field(stream, "height", mp4_path)),
        r_frame_rate=_required_stream_field(stream, "r_frame_rate", mp4_path),
        pix_fmt=_required_stream_field(stream, "pix_fmt", mp4_path),
        rotation=_stream_rotation(stream),
        bit_rate=procs.ffprobe_int(stream.get("bit_rate") or fmt.get("bit_rate")),
        nb_frames=procs.ffprobe_int(stream.get("nb_frames")),
        creation_time=tags.get("creation_time"),
        container_location=tags.get("location") or tags.get("com.apple.quicktime.location.ISO6709"),
        audio_codec_name=audio_codec_name,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
    )


def _probe_audio_stream(
    mp4_path: Path, warnings: list[str] | None
) -> tuple[str | None, int | None, int | None]:
    """Separate, lightweight ffprobe call for the clip's first audio
    stream, if any. DJI Air3 clips can carry a camera-mic audio track --
    always silently discarded on every merge if not probed and preserved.
    All-None means the clip genuinely has no audio stream (common
    depending on camera settings), not a probe failure -- probe_clip()
    already requires a usable video stream and raises if that's missing;
    audio absence on its own is not an error condition here.

    Field disposition, unlike ClipProbe's own exhaustive inventory above:
      extracted:  codec_name, sample_rate, channels -- the three fields
                  merge_group's -c:a copy / -map 0:a:0 decision path
                  actually needs (matching audio presence/format across
                  every clip in a group before allowing the merge).
      dropped-with-reason: tags.language/handler_name, bits_per_sample,
                  channel_layout, duration, bit_rate -- no current
                  consumer needs the camera-mic track's own metadata
                  beyond "does it exist, and is it the same shape as the
                  other clips in this group." If a future feature needs
                  any of these (e.g. surfacing the mic's own duration
                  where it diverges from video), add it here rather than
                  a second ffprobe call.
    """
    out = _run([
        FFPROBE, "-v", "error",
        "-select_streams", "a:0",
        "-show_streams",
        "-of", "json",
        str(mp4_path),
    ], warnings)
    streams = json.loads(out).get("streams") or []
    if not streams:
        return None, None, None
    stream = streams[0]
    return (
        stream.get("codec_name"),
        procs.ffprobe_int(stream.get("sample_rate")),
        procs.ffprobe_int(stream.get("channels")),
    )


def _filename_start(mp4_path: Path) -> tuple[datetime, str]:
    """Returns (embedded start timestamp, raw filename remainder after the
    timestamp, e.g. "0001_D")."""
    name = mp4_path.name
    m = FILENAME_TS_RE.search(name)
    if not m:
        raise ValueError(f"can't parse start time from filename: {name}")
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    start_dt = datetime(y, mo, d, h, mi, s)
    suffix = Path(name).stem[m.end():]
    return start_dt, suffix


def _filename_part_index(mp4_path: Path) -> int | None:
    """Best-effort part-sequence number from the filename, independent of
    whether SRT telemetry is available (unlike filename_suffix). None if
    FILENAME_TS_RE doesn't match at all, or the remainder after the
    timestamp doesn't start with digits -- both silently skip the gap-
    detection warning below rather than guessing."""
    m = FILENAME_TS_RE.search(mp4_path.name)
    if not m:
        return None
    remainder = Path(mp4_path.name).stem[m.end():]
    idx_m = _PART_INDEX_RE.match(remainder)
    return int(idx_m.group(1)) if idx_m else None


def discover_clips(source_dir: Path) -> tuple[list[Clip], list[str]]:
    """Returns (clips sorted by wall-clock start time, list of warning strings)."""
    warnings: list[str] = []
    mp4_paths = []
    for p in Path(source_dir).rglob("*"):
        # _raw_splits/ holds source clips a *previous* run already merged
        # and archived -- rediscovering them here would re-merge already-
        # processed footage into a duplicate output on every rerun (e.g.
        # after a crash mid-import on a reused staging dir).
        if RAW_SPLITS_DIRNAME in p.relative_to(source_dir).parts[:-1]:
            continue
        try:
            is_mp4 = p.is_file() and p.suffix.lower() == ".mp4"
        except OSError as e:
            # A broken symlink or permission error mid-scan must not
            # abort discovery of the entire source directory -- same
            # discipline as the per-clip probe failures caught below.
            warnings.append(f"{p.name}: skipped during scan -- {e}")
            continue
        if is_mp4:
            mp4_paths.append(p)
    mp4_paths.sort()
    clips: list[Clip] = []
    for mp4_path in mp4_paths:
        try:
            probe = probe_clip(mp4_path, warnings)
        except RuntimeError as e:
            # One corrupt/zero-byte clip anywhere on the card shouldn't
            # abort discovery of the entire source directory -- record it
            # as a per-item failure and keep going.
            warnings.append(f"{mp4_path.name}: skipped -- ffprobe failed: {e}")
            continue

        srt_path = mp4_path.with_suffix(".SRT")
        if not srt_path.exists():
            srt_path = mp4_path.with_suffix(".srt")
        srt_exists = srt_path.exists()

        cues: list[SrtCue] = []
        srt_error = None
        if srt_exists:
            try:
                cues = parse_srt(srt_path)
            except SrtParseError as e:
                srt_error = str(e)
                warnings.append(f"{mp4_path.name}: failed to parse SRT telemetry: {e}")
        else:
            srt_error = "no .SRT sidecar found"
            warnings.append(f"{mp4_path.name}: no .SRT sidecar found; grouping and "
                             f"metadata for this clip fall back to filename timestamp only")

        start_is_estimated = not bool(cues)
        filename_suffix = None
        if cues:
            start_dt = cues[0].wall_clock
            end_dt = cues[-1].wall_clock
        else:
            try:
                start_dt, filename_suffix = _filename_start(mp4_path)
            except ValueError as e:
                warnings.append(f"{mp4_path.name}: skipped -- {e}")
                continue
            end_dt = start_dt + timedelta(seconds=probe.duration_s)

        clips.append(Clip(
            mp4_path=mp4_path,
            srt_path=srt_path if srt_exists else None,
            cues=cues,
            srt_error=srt_error,
            probe=probe,
            start_dt=start_dt,
            end_dt=end_dt,
            start_is_estimated=start_is_estimated,
            filename_suffix=filename_suffix,
            part_index=_filename_part_index(mp4_path),
        ))

    clips.sort(key=lambda c: (c.start_dt, c.filename_suffix or ""))
    return clips, warnings


def group_clips(clips: list[Clip], gap_threshold_s: float = GAP_THRESHOLD_DEFAULT_S) -> list[ClipGroup]:
    if not clips:
        return []
    buckets: list[list[Clip]] = [[clips[0]]]
    for prev, cur in zip(clips, clips[1:]):
        gap = (cur.start_dt - prev.end_dt).total_seconds()
        if gap < gap_threshold_s:
            buckets[-1].append(cur)
        else:
            buckets.append([cur])
    groups = [ClipGroup(clips=b) for b in buckets]
    for i in range(len(groups) - 1):
        gap = (groups[i + 1].clips[0].start_dt - groups[i].clips[-1].end_dt).total_seconds()
        groups[i].gap_to_next_s = gap
    return groups


def group_part_index_gap_warning(group: ClipGroup) -> str | None:
    """Detects a missing-middle-segment merge: group_clips() groups purely
    by wall-clock gap, with no part-count/sequence check at all -- a
    middle segment that failed to copy (corrupt sector, dropped file)
    whose neighbors still fall within the gap threshold merges into a
    "complete-looking" file with a chunk silently gone. Returns a warning
    string naming the jump if two clips with parseable filename part
    indices (see Clip.part_index) aren't consecutive; None if there's
    nothing to report (fewer than 2 parseable indices, or all
    consecutive).

    Deliberately advisory, not a merge-blocking check: part_index is a
    best-effort filename read this tool doesn't trust as a verified hard
    contract (see _PART_INDEX_RE's docstring) -- surfacing the gap for a
    human to double-check is safer than silently guessing whether to
    refuse the merge.
    """
    indexed = [(c, c.part_index) for c in group.clips if c.part_index is not None]
    for (prev_clip, prev_idx), (cur_clip, cur_idx) in zip(indexed, indexed[1:]):
        if cur_idx != prev_idx + 1:
            return (
                f"{prev_clip.mp4_path.name} (part {prev_idx}) -> {cur_clip.mp4_path.name} "
                f"(part {cur_idx}): non-consecutive part numbers within one merged group -- "
                f"a segment may be missing from the middle of this recording"
            )
    return None


def _first_gps_cue(clips: list[Clip]) -> SrtCue | None:
    for c in clips:
        for cue in c.cues:
            if cue.has_gps:
                return cue
    return None


def _last_gps_cue(clips: list[Clip]) -> SrtCue | None:
    for c in reversed(clips):
        for cue in reversed(c.cues):
            if cue.has_gps:
                return cue
    return None


def group_summary(group: ClipGroup) -> dict:
    clips = group.clips
    first, last = clips[0], clips[-1]
    total_duration = sum(c.probe.duration_s for c in clips)
    total_size = sum(c.mp4_path.stat().st_size for c in clips)
    start_cue = _first_gps_cue(clips)
    end_cue = _last_gps_cue(clips)
    start_loc = (start_cue.latitude, start_cue.longitude) if start_cue else None
    end_loc = (end_cue.latitude, end_cue.longitude) if end_cue else None
    return {
        "clip_count": len(clips),
        "clip_names": [c.mp4_path.name for c in clips],
        # Group identity for the scan->select->process round trip: full
        # paths, not names. DJI cameras paginate onto multiple *MEDIA
        # folders and can restart file numbering per folder, so two clips
        # in different subfolders can share a filename -- name-only
        # identity would silently alias one selected group onto another.
        "clip_paths": [str(c.mp4_path) for c in clips],
        "start_dt": first.start_dt.isoformat(),
        "end_dt": last.end_dt.isoformat(),
        "total_duration_s": total_duration,
        "total_size_bytes": total_size,
        "start_location": start_loc,
        "end_location": end_loc,
        "start_is_estimated": first.start_is_estimated,
        "missing_srt": [c.mp4_path.name for c in clips if c.srt_error],
        "gap_to_next_s": group.gap_to_next_s,
    }


def _check_uniform_stream(clips: list[Clip]) -> str | None:
    def key(probe: ClipProbe) -> tuple:
        return (probe.codec_name, probe.width, probe.height, probe.r_frame_rate,
                probe.pix_fmt, probe.rotation,
                # Audio uniformity too: a mismatched audio track between
                # clips (or audio present on some clips but not others)
                # would silently break the concat-demuxer's single `0:a:0`
                # mapping merge_group uses -- refuse the same way a video
                # stream mismatch is refused, rather than let ffmpeg fail
                # confusingly (or silently pick one clip's audio format).
                probe.audio_codec_name, probe.audio_sample_rate, probe.audio_channels)

    def describe(probe: ClipProbe) -> str:
        audio = (
            f", audio {probe.audio_codec_name} {probe.audio_sample_rate}Hz {probe.audio_channels}ch"
            if probe.has_audio else ", no audio"
        )
        return (f"{probe.codec_name} {probe.width}x{probe.height}@{probe.r_frame_rate} "
                f"{probe.pix_fmt} rotate={probe.rotation}{audio}")

    first = clips[0].probe
    for c in clips[1:]:
        if key(c.probe) != key(first):
            return (
                f"video stream mismatch within group: {clips[0].mp4_path.name} is "
                f"{describe(first)}, but {c.mp4_path.name} is {describe(c.probe)}. "
                f"Refusing to concat mismatched streams."
            )
    return None


def _iso6709(lat: float, lon: float, alt: float) -> str:
    return f"{lat:+.4f}{lon:+.4f}{alt:+.3f}/"


def _build_merged_telemetry(clips: list[Clip]) -> tuple[str | None, list[str]]:
    """Builds merged-SRT text for a clip group.

    Returns (srt_text, extra_warnings). srt_text is None when no clip in
    the group has any parsed cues (nothing to mux). Any cueless clip
    within an otherwise-telemetry-bearing group contributes zero cue
    blocks but cumulative_offset still advances by its probed duration
    (so later clips' cues stay in sync) -- which means the merged
    subtitle track has a real time gap over that clip's span. That used
    to be discoverable only indirectly via the generic "no .SRT sidecar"
    warning; this makes the actual consequence (a gap, of this duration,
    at this point in the merged file) explicit.

    Pulled out as a pure function (no ffmpeg/ffprobe calls) so this exact
    logic -- the source of a real silent-gap bug -- can be unit tested
    directly instead of only being reachable through the full merge_group
    subprocess pipeline.
    """
    if not any(c.cues for c in clips):
        return None, []
    warnings: list[str] = []
    blocks: list[str] = []
    cumulative_offset = 0.0
    cue_index = 1
    for c in clips:
        if not c.cues:
            warnings.append(
                f"{c.mp4_path.name}: no telemetry parsed for this clip -- merged "
                f"subtitle track will have a ~{c.probe.duration_s:.1f}s gap here"
            )
        for cue in c.cues:
            blocks.append(format_cue_block(
                cue_index,
                cue.cue_start_s + cumulative_offset,
                cue.cue_end_s + cumulative_offset,
                cue,
            ))
            cue_index += 1
        cumulative_offset += c.probe.duration_s
    return "\n".join(blocks), warnings


def merge_group(group: ClipGroup, dest_dir: Path) -> MergeResult:
    clips = group.clips
    source_names = [c.mp4_path.name for c in clips]
    warnings = [f"{c.mp4_path.name}: {c.srt_error}" for c in clips if c.srt_error]

    mismatch = _check_uniform_stream(clips)
    if mismatch:
        return MergeResult(ok=False, output_path=None, source_files=source_names, error=mismatch)

    first = clips[0]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = _next_available_path(dest_dir / f"AIR3_{first.start_dt.strftime('%Y%m%d_%H%M%S')}.mp4")
    # ffmpeg writes to a hidden .partial name and we rename on success only,
    # so a mid-run failure (disk full, killed, decode error) never leaves a
    # broken file sitting at the real output name -- indistinguishable from
    # a real output to Lightroom or a plain directory listing.
    # Must keep the real extension (".mp4") -- not just append ".partial" --
    # because ffmpeg's output muxer is selected from the filename
    # extension; a name ending in ".partial" makes it fail immediately
    # with "Unable to choose an output format" before ever writing bytes.
    tmp_out_path = out_path.with_name(f".{out_path.stem}.partial{out_path.suffix}")

    merged_srt_text, telemetry_warnings = _build_merged_telemetry(clips)
    warnings += telemetry_warnings

    with tempfile.TemporaryDirectory(prefix="air3_ingest_") as tmp:
        tmp_path = Path(tmp)
        concat_list = tmp_path / "concat.txt"
        concat_list.write_text("\n".join(_quote_concat_path(c.mp4_path) for c in clips) + "\n")

        # -hide_banner -loglevel warning: without this, ffmpeg's stderr
        # carries its full version banner + per-frame progress line even
        # on success, which would drown out genuine non-fatal warnings
        # (e.g. timestamp discontinuities) in the warnings list surfaced
        # to the caller.
        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "warning",
               "-f", "concat", "-safe", "0", "-i", str(concat_list)]

        if merged_srt_text is not None:
            merged_srt = tmp_path / "merged.srt"
            merged_srt.write_text(merged_srt_text)
            cmd += ["-i", str(merged_srt)]

        # The concat demuxer treats the whole concatenated sequence as one
        # input (index 0); mapping 0:a:0 picks up each clip's audio track
        # continuously across the concat boundary the same way 0:v:0
        # already does for video -- no second audio-specific concat step
        # needed. _check_uniform_stream() already refused the merge if
        # audio presence/format isn't identical across every clip in the
        # group, so it's safe to map it whenever the first clip has it.
        cmd += ["-map", "0:v:0"]
        if first.probe.has_audio:
            cmd += ["-map", "0:a:0"]
        if merged_srt_text is not None:
            cmd += ["-map", "1:0"]

        cmd += ["-c:v", "copy"]
        if first.probe.has_audio:
            cmd += ["-c:a", "copy"]
        if merged_srt_text is not None:
            cmd += ["-c:s", "mov_text", "-metadata:s:s:0", "handler_name=Air3 Telemetry"]

        if first.probe.rotation:
            # ffmpeg's concat demuxer + stream copy doesn't reliably
            # propagate source rotation side data/tags across the concat
            # boundary; set it explicitly on the output video stream so
            # playback orientation isn't silently lost or corrupted.
            cmd += ["-metadata:s:v:0", f"rotate={first.probe.rotation}"]

        if first.probe.creation_time:
            cmd += ["-metadata", f"creation_time={first.probe.creation_time}"]
        gps_cue = _first_gps_cue(clips)
        if gps_cue:
            cmd += ["-metadata", f"location={_iso6709(gps_cue.latitude, gps_cue.longitude, gps_cue.abs_alt)}"]
        elif first.probe.container_location:
            # No SRT-derived GPS anywhere in the group, but the container
            # itself carries a location tag (e.g. written by DJI's own app)
            # -- use it as a fallback rather than leaving location unset.
            cmd += ["-metadata", f"location={first.probe.container_location}"]

        cmd += [str(tmp_out_path)]

        warnings_before = len(warnings)
        try:
            _run(cmd, warnings)
        except RuntimeError as e:
            tmp_out_path.unlink(missing_ok=True)
            # _next_available_path claimed out_path as an empty
            # placeholder to reserve the name atomically -- don't leave
            # it behind if the merge never actually completed.
            out_path.unlink(missing_ok=True)
            return MergeResult(ok=False, output_path=None, source_files=source_names,
                                error=str(e), warnings=warnings)

        demux_error = procs.concat_demux_error(warnings, warnings_before)
        if demux_error:
            # ffmpeg exited 0, but the concat demuxer failed to open/read
            # one of the listed clips mid-run (deleted, corrupt) and
            # silently proceeded without it -- the file at tmp_out_path
            # is real but missing content, so it must never be renamed
            # into place as if it were a complete merge.
            tmp_out_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
            return MergeResult(
                ok=False, output_path=None, source_files=source_names,
                error=f"ffmpeg reported success but the concat demuxer failed to read "
                      f"part of the input list -- output would be silently missing "
                      f"content: {demux_error}",
                warnings=warnings,
            )

        # Marker-independent backstop -- see concat_duration_shortfall's
        # docstring: a dropped clip changes the merged output's own probed
        # duration regardless of whether ffmpeg's stderr happened to match
        # the known marker text for this ffmpeg build.
        expected_duration = sum(c.probe.duration_s for c in clips)
        try:
            merged_probe = probe_clip(tmp_out_path, warnings)
        except RuntimeError as e:
            tmp_out_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
            return MergeResult(ok=False, output_path=None, source_files=source_names,
                                error=f"merged output failed to probe: {e}", warnings=warnings)
        if procs.concat_duration_shortfall(expected_duration, merged_probe.duration_s):
            tmp_out_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
            return MergeResult(
                ok=False, output_path=None, source_files=source_names,
                error=f"ffmpeg reported success but the merged output's duration "
                      f"({merged_probe.duration_s:.1f}s) is short of the "
                      f"{expected_duration:.1f}s expected from the source clips -- "
                      f"output is likely missing content even though no known "
                      f"failure marker matched stderr",
                warnings=warnings,
            )

    tmp_out_path.rename(out_path)
    return MergeResult(ok=True, output_path=out_path, source_files=source_names, warnings=warnings)
