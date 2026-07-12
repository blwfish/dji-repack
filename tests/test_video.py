"""Tests for dji_repack.video -- grouping, path-safety, and telemetry-lookup
logic.

These are pure-logic tests: no ffmpeg/ffprobe subprocess calls (that's
covered by real integration tests in test_integration.py, plus manual
verification against real Air3 footage). Constructs Clip/ClipProbe/SrtCue
directly so grouping/threshold behavior can be pinned precisely and
cheaply.

Ported from mneme's tests/test_repack_video.py.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from dji_repack import video
from dji_repack.srt_parser import SrtCue

BASE = datetime(2026, 6, 29, 10, 0, 0)


def make_probe(**overrides):
    fields = dict(
        duration_s=60.0, codec_name="hevc", width=3840, height=2160,
        r_frame_rate="60000/1001", pix_fmt="yuv420p10le", rotation=0,
        bit_rate=None, nb_frames=None, container_location=None,
        creation_time="2026-06-29T14:00:00.000000Z",
    )
    fields.update(overrides)
    return video.ClipProbe(**fields)


def make_cue(wall_clock, has_gps=True, **overrides):
    fields = dict(
        framecnt=1, difftime_ms=33, cue_start_s=0.0, cue_end_s=0.033,
        wall_clock=wall_clock, iso=100, shutter="1/2000.0", fnum=1.7, ev=0.0,
        color_md="default", focal_len=24.0,
        latitude=41.5 if has_gps else None,
        longitude=-75.5 if has_gps else None,
        rel_alt=10.0 if has_gps else None,
        abs_alt=200.0 if has_gps else None,
        ct=5500,
    )
    fields.update(overrides)
    return SrtCue(**fields)


def make_clip(mp4_path, start_dt, end_dt, cues=None, probe=None, srt_error=None, part_index=None):
    cues = cues or []
    return video.Clip(
        mp4_path=Path(mp4_path),
        srt_path=None,
        cues=cues,
        srt_error=srt_error,
        probe=probe or make_probe(),
        start_dt=start_dt,
        end_dt=end_dt,
        start_is_estimated=not bool(cues),
        part_index=part_index,
    )


class TestGroupClipsThreshold:
    """Pins the gap < gap_threshold_s contract (strict less-than): a gap
    exactly equal to the threshold must NOT merge."""

    def test_empty_list(self):
        assert video.group_clips([], gap_threshold_s=300) == []

    def test_single_clip(self):
        clip = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        groups = video.group_clips([clip], gap_threshold_s=300)
        assert len(groups) == 1
        assert groups[0].clips == [clip]
        assert groups[0].gap_to_next_s is None

    def test_gap_just_below_threshold_merges(self):
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        b_start = a.end_dt + timedelta(seconds=299.999)
        b = make_clip("/fake/b.mp4", b_start, b_start + timedelta(seconds=60))
        groups = video.group_clips([a, b], gap_threshold_s=300)
        assert len(groups) == 1
        assert groups[0].clips == [a, b]

    def test_gap_exactly_at_threshold_does_not_merge(self):
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        b_start = a.end_dt + timedelta(seconds=300.0)
        b = make_clip("/fake/b.mp4", b_start, b_start + timedelta(seconds=60))
        groups = video.group_clips([a, b], gap_threshold_s=300)
        assert len(groups) == 2

    def test_gap_just_above_threshold_splits(self):
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        b_start = a.end_dt + timedelta(seconds=300.001)
        b = make_clip("/fake/b.mp4", b_start, b_start + timedelta(seconds=60))
        groups = video.group_clips([a, b], gap_threshold_s=300)
        assert len(groups) == 2

    def test_negative_gap_merges(self):
        # Overlapping/out-of-order timestamps (e.g. clock jitter) should
        # still merge -- a negative gap is trivially < any positive threshold.
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        b_start = a.end_dt - timedelta(seconds=5)
        b = make_clip("/fake/b.mp4", b_start, b_start + timedelta(seconds=60))
        groups = video.group_clips([a, b], gap_threshold_s=300)
        assert len(groups) == 1

    def test_three_clips_two_groups_with_correct_gap_to_next(self):
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        b_start = a.end_dt + timedelta(seconds=1)  # chunk-split-style tiny gap
        b = make_clip("/fake/b.mp4", b_start, b_start + timedelta(seconds=60))
        c_start = b.end_dt + timedelta(seconds=3600)  # different flight
        c = make_clip("/fake/c.mp4", c_start, c_start + timedelta(seconds=60))
        groups = video.group_clips([a, b, c], gap_threshold_s=300)
        assert len(groups) == 2
        assert groups[0].clips == [a, b]
        assert groups[1].clips == [c]
        assert groups[0].gap_to_next_s == pytest.approx(3600.0)
        assert groups[1].gap_to_next_s is None


class TestIso6709:
    def test_positive_lat_negative_lon(self):
        # No zero-padding on the integer part -- matches real Apple/QuickTime
        # ISO6709 location tags (e.g. "+37.3349-122.0090+075.000/").
        assert video._iso6709(41.620801, -75.778972, 220.994) == "+41.6208-75.7790+220.994/"

    def test_precision_truncated_to_fixed_decimals(self):
        s = video._iso6709(1.23456789, -2.3456789, 3.456789)
        assert s == "+1.2346-2.3457+3.457/"


class TestNextAvailablePath:
    def test_no_collision_returns_original(self, tmp_path):
        target = tmp_path / "AIR3_20260629_100000.mp4"
        assert video._next_available_path(target) == target

    def test_claims_the_path_atomically_as_a_side_effect(self, tmp_path):
        """The MEDIUM regression: a `while path.exists(): ...` check-then-
        act loop left a TOCTOU window between checking and creating --
        two concurrent callers could both pick the same "available" name
        and the second's eventual rename() would silently clobber the
        first's output. The claimed path must now exist immediately (an
        O_CREAT|O_EXCL side effect), and a second call against the same
        original target must claim a DIFFERENT path."""
        target = tmp_path / "AIR3_20260629_100000.mp4"
        first = video._next_available_path(target)
        assert first.exists()

        second = video._next_available_path(target)
        assert second != first
        assert second.exists()

    def test_one_collision_returns_v2(self, tmp_path):
        target = tmp_path / "AIR3_20260629_100000.mp4"
        target.write_bytes(b"")
        result = video._next_available_path(target)
        assert result == tmp_path / "AIR3_20260629_100000_v2.mp4"

    def test_two_collisions_returns_v3(self, tmp_path):
        target = tmp_path / "AIR3_20260629_100000.mp4"
        target.write_bytes(b"")
        (tmp_path / "AIR3_20260629_100000_v2.mp4").write_bytes(b"")
        result = video._next_available_path(target)
        assert result == tmp_path / "AIR3_20260629_100000_v3.mp4"


class TestCheckUniformStream:
    def test_matching_streams_returns_none(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe())
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe())
        assert video._check_uniform_stream([a, b]) is None

    def test_mismatched_resolution_returns_message(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe(width=3840, height=2160))
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe(width=1920, height=1080))
        msg = video._check_uniform_stream([a, b])
        assert msg is not None
        assert "a.mp4" in msg and "b.mp4" in msg

    def test_mismatched_codec_returns_message(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe(codec_name="hevc"))
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe(codec_name="h264"))
        assert video._check_uniform_stream([a, b]) is not None

    def test_mismatched_frame_rate_returns_message(self):
        # Regression: r_frame_rate used to be in the comparison tuple in
        # name only -- every make_probe() call shared the same hardcoded
        # value, so a frame-rate-only mismatch was never actually
        # exercised. -c:v copy concat across differing frame rates
        # silently corrupts the output.
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe(r_frame_rate="60000/1001"))
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe(r_frame_rate="30000/1001"))
        assert video._check_uniform_stream([a, b]) is not None

    def test_mismatched_pix_fmt_returns_message(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe(pix_fmt="yuv420p"))
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe(pix_fmt="yuv420p10le"))
        assert video._check_uniform_stream([a, b]) is not None

    def test_mismatched_rotation_returns_message(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe(rotation=0))
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe(rotation=90))
        assert video._check_uniform_stream([a, b]) is not None

    def test_audio_presence_mismatch_returns_message(self):
        """The MEDIUM regression: audio wasn't part of the uniformity
        check at all -- one clip having a camera-mic track and another
        having none would have silently passed this check, then broken
        merge_group's single 0:a:0 mapping."""
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe(
            audio_codec_name="aac", audio_sample_rate=48000, audio_channels=2,
        ))
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe())  # no audio
        assert video._check_uniform_stream([a, b]) is not None

    def test_audio_codec_mismatch_returns_message(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe(
            audio_codec_name="aac", audio_sample_rate=48000, audio_channels=2,
        ))
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe(
            audio_codec_name="pcm_s16le", audio_sample_rate=48000, audio_channels=2,
        ))
        assert video._check_uniform_stream([a, b]) is not None

    def test_matching_audio_streams_returns_none(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe(
            audio_codec_name="aac", audio_sample_rate=48000, audio_channels=2,
        ))
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe(
            audio_codec_name="aac", audio_sample_rate=48000, audio_channels=2,
        ))
        assert video._check_uniform_stream([a, b]) is None

    def test_neither_clip_having_audio_returns_none(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe())
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe())
        assert video._check_uniform_stream([a, b]) is None
        assert a.probe.has_audio is False


class TestGpsCueLookup:
    def test_first_gps_cue_skips_clip_with_no_cues(self):
        no_srt = make_clip("/fake/a.mp4", BASE, BASE, cues=[])
        with_gps = make_clip("/fake/b.mp4", BASE, BASE, cues=[make_cue(BASE, has_gps=True)])
        found = video._first_gps_cue([no_srt, with_gps])
        assert found is not None
        assert found.latitude == 41.5

    def test_first_gps_cue_skips_cues_without_gps(self):
        clip = make_clip("/fake/a.mp4", BASE, BASE, cues=[
            make_cue(BASE, has_gps=False),
            make_cue(BASE, has_gps=True, latitude=1.0, longitude=2.0),
        ])
        found = video._first_gps_cue([clip])
        assert found.latitude == 1.0

    def test_last_gps_cue_finds_last_valid_before_tail_dropout(self):
        # Mirrors the real observed pattern: last N cues in the last clip
        # lost GPS lock right before touchdown.
        clip = make_clip("/fake/a.mp4", BASE, BASE, cues=[
            make_cue(BASE, has_gps=True, latitude=9.0, longitude=9.0),
            make_cue(BASE, has_gps=False),
            make_cue(BASE, has_gps=False),
        ])
        found = video._last_gps_cue([clip])
        assert found is not None
        assert found.latitude == 9.0

    def test_no_gps_anywhere_returns_none(self):
        clip = make_clip("/fake/a.mp4", BASE, BASE, cues=[make_cue(BASE, has_gps=False)])
        assert video._first_gps_cue([clip]) is None
        assert video._last_gps_cue([clip]) is None


class TestGroupSummary:
    def _real_file(self, tmp_path, name, size=1024):
        p = tmp_path / name
        p.write_bytes(b"x" * size)
        return p

    def test_location_none_when_no_gps_in_group(self, tmp_path):
        clip = make_clip(self._real_file(tmp_path, "a.mp4"), BASE, BASE + timedelta(seconds=60),
                          cues=[make_cue(BASE, has_gps=False)])
        group = video.ClipGroup(clips=[clip])
        summary = video.group_summary(group)
        assert summary["start_location"] is None
        assert summary["end_location"] is None

    def test_missing_srt_listed_by_filename(self, tmp_path):
        clip = make_clip(self._real_file(tmp_path, "a.mp4"), BASE, BASE + timedelta(seconds=60),
                          cues=[], srt_error="no .SRT sidecar found")
        group = video.ClipGroup(clips=[clip])
        summary = video.group_summary(group)
        assert summary["missing_srt"] == ["a.mp4"]

    def test_total_size_and_duration_summed_across_clips(self, tmp_path):
        a = make_clip(self._real_file(tmp_path, "a.mp4", size=100), BASE, BASE + timedelta(seconds=10),
                       probe=make_probe(duration_s=10.0))
        b = make_clip(self._real_file(tmp_path, "b.mp4", size=200), BASE, BASE + timedelta(seconds=20),
                       probe=make_probe(duration_s=20.0))
        summary = video.group_summary(video.ClipGroup(clips=[a, b]))
        assert summary["total_size_bytes"] == 300
        assert summary["total_duration_s"] == 30.0

    def test_clip_paths_are_full_paths_not_just_names(self, tmp_path):
        # Regression: group identity used to be name-only (clip_names),
        # which collapses same-named clips from different *MEDIA
        # subfolders. clip_paths is the field the frontend now uses for
        # the scan->select->process round trip.
        nested = tmp_path / "100MEDIA"
        nested.mkdir()
        clip = make_clip(self._real_file(nested, "DJI_0001.MP4"), BASE, BASE + timedelta(seconds=10))
        summary = video.group_summary(video.ClipGroup(clips=[clip]))
        assert summary["clip_names"] == ["DJI_0001.MP4"]
        assert summary["clip_paths"] == [str(nested / "DJI_0001.MP4")]


class TestRequiredStreamField:
    """The HIGH regression: ffprobe's own "N/A" sentinel (distinct from
    the key being absent from the JSON entirely) must raise this
    function's own RuntimeError -- discover_clips only catches
    RuntimeError, so letting "N/A" fall through to an unguarded
    int("N/A") elsewhere raises ValueError instead and aborts discovery
    of the whole source directory rather than skipping one bad clip."""

    def test_present_value_is_returned(self):
        assert video._required_stream_field({"width": "1920"}, "width", Path("/fake/a.mp4")) == "1920"

    def test_missing_key_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="missing 'width'"):
            video._required_stream_field({}, "width", Path("/fake/a.mp4"))

    def test_na_sentinel_raises_runtime_error_not_value_error(self):
        with pytest.raises(RuntimeError, match="missing 'width'"):
            video._required_stream_field({"width": "N/A"}, "width", Path("/fake/a.mp4"))


class TestStreamRotation:
    def test_no_rotation_info_returns_zero(self):
        assert video._stream_rotation({}) == 0

    def test_side_data_rotation_used(self):
        stream = {"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]}
        assert video._stream_rotation(stream) == -90

    def test_legacy_rotate_tag_used_as_fallback(self):
        stream = {"tags": {"rotate": "180"}}
        assert video._stream_rotation(stream) == 180

    def test_side_data_takes_precedence_over_legacy_tag(self):
        stream = {
            "side_data_list": [{"side_data_type": "Display Matrix", "rotation": 90}],
            "tags": {"rotate": "270"},
        }
        assert video._stream_rotation(stream) == 90


class TestFilenameStart:
    def test_extracts_timestamp_and_suffix(self):
        start_dt, suffix = video._filename_start(Path("DJI_20260629142707_0001_D.MP4"))
        assert start_dt == datetime(2026, 6, 29, 14, 27, 7)
        assert suffix == "0001_D"

    def test_suffix_empty_when_nothing_follows_timestamp(self):
        _start_dt, suffix = video._filename_start(Path("DJI_20260629142707_.MP4"))
        assert suffix == ""

    def test_non_dji_filename_raises(self):
        with pytest.raises(ValueError):
            video._filename_start(Path("IMG_1234.MP4"))

    def test_missing_trailing_underscore_raises(self):
        # No trailing "_" immediately after the timestamp digits -- must
        # not silently misparse or fall through; a bad filename here used
        # to abort discovery of the entire source directory (now isolated
        # per-item in discover_clips, but _filename_start itself must
        # still fail loud).
        with pytest.raises(ValueError):
            video._filename_start(Path("DJI_20260629142707.MP4"))

    def test_calendrically_invalid_date_raises(self):
        with pytest.raises(ValueError):
            video._filename_start(Path("DJI_20261332999999_0001.MP4"))


class TestFilenamePartIndex:
    """_filename_part_index is independent of whether SRT telemetry is
    available (unlike filename_suffix) -- these tests exercise it
    directly against filenames, not through discover_clips."""

    def test_extracts_leading_digits(self):
        assert video._filename_part_index(Path("DJI_20260629142707_0001_D.MP4")) == 1

    def test_multi_digit_index(self):
        assert video._filename_part_index(Path("DJI_20260629142707_0042_D.MP4")) == 42

    def test_no_digits_after_timestamp_returns_none(self):
        assert video._filename_part_index(Path("DJI_20260629142707_D.MP4")) is None

    def test_non_dji_filename_returns_none(self):
        # Unlike _filename_start, this never raises -- it's a best-effort
        # supplementary signal, not something that can abort discovery.
        assert video._filename_part_index(Path("IMG_1234.MP4")) is None


class TestGroupPartIndexGapWarning:
    """The HIGH regression: group_clips() groups purely by wall-clock gap
    with no part-count check at all -- a middle segment that failed to
    copy, whose neighbors still fall within the gap threshold, merges
    into a "complete-looking" file with a chunk silently gone."""

    def test_fewer_than_two_parseable_indices_returns_none(self):
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60), part_index=1)
        b = make_clip("/fake/b.mp4", BASE, BASE + timedelta(seconds=60), part_index=None)
        group = video.ClipGroup(clips=[a, b])
        assert video.group_part_index_gap_warning(group) is None

    def test_consecutive_indices_returns_none(self):
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60), part_index=1)
        b = make_clip("/fake/b.mp4", BASE, BASE + timedelta(seconds=60), part_index=2)
        c = make_clip("/fake/c.mp4", BASE, BASE + timedelta(seconds=60), part_index=3)
        group = video.ClipGroup(clips=[a, b, c])
        assert video.group_part_index_gap_warning(group) is None

    def test_gap_in_indices_returns_a_warning_naming_both_clips(self):
        a = make_clip("/fake/DJI_x_0001.mp4", BASE, BASE + timedelta(seconds=60), part_index=1)
        b = make_clip("/fake/DJI_x_0003.mp4", BASE, BASE + timedelta(seconds=60), part_index=3)
        group = video.ClipGroup(clips=[a, b])
        warning = video.group_part_index_gap_warning(group)
        assert warning is not None
        assert "DJI_x_0001.mp4" in warning
        assert "DJI_x_0003.mp4" in warning

    def test_none_indices_interspersed_do_not_mask_a_real_gap(self):
        """A clip with no parseable index (e.g. it had usable SRT cues,
        so filename parsing was never attempted) must not silently
        swallow a genuine gap between its indexed neighbors."""
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60), part_index=1)
        b = make_clip("/fake/b.mp4", BASE, BASE + timedelta(seconds=60), part_index=None)
        c = make_clip("/fake/c.mp4", BASE, BASE + timedelta(seconds=60), part_index=3)
        group = video.ClipGroup(clips=[a, b, c])
        assert video.group_part_index_gap_warning(group) is not None

    def test_single_clip_group_returns_none(self):
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60), part_index=1)
        group = video.ClipGroup(clips=[a])
        assert video.group_part_index_gap_warning(group) is None


class TestMergeGroupFailureCleanup:
    """The MEDIUM regression: _next_available_path now claims out_path
    atomically (as an empty placeholder) before ffmpeg ever runs, to
    close a TOCTOU race between two concurrent merges picking the same
    "available" name. If the ffmpeg run then fails, that placeholder
    must be cleaned up too, not just tmp_out_path -- otherwise a failed
    merge leaves a stray empty file sitting at the real output name,
    which wasn't possible before the placeholder existed."""

    def test_run_failure_cleans_up_both_tmp_and_claimed_out_path(self, tmp_path, monkeypatch):
        clip = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        group = video.ClipGroup(clips=[clip])

        def failing_run(cmd, warnings):
            raise RuntimeError("simulated ffmpeg failure")

        monkeypatch.setattr(video, "_run", failing_run)

        dest_dir = tmp_path / "out"
        result = video.merge_group(group, dest_dir)

        assert not result.ok
        expected_out_path = dest_dir / f"AIR3_{BASE.strftime('%Y%m%d_%H%M%S')}.mp4"
        assert not expected_out_path.exists()
        assert list(dest_dir.glob("*")) == []  # no stray partial/placeholder files either


class TestBuildMergedTelemetry:
    """Pins the has_telemetry/cumulative_offset logic extracted from
    merge_group -- previously only reachable through the full ffmpeg
    subprocess pipeline, so the silent-gap bug it contains had zero test
    coverage."""

    def test_no_clips_have_cues_returns_none_and_no_warnings(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, cues=[], probe=make_probe(duration_s=10.0))
        b = make_clip("/fake/b.mp4", BASE, BASE, cues=[], probe=make_probe(duration_s=10.0))
        text, warnings = video._build_merged_telemetry([a, b])
        assert text is None
        assert warnings == []

    def test_all_clips_have_cues_no_gap_warnings(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, cues=[make_cue(BASE)], probe=make_probe(duration_s=10.0))
        b = make_clip("/fake/b.mp4", BASE, BASE, cues=[make_cue(BASE)], probe=make_probe(duration_s=10.0))
        text, warnings = video._build_merged_telemetry([a, b])
        assert text is not None
        assert warnings == []

    def test_cueless_clip_mid_group_produces_gap_warning(self):
        # Regression for the silent-gap bug: has_telemetry used to be a
        # group-wide OR with no signal that a cueless clip mid-group
        # leaves a real hole in the merged subtitle track.
        a = make_clip("/fake/a.mp4", BASE, BASE, cues=[make_cue(BASE)], probe=make_probe(duration_s=10.0))
        b = make_clip("/fake/b.mp4", BASE, BASE, cues=[], probe=make_probe(duration_s=25.0))
        c = make_clip("/fake/c.mp4", BASE, BASE, cues=[make_cue(BASE)], probe=make_probe(duration_s=10.0))
        text, warnings = video._build_merged_telemetry([a, b, c])
        assert text is not None
        assert len(warnings) == 1
        assert "b.mp4" in warnings[0]
        assert "25.0s gap" in warnings[0]

    def test_cumulative_offset_advances_through_cueless_clip(self):
        # The cueless clip's duration must still be added to
        # cumulative_offset so a later clip's cues land at the right point
        # in the merged timeline instead of overlapping an earlier cue.
        a = make_clip("/fake/a.mp4", BASE, BASE,
                       cues=[make_cue(BASE, cue_start_s=0.0, cue_end_s=1.0)],
                       probe=make_probe(duration_s=10.0))
        b = make_clip("/fake/b.mp4", BASE, BASE, cues=[], probe=make_probe(duration_s=25.0))
        c = make_clip("/fake/c.mp4", BASE, BASE,
                       cues=[make_cue(BASE, cue_start_s=0.0, cue_end_s=1.0)],
                       probe=make_probe(duration_s=10.0))
        text, _warnings = video._build_merged_telemetry([a, b, c])
        # clip c's cue starts at cumulative_offset = 10.0 (a) + 25.0 (b) = 35.0s
        assert "00:00:35,000 --> 00:00:36,000" in text
