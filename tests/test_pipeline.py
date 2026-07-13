"""Tests for dji_repack.pipeline.run_merge_pipeline -- the shared
merge-orchestration sequence previously duplicated independently (and
untested) in cli.py and gui.py.

merge_group itself is monkeypatched here (it's exercised for real,
against real ffmpeg, in test_integration.py) so these tests stay fast and
focus purely on the orchestration: dispatch, archiving, stills, sweeping,
and the callbacks the CLI/GUI each need.
"""

from datetime import datetime

from dji_repack import pipeline
from dji_repack.constants import RAW_SPLITS_DIRNAME
from dji_repack.video import Clip, ClipGroup, ClipProbe, MergeResult

BASE = datetime(2026, 6, 29, 10, 0, 0)


def make_probe(**overrides):
    fields = dict(
        duration_s=60.0, codec_name="hevc", width=3840, height=2160,
        r_frame_rate="60000/1001", pix_fmt="yuv420p10le", rotation=0,
        bit_rate=None, nb_frames=None, container_location=None,
        creation_time=None,
    )
    fields.update(overrides)
    return ClipProbe(**fields)


def make_clip(mp4_path, part_index=None):
    return Clip(
        mp4_path=mp4_path, srt_path=None, cues=[], srt_error=None,
        probe=make_probe(), start_dt=BASE, end_dt=BASE, start_is_estimated=True,
        part_index=part_index,
    )


class TestLoneClipDispatch:
    def test_single_clip_group_is_copied_not_merged(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        mp4 = src / "DJI_20260629100000_0001.MP4"
        mp4.write_bytes(b"video")
        group = ClipGroup(clips=[make_clip(mp4)])

        logs = []
        done = []
        result = pipeline.run_merge_pipeline(
            [group], src, dest,
            log=lambda msg, stderr=False: logs.append((msg, stderr)),
            on_group_done=lambda g: done.append(g),
        )

        assert result.copied_lone_count == 1
        assert result.merged_count == 0
        assert (dest / mp4.name).read_bytes() == b"video"
        assert done == [group]
        assert any("copied" in msg for msg, _ in logs)


class TestMergeDispatch:
    def test_multi_clip_group_calls_merge_group_and_archives_on_success(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        a = src / "a.MP4"
        b = src / "b.MP4"
        a.write_bytes(b"video")
        b.write_bytes(b"video")
        group = ClipGroup(clips=[make_clip(a), make_clip(b)])

        merge_calls = []

        def fake_merge_group(group, dest_dir):
            merge_calls.append(group)
            out = dest_dir / "AIR3_merged.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"merged")
            return MergeResult(ok=True, output_path=out, source_files=[c.mp4_path.name for c in group.clips])

        monkeypatch.setattr(pipeline, "merge_group", fake_merge_group)

        result = pipeline.run_merge_pipeline([group], src, dest, do_stills=False)

        assert len(merge_calls) == 1
        assert result.merged_count == 1
        raw_splits = dest / RAW_SPLITS_DIRNAME
        assert (raw_splits / "a.MP4").exists()
        assert (raw_splits / "b.MP4").exists()
        assert not a.exists()  # archived out of the source dir
        assert not b.exists()

    def test_do_archive_false_leaves_source_clips_in_place(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        a = src / "a.MP4"
        b = src / "b.MP4"
        a.write_bytes(b"video")
        b.write_bytes(b"video")
        group = ClipGroup(clips=[make_clip(a), make_clip(b)])

        def fake_merge_group(group, dest_dir):
            out = dest_dir / "AIR3_merged.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"merged")
            return MergeResult(ok=True, output_path=out, source_files=[c.mp4_path.name for c in group.clips])

        monkeypatch.setattr(pipeline, "merge_group", fake_merge_group)

        pipeline.run_merge_pipeline([group], src, dest, do_archive=False, do_stills=False)

        assert a.exists()
        assert b.exists()
        assert not (dest / RAW_SPLITS_DIRNAME).exists()

    def test_merge_failure_is_not_archived_and_counts_as_failed(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        a = src / "a.MP4"
        b = src / "b.MP4"
        a.write_bytes(b"video")
        b.write_bytes(b"video")
        group = ClipGroup(clips=[make_clip(a), make_clip(b)])

        def fake_merge_group(group, dest_dir):
            return MergeResult(ok=False, output_path=None,
                                source_files=[c.mp4_path.name for c in group.clips],
                                error="simulated failure")

        monkeypatch.setattr(pipeline, "merge_group", fake_merge_group)

        logs = []
        result = pipeline.run_merge_pipeline(
            [group], src, dest, do_stills=False,
            log=lambda msg, stderr=False: logs.append((msg, stderr)),
        )

        assert result.failed_count == 1
        assert result.merged_count == 0
        assert a.exists() and b.exists()  # never archived
        assert not (dest / RAW_SPLITS_DIRNAME).exists()
        assert any("FAILED" in msg and stderr for msg, stderr in logs)

    def test_gap_warning_is_logged_for_a_multi_clip_group(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        a = src / "DJI_x_0001.MP4"
        b = src / "DJI_x_0003.MP4"
        a.write_bytes(b"video")
        b.write_bytes(b"video")
        group = ClipGroup(clips=[make_clip(a, part_index=1), make_clip(b, part_index=3)])

        def fake_merge_group(group, dest_dir):
            out = dest_dir / "AIR3_merged.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"merged")
            return MergeResult(ok=True, output_path=out, source_files=[c.mp4_path.name for c in group.clips])

        monkeypatch.setattr(pipeline, "merge_group", fake_merge_group)

        logs = []
        pipeline.run_merge_pipeline(
            [group], src, dest, do_stills=False,
            log=lambda msg, stderr=False: logs.append(msg),
        )

        assert any("non-consecutive part numbers" in msg for msg in logs)


class TestStillsIntegration:
    def test_stills_copied_when_enabled(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        (src / "a.dng").write_bytes(b"raw")

        result = pipeline.run_merge_pipeline([], src, dest, do_stills=True)

        assert result.stills_copied == 1
        assert (dest / "a.dng").exists()

    def test_stills_skipped_when_disabled(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        (src / "a.dng").write_bytes(b"raw")

        result = pipeline.run_merge_pipeline([], src, dest, do_stills=False)

        assert result.stills_copied == 0
        assert not (dest / "a.dng").exists()

    def test_on_still_progress_callback_fires(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        (src / "a.dng").write_bytes(b"raw")

        events = []
        pipeline.run_merge_pipeline(
            [], src, dest, do_stills=True,
            on_still_progress=lambda name, elapsed: events.append(name),
        )

        assert events == ["a.dng"]


class TestSweepIntegration:
    def test_stale_partial_in_dest_is_swept_and_logged(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        stale = dest / ".AIR3_20260629_100000.partial.mp4"
        stale.write_bytes(b"x")

        logs = []
        result = pipeline.run_merge_pipeline(
            [], src, dest, do_stills=False,
            log=lambda msg, stderr=False: logs.append(msg),
        )

        assert result.swept_partials == [stale.name]
        assert not stale.exists()
        assert any("removed leftover partial" in msg for msg in logs)


class TestEmptyGroupsList:
    def test_no_groups_and_no_stills_is_a_no_op(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()

        result = pipeline.run_merge_pipeline([], src, dest, do_stills=False)

        assert result.merged_count == 0
        assert result.copied_lone_count == 0
        assert result.failed_count == 0
