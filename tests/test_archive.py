"""Tests for dji_repack.archive.archive_merged_group and copy_lone_clip."""

from datetime import datetime

from dji_repack.archive import archive_merged_group, copy_lone_clip
from dji_repack.constants import RAW_SPLITS_DIRNAME
from dji_repack.video import Clip, ClipGroup, ClipProbe

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


def make_clip(mp4_path, srt_path=None):
    return Clip(
        mp4_path=mp4_path, srt_path=srt_path, cues=[], srt_error=None,
        probe=make_probe(), start_dt=BASE, end_dt=BASE, start_is_estimated=True,
    )


class TestArchiveMergedGroup:
    def test_moves_mp4_and_srt_into_raw_splits(self, tmp_path):
        mp4 = tmp_path / "DJI_20260629100000_0001.MP4"
        srt = tmp_path / "DJI_20260629100000_0001.SRT"
        mp4.write_bytes(b"video")
        srt.write_text("telemetry")
        group = ClipGroup(clips=[make_clip(mp4, srt)])

        warnings = archive_merged_group(group, tmp_path)

        assert warnings == []
        raw_splits = tmp_path / RAW_SPLITS_DIRNAME
        assert (raw_splits / mp4.name).exists()
        assert (raw_splits / srt.name).exists()
        assert not mp4.exists()
        assert not srt.exists()

    def test_clip_with_no_srt_moves_only_the_mp4(self, tmp_path):
        mp4 = tmp_path / "DJI_20260629100000_0001.MP4"
        mp4.write_bytes(b"video")
        group = ClipGroup(clips=[make_clip(mp4)])

        warnings = archive_merged_group(group, tmp_path)

        assert warnings == []
        assert (tmp_path / RAW_SPLITS_DIRNAME / mp4.name).exists()

    def test_creates_raw_splits_dir_if_missing(self, tmp_path):
        mp4 = tmp_path / "DJI_20260629100000_0001.MP4"
        mp4.write_bytes(b"video")
        group = ClipGroup(clips=[make_clip(mp4)])

        archive_merged_group(group, tmp_path)

        assert (tmp_path / RAW_SPLITS_DIRNAME).is_dir()

    def test_one_bad_rename_does_not_abort_the_rest_of_the_group(self, tmp_path):
        # Second clip's rename destination is pre-occupied by a directory,
        # forcing an OSError on that one rename -- the first clip must
        # still be archived rather than the whole group aborting.
        mp4_a = tmp_path / "a.MP4"
        mp4_b = tmp_path / "b.MP4"
        mp4_a.write_bytes(b"video")
        mp4_b.write_bytes(b"video")
        raw_splits = tmp_path / RAW_SPLITS_DIRNAME
        raw_splits.mkdir()
        (raw_splits / "b.MP4").mkdir()  # collides with b.MP4's rename target

        group = ClipGroup(clips=[make_clip(mp4_a), make_clip(mp4_b)])
        warnings = archive_merged_group(group, tmp_path)

        assert not mp4_a.exists()
        assert (raw_splits / "a.MP4").exists()
        assert mp4_b.exists()  # left in place, since the rename failed
        assert len(warnings) == 1
        assert "b.MP4" in warnings[0]


class TestCopyLoneClip:
    def test_copies_mp4_and_srt_to_a_different_dest(self, tmp_path):
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        mp4 = src_dir / "DJI_20260629100000_0001.MP4"
        srt = src_dir / "DJI_20260629100000_0001.SRT"
        mp4.write_bytes(b"video")
        srt.write_text("telemetry")

        warnings = copy_lone_clip(make_clip(mp4, srt), dest_dir)

        assert warnings == []
        assert (dest_dir / mp4.name).read_bytes() == b"video"
        assert (dest_dir / srt.name).read_text() == "telemetry"
        # source is untouched -- a lone clip is copied, never moved
        assert mp4.exists()
        assert srt.exists()

    def test_clip_with_no_srt_copies_only_the_mp4(self, tmp_path):
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        mp4 = src_dir / "DJI_20260629100000_0001.MP4"
        mp4.write_bytes(b"video")

        warnings = copy_lone_clip(make_clip(mp4), dest_dir)

        assert warnings == []
        assert (dest_dir / mp4.name).exists()
        assert list(dest_dir.iterdir()) == [dest_dir / mp4.name]

    def test_in_place_merge_where_dest_equals_source_is_a_safe_no_op(self, tmp_path):
        # dest_dir == the clip's own folder: the "destination" file IS the
        # source file, so this must not raise shutil.SameFileError or
        # otherwise touch it.
        mp4 = tmp_path / "DJI_20260629100000_0001.MP4"
        mp4.write_bytes(b"video")

        warnings = copy_lone_clip(make_clip(mp4), tmp_path)

        assert warnings == []
        assert mp4.read_bytes() == b"video"

    def test_preexisting_file_at_dest_is_skipped_not_overwritten(self, tmp_path):
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        dest_dir.mkdir()
        mp4 = src_dir / "DJI_20260629100000_0001.MP4"
        mp4.write_bytes(b"new content")
        (dest_dir / mp4.name).write_bytes(b"already there")

        warnings = copy_lone_clip(make_clip(mp4), dest_dir)

        assert warnings == []
        assert (dest_dir / mp4.name).read_bytes() == b"already there"

    def test_does_not_archive_into_raw_splits(self, tmp_path):
        # A lone clip was never merged into anything -- it must land at
        # dest_dir's top level, not get tucked into _raw_splits/ the way a
        # successfully-merged group's originals do.
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_dir.mkdir()
        mp4 = src_dir / "DJI_20260629100000_0001.MP4"
        mp4.write_bytes(b"video")

        copy_lone_clip(make_clip(mp4), dest_dir)

        assert not (dest_dir / RAW_SPLITS_DIRNAME).exists()
        assert (dest_dir / mp4.name).exists()

    def test_creates_dest_dir_if_missing(self, tmp_path):
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "does" / "not" / "exist"
        src_dir.mkdir()
        mp4 = src_dir / "DJI_20260629100000_0001.MP4"
        mp4.write_bytes(b"video")

        copy_lone_clip(make_clip(mp4), dest_dir)

        assert (dest_dir / mp4.name).exists()
