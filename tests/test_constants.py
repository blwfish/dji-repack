"""Tests for dji_repack.constants -- the single source of truth for the
_raw_splits/ exclusion check shared by video.discover_clips and
stills.discover_stills."""

from dji_repack.constants import RAW_SPLITS_DIRNAME, is_within_raw_splits


class TestIsWithinRawSplits:
    def test_top_level_file_is_not_within(self, tmp_path):
        p = tmp_path / "a.mp4"
        assert is_within_raw_splits(p, tmp_path) is False

    def test_file_directly_inside_raw_splits(self, tmp_path):
        p = tmp_path / RAW_SPLITS_DIRNAME / "a.mp4"
        assert is_within_raw_splits(p, tmp_path) is True

    def test_file_nested_inside_raw_splits(self, tmp_path):
        p = tmp_path / RAW_SPLITS_DIRNAME / "sub" / "a.mp4"
        assert is_within_raw_splits(p, tmp_path) is True

    def test_file_in_sibling_dir_not_within(self, tmp_path):
        p = tmp_path / "100MEDIA" / "a.mp4"
        assert is_within_raw_splits(p, tmp_path) is False

    def test_raw_splits_directory_node_itself_is_not_within(self, tmp_path):
        # The _raw_splits directory entry itself (not a file inside it) --
        # `parts[:-1]` excludes the path's own final component, matching
        # both callers' rglob("*") usage where `p` can be the directory
        # node yielded by rglob, not just files under it.
        p = tmp_path / RAW_SPLITS_DIRNAME
        assert is_within_raw_splits(p, tmp_path) is False


class TestVideoStillsRawSplitsParity:
    """Parity test: video.discover_clips and stills.discover_stills must
    agree on what counts as "already processed" -- both now call the same
    is_within_raw_splits helper, and this test would catch either module
    reverting to its own hand-rolled copy of the check (the HIGH
    regression this shared helper closes off)."""

    def test_both_scanners_exclude_raw_splits_identically(self, tmp_path):
        from dji_repack import stills, video

        raw_splits = tmp_path / RAW_SPLITS_DIRNAME
        raw_splits.mkdir()
        (raw_splits / "old.mp4").write_bytes(b"x")
        (raw_splits / "old.dng").write_bytes(b"x")
        (tmp_path / "new.dng").write_bytes(b"x")

        _clips, video_warnings = video.discover_clips(tmp_path)
        # old.mp4 must never even be attempted -- if the exclusion broke,
        # discover_clips would try to probe it and this would show up as
        # a per-clip warning naming it.
        assert not any("old.mp4" in w for w in video_warnings)

        found_stills, stills_warnings = stills.discover_stills(tmp_path)
        assert found_stills == [tmp_path / "new.dng"]
        assert stills_warnings == []
