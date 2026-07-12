"""Tests for dji_repack.stills -- discovering and copying still images
alongside video consolidation."""

from dji_repack.constants import RAW_SPLITS_DIRNAME
from dji_repack.stills import copy_stills, discover_stills


class TestDiscoverStills:
    def test_finds_dng_and_jpg_and_jpeg_case_insensitively(self, tmp_path):
        (tmp_path / "a.DNG").write_bytes(b"x")
        (tmp_path / "b.jpg").write_bytes(b"x")
        (tmp_path / "c.JPEG").write_bytes(b"x")
        (tmp_path / "d.mp4").write_bytes(b"x")  # not a still
        (tmp_path / "e.SRT").write_text("x")  # not a still

        found = {p.name for p in discover_stills(tmp_path)}
        assert found == {"a.DNG", "b.jpg", "c.JPEG"}

    def test_recurses_into_subfolders(self, tmp_path):
        nested = tmp_path / "100MEDIA"
        nested.mkdir()
        (nested / "a.dng").write_bytes(b"x")

        found = discover_stills(tmp_path)
        assert len(found) == 1
        assert found[0] == nested / "a.dng"

    def test_skips_raw_splits_directory(self, tmp_path):
        raw_splits = tmp_path / RAW_SPLITS_DIRNAME
        raw_splits.mkdir()
        (raw_splits / "a.dng").write_bytes(b"x")

        assert discover_stills(tmp_path) == []

    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert discover_stills(tmp_path) == []


class TestCopyStills:
    def test_copies_each_still_to_dest(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        (src / "a.dng").write_bytes(b"raw data")
        (src / "b.jpg").write_bytes(b"jpeg data")

        copied, skipped, warnings = copy_stills(discover_stills(src), dest)

        assert copied == 2
        assert skipped == 0
        assert warnings == []
        assert (dest / "a.dng").read_bytes() == b"raw data"
        assert (dest / "b.jpg").read_bytes() == b"jpeg data"
        # source is untouched -- stills are copied, never moved
        assert (src / "a.dng").exists()

    def test_creates_dest_dir_if_missing(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "does" / "not" / "exist"
        src.mkdir()
        (src / "a.dng").write_bytes(b"x")

        copy_stills(discover_stills(src), dest)

        assert dest.is_dir()
        assert (dest / "a.dng").exists()

    def test_preexisting_file_at_dest_is_skipped_not_overwritten(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        (src / "a.dng").write_bytes(b"new content")
        (dest / "a.dng").write_bytes(b"already there")

        copied, skipped, warnings = copy_stills(discover_stills(src), dest)

        assert copied == 0
        assert skipped == 1
        assert (dest / "a.dng").read_bytes() == b"already there"

    def test_nested_source_flattens_into_dest_root(self, tmp_path):
        src = tmp_path / "src"
        nested = src / "100MEDIA"
        dest = tmp_path / "dest"
        nested.mkdir(parents=True)
        (nested / "a.dng").write_bytes(b"x")

        copy_stills(discover_stills(src), dest)

        assert (dest / "a.dng").exists()
        assert not (dest / "100MEDIA").exists()

    def test_rerun_is_idempotent(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        (src / "a.dng").write_bytes(b"x")

        stills = discover_stills(src)
        copy_stills(stills, dest)
        copied, skipped, _ = copy_stills(stills, dest)

        assert copied == 0
        assert skipped == 1
