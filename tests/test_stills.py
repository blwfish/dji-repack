"""Tests for dji_repack.stills -- discovering and copying still images
alongside video consolidation."""

from pathlib import Path

from dji_repack.constants import RAW_SPLITS_DIRNAME
from dji_repack.stills import copy_stills, discover_stills


class TestDiscoverStills:
    def test_finds_dng_and_jpg_and_jpeg_case_insensitively(self, tmp_path):
        (tmp_path / "a.DNG").write_bytes(b"x")
        (tmp_path / "b.jpg").write_bytes(b"x")
        (tmp_path / "c.JPEG").write_bytes(b"x")
        (tmp_path / "d.mp4").write_bytes(b"x")  # not a still
        (tmp_path / "e.SRT").write_text("x")  # not a still

        found, warnings = discover_stills(tmp_path)
        assert {p.name for p in found} == {"a.DNG", "b.jpg", "c.JPEG"}
        assert warnings == []

    def test_recurses_into_subfolders(self, tmp_path):
        nested = tmp_path / "100MEDIA"
        nested.mkdir()
        (nested / "a.dng").write_bytes(b"x")

        found, warnings = discover_stills(tmp_path)
        assert len(found) == 1
        assert found[0] == nested / "a.dng"
        assert warnings == []

    def test_skips_raw_splits_directory(self, tmp_path):
        raw_splits = tmp_path / RAW_SPLITS_DIRNAME
        raw_splits.mkdir()
        (raw_splits / "a.dng").write_bytes(b"x")

        found, warnings = discover_stills(tmp_path)
        assert found == []
        assert warnings == []

    def test_empty_dir_returns_empty_list(self, tmp_path):
        found, warnings = discover_stills(tmp_path)
        assert found == []
        assert warnings == []


class TestDiscoverStillsScanErrors:
    def test_per_file_stat_error_is_warned_not_silently_dropped(self, tmp_path, monkeypatch):
        """The HIGH regression: discover_stills used to swallow a
        per-file OSError (broken symlink, permission denied) with a bare
        `continue` and no warnings channel at all -- unlike
        video.discover_clips's structurally identical scan loop, which
        returns a warning for the same failure. A still image dropped
        here used to be invisible: not counted, not logged,
        indistinguishable from "there was no such file."."""
        (tmp_path / "a.dng").write_bytes(b"x")
        bad = tmp_path / "bad.dng"
        bad.write_bytes(b"x")

        real_is_file = Path.is_file

        def flaky_is_file(self):
            if self.name == "bad.dng":
                raise OSError("permission denied")
            return real_is_file(self)

        monkeypatch.setattr(Path, "is_file", flaky_is_file)

        found, warnings = discover_stills(tmp_path)

        assert [p.name for p in found] == ["a.dng"]
        assert any("bad.dng" in w for w in warnings)


class TestCopyStills:
    def test_copies_each_still_to_dest(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        (src / "a.dng").write_bytes(b"raw data")
        (src / "b.jpg").write_bytes(b"jpeg data")

        found, _ = discover_stills(src)
        copied, skipped, warnings = copy_stills(found, dest)

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

        found, _ = discover_stills(src)
        copy_stills(found, dest)

        assert dest.is_dir()
        assert (dest / "a.dng").exists()

    def test_preexisting_file_at_dest_is_skipped_not_overwritten(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        (src / "a.dng").write_bytes(b"new content")
        (dest / "a.dng").write_bytes(b"already there")

        found, _ = discover_stills(src)
        copied, skipped, warnings = copy_stills(found, dest)

        assert copied == 0
        assert skipped == 1
        assert (dest / "a.dng").read_bytes() == b"already there"

    def test_nested_source_flattens_into_dest_root(self, tmp_path):
        src = tmp_path / "src"
        nested = src / "100MEDIA"
        dest = tmp_path / "dest"
        nested.mkdir(parents=True)
        (nested / "a.dng").write_bytes(b"x")

        found, _ = discover_stills(src)
        copy_stills(found, dest)

        assert (dest / "a.dng").exists()
        assert not (dest / "100MEDIA").exists()

    def test_rerun_is_idempotent(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        (src / "a.dng").write_bytes(b"x")

        stills, _ = discover_stills(src)
        copy_stills(stills, dest)
        copied, skipped, _ = copy_stills(stills, dest)

        assert copied == 0
        assert skipped == 1


class TestCopyStillsProgress:
    def test_on_progress_called_once_per_copied_file_with_a_nonnegative_elapsed(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.dng").write_bytes(b"x")
        (src / "b.jpg").write_bytes(b"x")

        events = []
        found, _ = discover_stills(src)
        copy_stills(
            found, tmp_path / "dest",
            on_progress=lambda name, elapsed: events.append((name, elapsed)),
        )

        assert {name for name, _ in events} == {"a.dng", "b.jpg"}
        assert all(elapsed >= 0.0 for _, elapsed in events)

    def test_on_progress_not_called_for_a_skipped_file(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        dest.mkdir()
        (src / "a.dng").write_bytes(b"x")
        (dest / "a.dng").write_bytes(b"already there")  # forces a skip, not a copy

        events = []
        found, _ = discover_stills(src)
        copy_stills(found, dest, on_progress=lambda name, elapsed: events.append(name))

        assert events == []

    def test_on_progress_defaults_to_none_without_raising(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.dng").write_bytes(b"x")

        found, _ = discover_stills(src)
        copied, skipped, warnings = copy_stills(found, tmp_path / "dest")

        assert copied == 1
        assert warnings == []
