"""Tests for dji_repack.gui.App -- the Tkinter GUI's own logic (gap-
threshold parsing/validation, merge-worker wiring, tree population),
independent of a real ffmpeg install or actual button clicks.

The CRITICAL/HIGH gap this closes: gui.py previously had zero test
coverage of any kind. A real (but hidden/withdrawn) Tk root is used since
App's widgets are real Tk variables/widgets -- macOS Tk works headlessly
without an X server, but if no display is available at all the fixture
skips rather than failing the whole suite.
"""

from datetime import datetime, timedelta

import pytest

tk = pytest.importorskip("tkinter")

from dji_repack import gui
from dji_repack.video import Clip, ClipGroup, ClipProbe


def make_probe(**overrides):
    fields = dict(
        duration_s=60.0, codec_name="hevc", width=3840, height=2160,
        r_frame_rate="60000/1001", pix_fmt="yuv420p10le", rotation=0,
        bit_rate=None, nb_frames=None, container_location=None,
        creation_time=None,
    )
    fields.update(overrides)
    return ClipProbe(**fields)


BASE = datetime(2026, 6, 29, 10, 0, 0)


def make_clip(mp4_path):
    return Clip(
        mp4_path=mp4_path, srt_path=None, cues=[], srt_error=None,
        probe=make_probe(), start_dt=BASE, end_dt=BASE + timedelta(seconds=10),
        start_is_estimated=True,
    )


@pytest.fixture
def app(monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no Tk display available: {e}")
    root.withdraw()
    # Never let a real modal dialog pop up and block the test run --
    # record calls instead so assertions can check what would have shown.
    shown_errors = []
    shown_info = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda title, msg: shown_errors.append(msg))
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda title, msg: shown_info.append(msg))
    application = gui.App(root)
    application.shown_errors = shown_errors
    application.shown_info = shown_info
    yield application
    root.destroy()


class TestGapThreshold:
    def test_empty_field_returns_default(self, app):
        app.gap_var.set("")
        assert app._gap_threshold() == gui.GAP_THRESHOLD_DEFAULT_S

    def test_whitespace_only_field_returns_default(self, app):
        app.gap_var.set("   ")
        assert app._gap_threshold() == gui.GAP_THRESHOLD_DEFAULT_S

    def test_valid_number_is_parsed(self, app):
        app.gap_var.set("120")
        assert app._gap_threshold() == 120.0

    def test_non_numeric_junk_returns_none_and_shows_error(self, app):
        """The MEDIUM regression: this used to silently collapse to the
        default, indistinguishable from an intentionally-blank field."""
        app.gap_var.set("not-a-number")
        assert app._gap_threshold() is None
        assert len(app.shown_errors) == 1

    def test_negative_value_returns_none_and_shows_error(self, app):
        app.gap_var.set("-5")
        assert app._gap_threshold() is None
        assert len(app.shown_errors) == 1

    def test_zero_returns_none_and_shows_error(self, app):
        app.gap_var.set("0")
        assert app._gap_threshold() is None
        assert len(app.shown_errors) == 1


class TestOnScanAbortsOnInvalidInput:
    def test_scan_does_not_start_a_thread_when_gap_threshold_invalid(self, app, monkeypatch, tmp_path):
        app.source_var.set(str(tmp_path))
        app.gap_var.set("garbage")
        monkeypatch.setattr(gui.shutil, "which", lambda name: f"/usr/bin/{name}")

        started = []
        monkeypatch.setattr(
            gui.threading, "Thread",
            lambda *a, **kw: type("FakeThread", (), {"start": lambda self: started.append(True)})(),
        )

        app._on_scan()

        assert started == []
        assert len(app.shown_errors) == 1

    def test_scan_does_not_start_a_thread_when_source_dir_missing(self, app, monkeypatch):
        app.source_var.set("")
        monkeypatch.setattr(gui.shutil, "which", lambda name: f"/usr/bin/{name}")

        started = []
        monkeypatch.setattr(
            gui.threading, "Thread",
            lambda *a, **kw: type("FakeThread", (), {"start": lambda self: started.append(True)})(),
        )

        app._on_scan()

        assert started == []


class TestPopulateTreeSizeUnavailable:
    def test_missing_clip_file_surfaces_as_a_tree_warning(self, app, tmp_path):
        present = tmp_path / "a.mp4"
        present.write_bytes(b"x" * 100)
        missing = tmp_path / "b.mp4"  # never created
        app.groups = [ClipGroup(clips=[make_clip(present), make_clip(missing)])]

        app._populate_tree()

        values = app.tree.item(app.tree.get_children()[0], "values")
        warnings_col = values[-1]
        assert "some clip sizes unavailable" in warnings_col


class TestMergeWorker:
    def test_merge_worker_calls_pipeline_and_signals_completion(self, app, monkeypatch, tmp_path):
        clip = make_clip(tmp_path / "a.mp4")
        group = ClipGroup(clips=[clip])

        captured = {}

        def fake_pipeline(groups, source_dir, dest_dir, **kwargs):
            captured["groups"] = groups
            captured["do_archive"] = kwargs["do_archive"]
            captured["do_stills"] = kwargs["do_stills"]
            kwargs["on_group_done"](groups[0])
            return None

        monkeypatch.setattr(gui, "run_merge_pipeline", fake_pipeline)

        app._merge_worker([group], tmp_path, tmp_path, True, False)

        assert captured["groups"] == [group]
        assert captured["do_archive"] is True
        assert captured["do_stills"] is False

        kinds = []
        while not app.queue.empty():
            kinds.append(app.queue.get_nowait()[0])
        assert "merged_group" in kinds
        assert "merge_done" in kinds

    def test_merge_worker_reports_exceptions_via_queue_not_by_crashing(self, app, monkeypatch, tmp_path):
        def fake_pipeline(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(gui, "run_merge_pipeline", fake_pipeline)

        app._merge_worker([], tmp_path, tmp_path, True, True)

        kind, message = app.queue.get_nowait()
        assert kind == "error"
        assert "boom" in message
