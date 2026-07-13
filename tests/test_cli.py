"""Tests for dji_repack.cli -- the `dji-repack scan|merge` entry point.

The CRITICAL/HIGH gap this closes: cli.py previously had zero test
coverage of any kind. discover_clips/group_clips/run_merge_pipeline are
monkeypatched at the cli module boundary so these tests exercise the
CLI's own argument-parsing and wiring (which flag maps to which pipeline
call, what gets printed, what exit code comes back) without needing a
real ffmpeg install -- pipeline.py's own orchestration logic is already
covered by test_pipeline.py, and merge_group's real ffmpeg behavior by
test_integration.py.
"""

from datetime import datetime

import pytest

from dji_repack import cli
from dji_repack.pipeline import MergePipelineResult
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


def make_clip(mp4_path):
    return Clip(
        mp4_path=mp4_path, srt_path=None, cues=[], srt_error=None,
        probe=make_probe(), start_dt=BASE, end_dt=BASE, start_is_estimated=True,
    )


@pytest.fixture(autouse=True)
def _fake_ffmpeg_present(monkeypatch):
    # Keep these tests independent of whether ffmpeg is actually installed
    # on the machine running the suite -- CLI wiring is what's under test
    # here, not ffmpeg itself (see test_integration.py for that).
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")


class TestBuildParser:
    def test_scan_defaults(self):
        args = cli.build_parser().parse_args(["scan", "/some/dir"])
        assert args.source_dir == "/some/dir"
        assert args.gap_threshold == cli.GAP_THRESHOLD_DEFAULT_S
        assert args.no_stills is False

    def test_merge_defaults(self):
        args = cli.build_parser().parse_args(["merge", "/some/dir"])
        assert args.dest is None
        assert args.no_archive is False
        assert args.no_stills is False

    def test_merge_flags_parsed(self):
        args = cli.build_parser().parse_args([
            "merge", "/some/dir", "--dest", "/out", "--gap-threshold", "60",
            "--no-archive", "--no-stills",
        ])
        assert args.dest == "/out"
        assert args.gap_threshold == 60.0
        assert args.no_archive is True
        assert args.no_stills is True


class TestCmdFfmpegMissing:
    def test_missing_ffmpeg_exits_nonzero(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli.shutil, "which", lambda name: None)
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["scan", str(tmp_path)])
        assert exc_info.value.code == 1


class TestCmdScan:
    def test_no_clips_found_is_reported(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "discover_clips", lambda source_dir: ([], []))
        monkeypatch.setattr(cli, "group_clips", lambda clips, gap_threshold_s: [])

        rc = cli.main(["scan", str(tmp_path), "--no-stills"])

        assert rc == 0
        assert "no video clips found" in capsys.readouterr().out

    def test_group_details_are_printed(self, monkeypatch, tmp_path, capsys):
        clip = make_clip(tmp_path / "a.mp4")
        group = ClipGroup(clips=[clip])
        monkeypatch.setattr(cli, "discover_clips", lambda source_dir: ([clip], ["some warning"]))
        monkeypatch.setattr(cli, "group_clips", lambda clips, gap_threshold_s: [group])

        rc = cli.main(["scan", str(tmp_path), "--no-stills"])
        out, err = capsys.readouterr()

        assert rc == 0
        assert "1 clip(s)" in out
        assert "a.mp4" in out
        assert "some warning" in err

    def test_stills_count_reported_and_warnings_go_to_stderr(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "discover_clips", lambda source_dir: ([], []))
        monkeypatch.setattr(cli, "group_clips", lambda clips, gap_threshold_s: [])
        monkeypatch.setattr(
            cli, "discover_stills", lambda source_dir: ([tmp_path / "a.dng"], ["bad.dng: skipped -- boom"]),
        )

        rc = cli.main(["scan", str(tmp_path)])
        out, err = capsys.readouterr()

        assert rc == 0
        assert "1 still image(s) found" in out
        assert "bad.dng" in err


class TestCmdMerge:
    def test_no_clips_found_still_runs_the_stills_pass(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "discover_clips", lambda source_dir: ([], []))
        calls = {}

        def fake_pipeline(groups, source_dir, dest_dir, **kwargs):
            calls["groups"] = groups
            calls["kwargs"] = kwargs
            return MergePipelineResult(stills_copied=2, stills_skipped=1)

        monkeypatch.setattr(cli, "run_merge_pipeline", fake_pipeline)

        rc = cli.main(["merge", str(tmp_path)])
        out, _err = capsys.readouterr()

        assert rc == 0
        assert calls["groups"] == []
        assert "no video clips found" in out
        assert "done: 0 merged, 0 single clip(s) copied, 0 failed, 2 still(s) copied (1 already present)" in out

    def test_no_archive_and_no_stills_flags_forwarded_to_pipeline(self, monkeypatch, tmp_path):
        clip = make_clip(tmp_path / "a.mp4")
        group = ClipGroup(clips=[clip])
        monkeypatch.setattr(cli, "discover_clips", lambda source_dir: ([clip], []))
        monkeypatch.setattr(cli, "group_clips", lambda clips, gap_threshold_s: [group])

        captured = {}

        def fake_pipeline(groups, source_dir, dest_dir, **kwargs):
            captured["do_archive"] = kwargs["do_archive"]
            captured["do_stills"] = kwargs["do_stills"]
            captured["dest_dir"] = dest_dir
            return MergePipelineResult()

        monkeypatch.setattr(cli, "run_merge_pipeline", fake_pipeline)

        cli.main(["merge", str(tmp_path), "--no-archive", "--no-stills", "--dest", str(tmp_path / "out")])

        assert captured["do_archive"] is False
        assert captured["do_stills"] is False
        assert captured["dest_dir"] == tmp_path / "out"

    def test_failed_groups_yield_nonzero_exit_code(self, monkeypatch, tmp_path):
        clip = make_clip(tmp_path / "a.mp4")
        group = ClipGroup(clips=[clip])
        monkeypatch.setattr(cli, "discover_clips", lambda source_dir: ([clip], []))
        monkeypatch.setattr(cli, "group_clips", lambda clips, gap_threshold_s: [group])

        def fake_pipeline(groups, source_dir, dest_dir, **kwargs):
            result = MergePipelineResult()
            from dji_repack.pipeline import GroupOutcome
            result.group_outcomes.append(GroupOutcome(group, "failed", error="boom"))
            return result

        monkeypatch.setattr(cli, "run_merge_pipeline", fake_pipeline)

        rc = cli.main(["merge", str(tmp_path)])

        assert rc == 1

    def test_dest_defaults_to_source_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "discover_clips", lambda source_dir: ([], []))
        captured = {}

        def fake_pipeline(groups, source_dir, dest_dir, **kwargs):
            captured["dest_dir"] = dest_dir
            return MergePipelineResult()

        monkeypatch.setattr(cli, "run_merge_pipeline", fake_pipeline)

        cli.main(["merge", str(tmp_path)])

        assert captured["dest_dir"] == tmp_path
