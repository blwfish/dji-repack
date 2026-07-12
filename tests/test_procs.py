"""Tests for dji_repack.procs -- subprocess/path helpers shared by video.py.

Ported from mneme's tests/test_repack_procs.py.
"""

import pytest

from dji_repack import procs


class TestCountSkipped:
    def test_empty_list_is_zero(self):
        assert procs.count_skipped([]) == 0

    def test_counts_only_dropped_clip_messages(self):
        warnings = [
            "a.mp4: skipped during scan -- permission denied",
            "b.mp4: no .SRT sidecar found; grouping and metadata for this clip fall back to filename timestamp only",
            "c.mp4: skipped -- ffprobe failed: corrupt file",
        ]
        assert procs.count_skipped(warnings) == 2

    def test_all_informational_is_zero(self):
        warnings = ["b.mp4: no .SRT sidecar found; grouping and metadata for this clip fall back to filename timestamp only"]
        assert procs.count_skipped(warnings) == 0


class TestRun:
    def test_empty_command_list_raises_runtime_error_not_index_error(self):
        """The MEDIUM regression: subprocess.run([]) raises a bare
        IndexError from deep inside the subprocess module, before this
        function's own error handling ever runs -- every other call site
        in this package only catches RuntimeError."""
        with pytest.raises(RuntimeError):
            procs.run([])

    def test_missing_binary_raises_runtime_error_not_file_not_found_error(self):
        """The MEDIUM regression: subprocess.run raises FileNotFoundError
        (not a RuntimeError subclass) when the binary isn't on PATH --
        callers everywhere in this package only catch RuntimeError, so a
        missing ffmpeg/ffprobe install used to crash outright instead of
        degrading the same way a nonzero-exit failure does."""
        with pytest.raises(RuntimeError, match="not found"):
            procs.run(["definitely_not_a_real_binary_xyz123"])

    def test_nonzero_exit_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            procs.run(["python3", "-c", "import sys; sys.exit(1)"])

    def test_successful_stdout_is_returned(self):
        out = procs.run(["python3", "-c", "print('hello')"])
        assert out.strip() == "hello"

    def test_stderr_on_success_is_appended_to_warnings(self):
        warnings = []
        procs.run(["python3", "-c", "import sys; sys.stderr.write('warn text')"], warnings)
        assert any("warn text" in w for w in warnings)

    def test_no_warnings_list_means_stderr_is_just_discarded_not_an_error(self):
        # must not raise even though stderr has content and no warnings list was given
        procs.run(["python3", "-c", "import sys; sys.stderr.write('warn text')"])

    def test_a_hung_command_raises_runtime_error_not_timeout_expired(self):
        """The LOW regression: no timeout meant a hung ffmpeg/ffprobe call
        (a corrupt input triggering a decode stall, a stalled network-
        mounted source) blocked the whole batch indefinitely. Every
        caller in this package only catches RuntimeError, same as the
        FileNotFoundError/empty-command conversions above."""
        with pytest.raises(RuntimeError, match="timed out"):
            procs.run(["python3", "-c", "import time; time.sleep(5)"], timeout=0.2)

    def test_a_command_finishing_within_the_timeout_is_unaffected(self):
        out = procs.run(["python3", "-c", "print('hello')"], timeout=30)
        assert out.strip() == "hello"

    def test_non_utf8_stderr_degrades_gracefully_instead_of_raising(self):
        """errors="replace" (not the subprocess default "strict") --
        stderr is diagnostic text only ever logged, never parsed, so a
        stray non-UTF-8 byte must not raise UnicodeDecodeError, which
        callers can't catch via `except RuntimeError`."""
        warnings = []
        procs.run(
            ["python3", "-c", "import sys; sys.stderr.buffer.write(b'bad byte: \\xff')"],
            warnings,
        )
        assert any("bad byte" in w for w in warnings)


class TestConcatDemuxError:
    """ffmpeg's concat demuxer exits 0 even when a listed input failed
    mid-run -- callers must scan captured stderr for this marker instead
    of trusting the return code alone."""

    def test_no_warnings_returns_none(self):
        assert procs.concat_demux_error([], 0) is None

    def test_marker_present_returns_the_offending_entry(self):
        warnings = ["ffmpeg warning: [in#0/concat] Impossible to open 'b.mp4'\n"
                    "[in#0/concat] Error during demuxing: No such file or directory"]
        result = procs.concat_demux_error(warnings, 0)
        assert result is not None
        assert "Error during demuxing" in result

    def test_marker_absent_returns_none(self):
        warnings = ["ffmpeg warning: [mov,mp4,m4a @ 0x1] some harmless timestamp note"]
        assert procs.concat_demux_error(warnings, 0) is None

    def test_since_index_ignores_warnings_before_this_call(self):
        """A prior, unrelated ffmpeg/ffprobe call's stderr (e.g. from
        probing a clip earlier in the pipeline) must not be mistaken for
        this concat run's own failure."""
        warnings = ["ffmpeg warning: Error during demuxing: unrelated prior call"]
        since_index = len(warnings)
        warnings.append("ffmpeg warning: clean run, nothing wrong")
        assert procs.concat_demux_error(warnings, since_index) is None

    def test_since_index_still_finds_a_marker_added_after_it(self):
        warnings = ["ffmpeg warning: unrelated prior call, all fine"]
        since_index = len(warnings)
        warnings.append("ffmpeg warning: Error during demuxing: No such file or directory")
        result = procs.concat_demux_error(warnings, since_index)
        assert result is not None


class TestConcatDurationShortfall:
    """Marker-independent backstop for TestConcatDemuxError above: a
    dropped clip changes the merged output's own probed duration
    regardless of whether ffmpeg's stderr happened to match
    CONCAT_DEMUX_ERROR_MARKER for this build/version/locale."""

    def test_exact_match_is_not_a_shortfall(self):
        assert procs.concat_duration_shortfall(100.0, 100.0) is False

    def test_within_absolute_tolerance_is_not_a_shortfall(self):
        # default abs_tolerance_s=2.0; short by exactly 2.0s is the
        # boundary itself -- must NOT trip (strict `>`, not `>=`).
        assert procs.concat_duration_shortfall(100.0, 98.0) is False

    def test_just_over_absolute_tolerance_is_a_shortfall(self):
        assert procs.concat_duration_shortfall(100.0, 100.0 - 2.0 - 0.01) is True

    def test_relative_tolerance_dominates_for_large_expected_durations(self):
        # 1% of 1000s = 10s > the 2s absolute floor -- short by 9s must
        # NOT trip; short by 11s must.
        assert procs.concat_duration_shortfall(1000.0, 991.0) is False
        assert procs.concat_duration_shortfall(1000.0, 989.0) is True

    def test_a_whole_missing_clip_trips_it(self):
        # Three ~10s clips merged, one silently dropped -- output is ~10s
        # short of the ~30s expected, far past either tolerance.
        assert procs.concat_duration_shortfall(30.0, 20.0) is True

    def test_zero_expected_duration_does_not_divide_by_zero(self):
        assert procs.concat_duration_shortfall(0.0, 0.0) is False


class TestQuoteConcatPath:
    def test_simple_path(self, tmp_path):
        p = tmp_path / "clip.mp4"
        p.touch()
        assert procs.quote_concat_path(p) == f"file '{p.resolve()}'"

    def test_embedded_single_quote_is_escaped(self, tmp_path):
        p = tmp_path / "it's a clip.mp4"
        p.touch()
        quoted = procs.quote_concat_path(p)
        assert quoted == f"file '{str(p.resolve()).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
        # decoded back through a shell-like split, the escaped quote survives
        assert "it'\\''s a clip.mp4" in quoted


class TestNextAvailablePath:
    def test_no_collision_returns_the_original_path(self, tmp_path):
        target = tmp_path / "out.mp4"
        assert procs.next_available_path(target) == target

    def test_one_collision_appends_v2(self, tmp_path):
        target = tmp_path / "out.mp4"
        target.touch()
        assert procs.next_available_path(target) == tmp_path / "out_v2.mp4"

    def test_two_collisions_appends_v3(self, tmp_path):
        target = tmp_path / "out.mp4"
        target.touch()
        (tmp_path / "out_v2.mp4").touch()
        assert procs.next_available_path(target) == tmp_path / "out_v3.mp4"


class TestFfprobeInt:
    def test_none_returns_default(self):
        assert procs.ffprobe_int(None) is None
        assert procs.ffprobe_int(None, default=0) == 0

    def test_na_sentinel_returns_default(self):
        assert procs.ffprobe_int("N/A") is None

    def test_numeric_string_is_parsed(self):
        assert procs.ffprobe_int("48000") == 48000

    def test_float_string_is_truncated_not_rounded_via_int_of_float(self):
        assert procs.ffprobe_int("48000.7") == 48000

    def test_real_int_passes_through(self):
        assert procs.ffprobe_int(48000) == 48000

    def test_zero_is_preserved_not_treated_as_missing(self):
        """Distinct from the `or 0` anti-pattern this replaces: an actual
        0 value must round-trip as 0, not collapse indistinguishably with
        a missing/None field."""
        assert procs.ffprobe_int(0) == 0
        assert procs.ffprobe_int("0") == 0


class TestFfprobeFloat:
    def test_none_returns_default(self):
        assert procs.ffprobe_float(None) is None

    def test_na_sentinel_returns_default(self):
        assert procs.ffprobe_float("N/A") is None

    def test_numeric_string_is_parsed(self):
        assert procs.ffprobe_float("12.345") == pytest.approx(12.345)

    def test_real_float_passes_through(self):
        assert procs.ffprobe_float(12.345) == pytest.approx(12.345)

    def test_zero_is_preserved_not_treated_as_missing(self):
        assert procs.ffprobe_float(0.0) == 0.0
