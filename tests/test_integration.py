"""Real ffmpeg/ffprobe integration tests for dji_repack.video.

The pure-logic tests in test_video.py construct Clip objects directly and
never touch a real subprocess -- these exercise the real subprocess path
end to end (synthetic clips generated via ffmpeg's lavfi test sources, not
real drone footage, but the same probe_clip -> merge_group -> real
ffprobe-of-the-output round trip).

Ported from mneme's tests/test_repack_integration.py (TestRealVideoProbeAndMerge only).
"""

import subprocess
from pathlib import Path

import pytest

from dji_repack import procs, video


def _has(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "-version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


HAS_FFMPEG = _has("ffmpeg")


def _make_test_mp4(path: Path, duration_s: float = 1.0, size="320x240", rate=30):
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}",
            "-t", str(duration_s), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _make_test_mp4_with_audio(path: Path, duration_s: float = 1.0, size="320x240", rate=30):
    """Same as _make_test_mp4, but with a camera-mic-style AAC audio track
    -- exercises the audio probe/preserve-through-merge path, which a
    video-only clip (the default fixture) can't."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", str(duration_s), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ac", "2",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not available")
class TestRealVideoProbeAndMerge:
    def test_probe_clip_reads_real_ffprobe_output(self, tmp_path):
        clip_path = tmp_path / "a.mp4"
        _make_test_mp4(clip_path, duration_s=1.0)

        probe = video.probe_clip(clip_path)

        assert probe.codec_name == "h264"
        assert probe.width == 320
        assert probe.height == 240
        assert probe.duration_s == pytest.approx(1.0, abs=0.2)
        assert probe.has_audio is False

    def test_probe_clip_detects_a_real_audio_track(self, tmp_path):
        """The MEDIUM regression: a clip's camera-mic audio track was
        never probed at all -- has_audio must be True and the real codec/
        rate/channel values must come through, not just a placeholder."""
        clip_path = tmp_path / "a.mp4"
        _make_test_mp4_with_audio(clip_path, duration_s=1.0)

        probe = video.probe_clip(clip_path)

        assert probe.has_audio is True
        assert probe.audio_codec_name == "aac"
        assert probe.audio_sample_rate == 48000
        assert probe.audio_channels == 2

    def test_merge_group_preserves_audio_through_a_real_concat(self, tmp_path):
        """The MEDIUM regression, exercised end to end: merge_group used to
        map only 0:v:0, silently dropping every clip's audio track on
        every merge. The merged output must now carry an audio stream
        too."""
        from datetime import datetime, timedelta

        a_path, b_path = tmp_path / "a.mp4", tmp_path / "b.mp4"
        _make_test_mp4_with_audio(a_path, duration_s=1.0)
        _make_test_mp4_with_audio(b_path, duration_s=1.0)

        probe_a, probe_b = video.probe_clip(a_path), video.probe_clip(b_path)
        base = datetime(2026, 1, 1, 12, 0, 0)
        clip_a = video.Clip(
            mp4_path=a_path, srt_path=None, cues=[], srt_error="no .SRT sidecar found",
            probe=probe_a, start_dt=base, end_dt=base + timedelta(seconds=probe_a.duration_s),
            start_is_estimated=True,
        )
        clip_b = video.Clip(
            mp4_path=b_path, srt_path=None, cues=[], srt_error="no .SRT sidecar found",
            probe=probe_b, start_dt=clip_a.end_dt,
            end_dt=clip_a.end_dt + timedelta(seconds=probe_b.duration_s),
            start_is_estimated=True,
        )
        group = video.ClipGroup(clips=[clip_a, clip_b])

        result = video.merge_group(group, tmp_path / "out")

        assert result.ok, result.error
        merged_probe = video.probe_clip(result.output_path)
        assert merged_probe.has_audio is True
        assert merged_probe.audio_codec_name == "aac"
        assert merged_probe.duration_s == pytest.approx(2.0, abs=0.3)

    def test_merge_group_refuses_audio_presence_mismatch_within_group(self, tmp_path):
        """A group with one audio-bearing clip and one silent clip can't
        be safely mapped with a single 0:a:0 stream -- must be refused the
        same way a resolution mismatch is, not silently merged with
        whatever ffmpeg happens to do."""
        from datetime import datetime, timedelta

        a_path, b_path = tmp_path / "a.mp4", tmp_path / "b.mp4"
        _make_test_mp4_with_audio(a_path, duration_s=1.0)
        _make_test_mp4(b_path, duration_s=1.0)  # no audio

        probe_a, probe_b = video.probe_clip(a_path), video.probe_clip(b_path)
        base = datetime(2026, 1, 1, 12, 0, 0)
        clip_a = video.Clip(
            mp4_path=a_path, srt_path=None, cues=[], srt_error="no .SRT sidecar found",
            probe=probe_a, start_dt=base, end_dt=base + timedelta(seconds=probe_a.duration_s),
            start_is_estimated=True,
        )
        clip_b = video.Clip(
            mp4_path=b_path, srt_path=None, cues=[], srt_error="no .SRT sidecar found",
            probe=probe_b, start_dt=clip_a.end_dt,
            end_dt=clip_a.end_dt + timedelta(seconds=probe_b.duration_s),
            start_is_estimated=True,
        )
        group = video.ClipGroup(clips=[clip_a, clip_b])

        result = video.merge_group(group, tmp_path / "out")

        assert result.ok is False
        assert "mismatch" in result.error

    def test_discover_clips_skips_files_with_no_srt_and_no_dji_filename_timestamp(self, tmp_path):
        # Confirms the real (not assumed) behavior for a synthetic clip that
        # is neither DJI-named nor has an SRT sidecar: both signals for a
        # start time are absent, so discover_clips skips it with a warning
        # rather than fabricating a timestamp.
        _make_test_mp4(tmp_path / "a.mp4", duration_s=1.0)
        clips, warnings = video.discover_clips(tmp_path)
        assert clips == []
        assert procs.count_skipped(warnings) == 1

    def test_merge_group_concatenates_two_real_clips_without_reencoding(self, tmp_path):
        a_path, b_path = tmp_path / "a.mp4", tmp_path / "b.mp4"
        _make_test_mp4(a_path, duration_s=1.0)
        _make_test_mp4(b_path, duration_s=1.0)

        # discover_clips can't derive a start time for these (no SRT
        # sidecar, no DJI-style filename) -- build the group manually
        # instead, mirroring how the pure-logic tests construct Clip
        # objects directly; merge_group itself doesn't care where its
        # ClipGroup came from.
        probe_a, probe_b = video.probe_clip(a_path), video.probe_clip(b_path)
        from datetime import datetime, timedelta

        base = datetime(2026, 1, 1, 12, 0, 0)
        clip_a = video.Clip(
            mp4_path=a_path, srt_path=None, cues=[], srt_error="no .SRT sidecar found",
            probe=probe_a, start_dt=base, end_dt=base + timedelta(seconds=probe_a.duration_s),
            start_is_estimated=True,
        )
        clip_b = video.Clip(
            mp4_path=b_path, srt_path=None, cues=[], srt_error="no .SRT sidecar found",
            probe=probe_b, start_dt=clip_a.end_dt,
            end_dt=clip_a.end_dt + timedelta(seconds=probe_b.duration_s),
            start_is_estimated=True,
        )
        group = video.ClipGroup(clips=[clip_a, clip_b])

        result = video.merge_group(group, tmp_path / "out")

        assert result.ok, result.error
        assert result.output_path.exists()
        merged_probe = video.probe_clip(result.output_path)
        assert merged_probe.duration_s == pytest.approx(2.0, abs=0.3)
        assert merged_probe.codec_name == "h264"

    def test_merge_group_refuses_mismatched_resolution_without_touching_ffmpeg(self, tmp_path):
        a_path, b_path = tmp_path / "a.mp4", tmp_path / "b.mp4"
        _make_test_mp4(a_path, duration_s=1.0, size="320x240")
        _make_test_mp4(b_path, duration_s=1.0, size="640x480")

        from datetime import datetime

        base = datetime(2026, 1, 1, 12, 0, 0)
        probe_a, probe_b = video.probe_clip(a_path), video.probe_clip(b_path)
        clip_a = video.Clip(mp4_path=a_path, srt_path=None, cues=[], srt_error=None,
                             probe=probe_a, start_dt=base, end_dt=base, start_is_estimated=True)
        clip_b = video.Clip(mp4_path=b_path, srt_path=None, cues=[], srt_error=None,
                             probe=probe_b, start_dt=base, end_dt=base, start_is_estimated=True)

        result = video.merge_group(video.ClipGroup(clips=[clip_a, clip_b]), tmp_path / "out")

        assert result.ok is False
        assert "mismatch" in result.error
        assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []

    def _three_clip_group(self, tmp_path):
        from datetime import datetime, timedelta

        a_path, b_path, c_path = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"
        _make_test_mp4(a_path, duration_s=1.0)
        _make_test_mp4(b_path, duration_s=1.0)
        _make_test_mp4(c_path, duration_s=1.0)
        probe_a, probe_b, probe_c = (video.probe_clip(p) for p in (a_path, b_path, c_path))
        base = datetime(2026, 1, 1, 12, 0, 0)
        clip_a = video.Clip(mp4_path=a_path, srt_path=None, cues=[], srt_error=None,
                             probe=probe_a, start_dt=base,
                             end_dt=base + timedelta(seconds=probe_a.duration_s), start_is_estimated=True)
        clip_b = video.Clip(mp4_path=b_path, srt_path=None, cues=[], srt_error=None,
                             probe=probe_b, start_dt=clip_a.end_dt,
                             end_dt=clip_a.end_dt + timedelta(seconds=probe_b.duration_s), start_is_estimated=True)
        clip_c = video.Clip(mp4_path=c_path, srt_path=None, cues=[], srt_error=None,
                             probe=probe_c, start_dt=clip_b.end_dt,
                             end_dt=clip_b.end_dt + timedelta(seconds=probe_c.duration_s), start_is_estimated=True)
        return video.ClipGroup(clips=[clip_a, clip_b, clip_c])

    def test_merge_group_fails_when_a_middle_clip_disappears_before_the_concat_runs(self, tmp_path):
        """The data-integrity gap found during full-review testing: ffmpeg's
        concat demuxer exits 0 even when a listed input vanishes mid-list --
        it logs "Error during demuxing" to stderr and silently proceeds
        without that segment. Confirmed by direct reproduction (ffmpeg
        8.1.1) before this fix existed: result.ok came back True with a
        truncated 1-clip-worth-of-duration output and the demuxing error
        buried in warnings. merge_group must now detect this marker and
        report failure instead of a silently-incomplete "success"."""
        group = self._three_clip_group(tmp_path)
        group.clips[1].mp4_path.unlink()

        result = video.merge_group(group, tmp_path / "out")

        assert result.ok is False
        assert "demux" in result.error.lower()
        assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []

    def test_merge_group_fails_when_the_last_clip_disappears_before_the_concat_runs(self, tmp_path):
        """Same failure mode, different position in the list -- confirmed
        during reproduction that a missing LAST clip also exits 0 (only a
        missing FIRST clip exits nonzero, via a different ffmpeg code path
        already caught by the ordinary RuntimeError branch)."""
        group = self._three_clip_group(tmp_path)
        group.clips[2].mp4_path.unlink()

        result = video.merge_group(group, tmp_path / "out")

        assert result.ok is False
        assert "demux" in result.error.lower()

    def test_merge_group_fails_when_the_first_clip_disappears_before_the_concat_runs(self, tmp_path):
        """Boundary companion to the two tests above: a missing FIRST clip
        takes ffmpeg's other, nonzero-exit failure path (confirmed by
        reproduction) -- must still be reported as a failure, just via the
        pre-existing RuntimeError branch rather than the new demux-marker
        check."""
        group = self._three_clip_group(tmp_path)
        group.clips[0].mp4_path.unlink()

        result = video.merge_group(group, tmp_path / "out")

        assert result.ok is False

    def test_merge_group_succeeds_when_nothing_is_missing(self, tmp_path):
        """Companion to the failure-injection tests above: confirms the
        demux-marker check doesn't false-positive on an ordinary, complete
        merge of the same three-clip shape."""
        group = self._three_clip_group(tmp_path)

        result = video.merge_group(group, tmp_path / "out")

        assert result.ok, result.error
        merged_probe = video.probe_clip(result.output_path)
        assert merged_probe.duration_s == pytest.approx(3.0, abs=0.3)
