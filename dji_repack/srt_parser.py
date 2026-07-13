"""
Parser for DJI Air3 .SRT telemetry sidecar files.

Each cue block looks like:

    1
    00:00:00,000 --> 00:00:00,016
    <font size="28">FrameCnt: 1, DiffTime: 16ms
    2026-06-29 10:27:07.249
    [iso: 210] [shutter: 1/5000.0] [fnum: 1.7] [ev: 0] [color_md : default] [focal_len: 24.00] [latitude: 41.620801] [longitude: -75.778972] [rel_alt: 36.400 abs_alt: 220.994] [ct: 5579] </font>

Every field observed in Air3-produced SRT files is captured below (iso,
shutter, fnum, ev, color_md, focal_len, latitude, longitude, rel_alt,
abs_alt, ct) -- none are silently dropped. There is exactly one regex for
the bracket line (no alternation, no fallback pattern) so parsing can't
silently diverge between cue blocks. This enumeration is drawn from a
single embedded sample block, not a corpus across firmware versions or
DJI's other (unrelated) camera models -- it is not a claim about "the DJI
SRT format" in general, some of which (e.g. zoom-lens models' digital
zoom ratio, or flight-telemetry fields like home distance/satellite count
on other DJI models) may carry fields this parser has never seen and
would raise SrtParseError on rather than silently accepting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re

TIMECODE_RE = re.compile(r"(\d+):(\d{2}):(\d{2}),(\d{3})")

HEADER_RE = re.compile(
    r"<font size=\"28\">FrameCnt:\s*(?P<framecnt>\d+),\s*DiffTime:\s*(?P<difftime_ms>\d+)ms"
)

DATETIME_RE = re.compile(
    r"(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
)

FIELDS_RE = re.compile(
    r"\[iso:\s*(?P<iso>\d+)\]\s*"
    r"\[shutter:\s*(?P<shutter>[^\]]+?)\s*\]\s*"
    r"\[fnum:\s*(?P<fnum>[\d.]+)\]\s*"
    r"\[ev:\s*(?P<ev>[-\d.]+)\]\s*"
    r"\[color_md\s*:\s*(?P<color_md>[^\]]+?)\s*\]\s*"
    r"\[focal_len:\s*(?P<focal_len>[\d.]+)\]\s*"
    r"(?:\[latitude:\s*(?P<latitude>[-\d.]+)\]\s*"
    r"\[longitude:\s*(?P<longitude>[-\d.]+)\]\s*"
    r"\[rel_alt:\s*(?P<rel_alt>[-\d.]+)\s+abs_alt:\s*(?P<abs_alt>[-\d.]+)\]\s*)?"
    r"\[ct:\s*(?P<ct>\d+)\]"
)
# The GPS block ([latitude]/[longitude]/[rel_alt abs_alt]) is captured as one
# atomic optional group, not four independently-optional fields: observed
# behavior on real Air3 footage is that DJI drops all four together during a
# GPS-lock dropout (e.g. the last ~12 frames before touchdown), never a
# partial subset. Treating them as independently optional would silently
# accept a malformed line where only some GPS fields are missing.


class SrtParseError(ValueError):
    def __init__(self, srt_path, block_text, reason):
        self.srt_path = srt_path
        self.block_text = block_text
        super().__init__(f"{srt_path}: {reason}\n--- block ---\n{block_text}\n-------------")


@dataclass
class SrtCue:
    framecnt: int
    difftime_ms: int
    cue_start_s: float
    cue_end_s: float
    wall_clock: datetime
    iso: int
    shutter: str
    fnum: float
    ev: float
    color_md: str
    focal_len: float
    latitude: float | None
    longitude: float | None
    rel_alt: float | None
    abs_alt: float | None
    ct: int
    raw_fields_text: str = ""  # the full bracket-fields source line,
        # verbatim, alongside the typed fields above -- so if DJI ever
        # inserts a field this parser doesn't know about (e.g. a
        # zoom-lens model's dzoom_ratio) between two known fields and
        # breaks FIELDS_RE, or firmware adds something new that FIELDS_RE
        # happens to still match around, the complete original text is
        # preserved rather than only the fields this parser recognizes.

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None


def _parse_timecode(tc: str) -> float:
    m = TIMECODE_RE.fullmatch(tc.strip())
    if not m:
        raise ValueError(f"unrecognized timecode: {tc!r}")
    h, mm, s, ms = (int(g) for g in m.groups())
    return h * 3600 + mm * 60 + s + ms / 1000.0


def parse_srt(srt_path) -> list[SrtCue]:
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    cues: list[SrtCue] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if len(lines) < 5:
            raise SrtParseError(srt_path, block, f"expected >=5 lines, got {len(lines)}")

        try:
            # This leading per-block sequential index is DJI's own SRT
            # cue counter; validated here only as "well-formed integer"
            # and then intentionally discarded (dropped-with-reason, per
            # this module's data-capture disposition convention -- see
            # the module docstring, which scopes its own "none are
            # silently dropped" claim to the bracket-fields line only,
            # not this one): cue ordering and merge timing both come from
            # the timecode line and cumulative_offset math in video.py,
            # not from this counter, and it is never cross-checked
            # against FrameCnt (parsed two lines below). A real
            # divergence between the two -- a firmware or reordering
            # glitch producing a skipped/duplicated cue index while
            # FrameCnt keeps incrementing, or vice versa -- currently
            # passes through unnoticed.
            int(lines[0].strip())
        except ValueError as e:
            raise SrtParseError(srt_path, block, f"bad cue index: {e}") from e

        if "-->" not in lines[1]:
            raise SrtParseError(srt_path, block, "missing '-->' timecode line")
        try:
            # split() (not split(..., 1)) is deliberate: a line with a
            # second stray "-->" must fail loudly here (too many values to
            # unpack) rather than silently taking just the first two
            # pieces of a malformed line.
            start_tc, end_tc = lines[1].split("-->")
            cue_start_s = _parse_timecode(start_tc)
            cue_end_s = _parse_timecode(end_tc)
        except ValueError as e:
            # Re-raise as SrtParseError (not a bare ValueError) so every
            # malformed-input path in this function shares one exception
            # type -- callers (discover_clips) only need to catch
            # SrtParseError to turn any parse failure into a per-clip
            # warning instead of an uncaught crash. Wrapping the split()
            # itself (not just _parse_timecode) means a line with a
            # second stray "-->" (too many values to unpack) is caught
            # here too, not just a bad timecode value.
            raise SrtParseError(srt_path, block, str(e)) from e
        if cue_end_s < cue_start_s:
            # An inverted cue (end before start) would otherwise parse
            # cleanly and flow straight into merge.py's cumulative_offset
            # arithmetic as a negative-duration cue, corrupting every
            # subsequent cue's timestamp in the merged subtitle track.
            raise SrtParseError(
                srt_path, block,
                f"cue end time ({end_tc.strip()}) is before start time ({start_tc.strip()})",
            )

        header_m = HEADER_RE.search(lines[2])
        if not header_m:
            raise SrtParseError(srt_path, block, "FrameCnt/DiffTime header didn't match")

        dt_m = DATETIME_RE.search(lines[3])
        if not dt_m:
            raise SrtParseError(srt_path, block, "wall-clock datetime line didn't match")
        wall_clock = datetime.strptime(dt_m.group("datetime"), "%Y-%m-%d %H:%M:%S.%f")

        fields_text = " ".join(lines[4:])
        fields_m = FIELDS_RE.search(fields_text)
        if not fields_m:
            raise SrtParseError(srt_path, block, "bracket-fields line didn't match")

        lat_raw = fields_m.group("latitude")

        try:
            cue = SrtCue(
                framecnt=int(header_m.group("framecnt")),
                difftime_ms=int(header_m.group("difftime_ms")),
                cue_start_s=cue_start_s,
                cue_end_s=cue_end_s,
                wall_clock=wall_clock,
                iso=int(fields_m.group("iso")),
                shutter=fields_m.group("shutter"),
                fnum=float(fields_m.group("fnum")),
                ev=float(fields_m.group("ev")),
                color_md=fields_m.group("color_md"),
                focal_len=float(fields_m.group("focal_len")),
                latitude=float(lat_raw) if lat_raw is not None else None,
                longitude=float(fields_m.group("longitude")) if lat_raw is not None else None,
                rel_alt=float(fields_m.group("rel_alt")) if lat_raw is not None else None,
                abs_alt=float(fields_m.group("abs_alt")) if lat_raw is not None else None,
                ct=int(fields_m.group("ct")),
                raw_fields_text=fields_text.strip(),
            )
        except (ValueError, TypeError) as e:
            # A field can match FIELDS_RE's character-class regex (e.g.
            # ev's `[-\d.]+` matching a malformed "--0") and still fail
            # int()/float() conversion -- that must not escape as a bare
            # ValueError/TypeError past this function's SrtParseError
            # contract, which discover_clips relies on to isolate one
            # corrupt file from aborting the whole card.
            raise SrtParseError(srt_path, block, f"numeric field conversion failed: {e}") from e
        cues.append(cue)
    return cues


def format_cue_block(index: int, start_s: float, end_s: float, cue: SrtCue) -> str:
    def fmt_tc(s: float) -> str:
        # The CRITICAL regression: rounding h/m/sec/ms independently from
        # the float seconds value meant a millisecond rounding up to 1000
        # only cascaded one level (sec += 1), never further into minutes
        # or hours -- fmt_tc(3599.9996) used to emit the invalid
        # "00:59:60,000" instead of "01:00:00,000". Converting to a single
        # integer millisecond count up front and deriving every field via
        # floor-division/modulo on that one integer makes a partial
        # cascade structurally impossible: there's only one rounding step,
        # not four independent ones that can disagree.
        total_ms = max(0, round(s * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        sec = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    gps = (
        f"[latitude: {cue.latitude:.6f}] [longitude: {cue.longitude:.6f}] "
        f"[rel_alt: {cue.rel_alt:.3f} abs_alt: {cue.abs_alt:.3f}] "
        if cue.has_gps else ""
    )
    return (
        f"{index}\n"
        f"{fmt_tc(start_s)} --> {fmt_tc(end_s)}\n"
        f"<font size=\"28\">FrameCnt: {index}, DiffTime: {cue.difftime_ms}ms\n"
        f"{cue.wall_clock.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
        f"[iso: {cue.iso}] [shutter: {cue.shutter}] [fnum: {cue.fnum}] [ev: {cue.ev:g}] "
        f"[color_md : {cue.color_md}] [focal_len: {cue.focal_len:.2f}] "
        f"{gps}[ct: {cue.ct}] </font>\n"
    )
