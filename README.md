# dji-repack

Reassembles DJI Air3 drone footage that got split into multiple sequential
`DJI_*.MP4` files (a FAT32 4GB file-size limit) back into one merged clip
per recording session, with the `.SRT` flight-telemetry sidecar merged back
in as an embedded subtitle track. Also pulls along still images (`.dng`,
`.jpg`, `.jpeg`) and any lone (unsplit) video clips, so a source folder or
card can be consolidated to a destination in one pass.

Extracted from [mneme](https://github.com/) so it can run standalone,
without the rest of mneme, for use alongside Lightroom or any other photo
tool.

## Requirements

- Python 3.12+
- `ffmpeg` and `ffprobe` on `PATH`

## Install

```sh
pip install -e '.[test]'
```

## Usage

### CLI

```sh
# Dry run: see what would be merged, without touching any files
dji-repack scan /path/to/card/or/folder

# Discover, merge, and archive originals into _raw_splits/
dji-repack merge /path/to/card/or/folder

# Merge into a different destination, leaving sources untouched
dji-repack merge /path/to/card/or/folder --dest /path/to/output --no-archive

# Video only, skip still images
dji-repack merge /path/to/card/or/folder --no-stills
```

Clips less than `--gap-threshold` seconds apart (default 300s) are treated
as one recording session and merged together; a bigger gap starts a new
output file. A group of only one clip has nothing to merge, so it's
copied to the destination as-is (original filename, untouched) instead.

Still images (`.dng`, `.jpg`, `.jpeg`, matched case-insensitively) found
anywhere under the source folder are copied — never moved — to the same
destination folder as the merged/copied video, flat (not preserving
subfolder structure). Rerunning is safe: a file already present at the
destination is left alone, not overwritten or re-copied. `--dest` can be
on a different volume than the source (e.g. copying straight off an SD
card) — video output always lands correctly, but the `--no-archive`
archive-into-`_raw_splits/` step only works within a single filesystem,
so pass `--no-archive` when merging across volumes to avoid a wall of
harmless-but-noisy per-clip warnings.

### GUI

```sh
dji-repack-gui
# or
python -m dji_repack
```

Pick a source folder (and optionally a different destination folder —
leave it blank to merge in place), click **Scan**, review the discovered
groups (clip count, start time, duration, size, warnings), then **Merge
All** or **Merge Selected**. Successfully-merged source clips move into
`_raw_splits/` inside the destination folder unless you uncheck the
archive option; still images copy along too unless you uncheck that
option.

## Tests

```sh
pytest
```

Unit tests run with no external dependencies. Integration tests
(`tests/test_integration.py`) exercise the real `ffmpeg`/`ffprobe`
subprocess path against synthetic clips and are skipped automatically if
`ffmpeg` isn't on `PATH`.
