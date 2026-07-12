"""Minimal Tkinter GUI for dji_repack: scan a folder for split DJI Air3
clips, review the discovered groups, then merge (and archive) them.

Discovery/merge/archive all shell out to ffmpeg/ffprobe, so every button
handler below hands the actual work off to a background thread and polls
a queue on the Tk mainloop via root.after -- a synchronous call here would
freeze the window for as long as the subprocess takes.
"""

from __future__ import annotations

import queue
import shutil
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .archive import archive_merged_group, copy_lone_clip
from .procs import sweep_stale_partials
from .stills import copy_stills, discover_stills
from .video import (
    GAP_THRESHOLD_DEFAULT_S,
    ClipGroup,
    discover_clips,
    group_clips,
    group_part_index_gap_warning,
    group_summary,
    merge_group,
)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("DJI Repack")
        root.geometry("880x600")

        self.groups: list[ClipGroup] = []
        self.queue: queue.Queue = queue.Queue()
        self.busy = False

        self._build_widgets()
        self._poll_queue()

    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Source folder:").grid(row=0, column=0, sticky="w")
        self.source_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.source_var, width=60).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(top, text="Browse...", command=self._browse_source).grid(row=0, column=2)

        ttk.Label(top, text="Destination folder:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.dest_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.dest_var, width=60).grid(row=1, column=1, sticky="we", padx=4, pady=(4, 0))
        ttk.Button(top, text="Browse...", command=self._browse_dest).grid(row=1, column=2, pady=(4, 0))
        ttk.Label(top, text="(leave blank to merge in place, same as source)", foreground="#666").grid(
            row=2, column=1, sticky="w",
        )
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Gap threshold (s):").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.gap_var = tk.StringVar(value=str(int(GAP_THRESHOLD_DEFAULT_S)))
        ttk.Entry(top, textvariable=self.gap_var, width=10).grid(row=3, column=1, sticky="w", pady=(4, 0))

        self.archive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top, text="Archive merged source clips to _raw_splits/", variable=self.archive_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.stills_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top, text="Also copy still images (.dng/.jpg/.jpeg) to the destination", variable=self.stills_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w")

        btns = ttk.Frame(self.root, padding=(8, 0))
        btns.pack(fill="x")
        self.scan_btn = ttk.Button(btns, text="Scan", command=self._on_scan)
        self.scan_btn.pack(side="left")
        self.merge_all_btn = ttk.Button(btns, text="Merge All", command=self._on_merge_all, state="disabled")
        self.merge_all_btn.pack(side="left", padx=4)
        self.merge_selected_btn = ttk.Button(
            btns, text="Merge Selected", command=self._on_merge_selected, state="disabled",
        )
        self.merge_selected_btn.pack(side="left")

        mid = ttk.Frame(self.root, padding=8)
        mid.pack(fill="both", expand=True)

        columns = ("clips", "start", "duration", "size", "warnings")
        self.tree = ttk.Treeview(mid, columns=columns, show="headings", selectmode="extended", height=10)
        for col, label, width in (
            ("clips", "# clips", 60),
            ("start", "Start", 160),
            ("duration", "Duration (s)", 100),
            ("size", "Size (MB)", 90),
            ("warnings", "Warnings", 300),
        ):
            self.tree.heading(col, text=label)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        detail_frame = ttk.LabelFrame(self.root, text="Clips in selected group", padding=4)
        detail_frame.pack(fill="x", padx=8)
        self.detail = tk.Listbox(detail_frame, height=4)
        self.detail.pack(fill="x")

        log_frame = ttk.LabelFrame(self.root, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.log = tk.Text(log_frame, height=10, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True)

    def _browse_source(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.source_var.set(path)

    def _browse_dest(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.dest_var.set(path)

    def _log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.scan_btn.configure(state=state)
        merge_state = "disabled" if busy or not self.groups else "normal"
        self.merge_all_btn.configure(state=merge_state)
        self.merge_selected_btn.configure(state=merge_state)

    def _gap_threshold(self) -> float:
        try:
            return float(self.gap_var.get())
        except ValueError:
            return GAP_THRESHOLD_DEFAULT_S

    def _source_dir(self) -> Path | None:
        raw = self.source_var.get().strip()
        if not raw:
            messagebox.showerror("dji-repack", "Choose a source folder first.")
            return None
        path = Path(raw)
        if not path.is_dir():
            messagebox.showerror("dji-repack", f"Not a folder: {path}")
            return None
        return path

    def _dest_dir(self, source_dir: Path) -> Path:
        raw = self.dest_var.get().strip()
        return Path(raw) if raw else source_dir

    def _check_ffmpeg(self) -> bool:
        missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
        if missing:
            messagebox.showerror(
                "dji-repack", f"{', '.join(missing)} not found on PATH -- install ffmpeg first.",
            )
            return False
        return True

    def _on_select(self, _event=None) -> None:
        self.detail.delete(0, "end")
        selection = self.tree.selection()
        if not selection:
            return
        index = self.tree.index(selection[0])
        for name in group_summary(self.groups[index])["clip_names"]:
            self.detail.insert("end", name)

    def _on_scan(self) -> None:
        if self.busy or not self._check_ffmpeg():
            return
        source_dir = self._source_dir()
        if source_dir is None:
            return
        gap_threshold = self._gap_threshold()
        self._set_busy(True)
        self._log(f"scanning {source_dir} ...")
        threading.Thread(target=self._scan_worker, args=(source_dir, gap_threshold), daemon=True).start()

    def _scan_worker(self, source_dir: Path, gap_threshold: float) -> None:
        try:
            clips, warnings = discover_clips(source_dir)
            groups = group_clips(clips, gap_threshold_s=gap_threshold)
            stills = discover_stills(source_dir)
            self.queue.put(("scan_done", groups, warnings, len(stills)))
        except Exception as e:  # noqa: BLE001 -- surfaced to the log, not swallowed
            self.queue.put(("error", str(e)))

    def _on_merge_all(self) -> None:
        self._start_merge(list(range(len(self.groups))))

    def _on_merge_selected(self) -> None:
        indices = [self.tree.index(item) for item in self.tree.selection()]
        if not indices:
            messagebox.showinfo("dji-repack", "Select one or more groups first.")
            return
        self._start_merge(indices)

    def _start_merge(self, indices: list[int]) -> None:
        if self.busy or not self._check_ffmpeg():
            return
        source_dir = self._source_dir()
        if source_dir is None:
            return
        dest_dir = self._dest_dir(source_dir)
        groups = [self.groups[i] for i in indices]
        if not groups:
            messagebox.showinfo("dji-repack", "Nothing selected.")
            return
        self._set_busy(True)
        threading.Thread(
            target=self._merge_worker,
            args=(groups, source_dir, dest_dir, self.archive_var.get(), self.stills_var.get()),
            daemon=True,
        ).start()

    def _merge_worker(
        self, groups: list[ClipGroup], source_dir: Path, dest_dir: Path, do_archive: bool, do_stills: bool,
    ) -> None:
        try:
            swept = sweep_stale_partials(dest_dir)
            for name in swept:
                self.queue.put(("log", f"removed leftover partial: {name}"))

            for group in groups:
                if len(group.clips) < 2:
                    clip = group.clips[0]
                    start = time.perf_counter()
                    copy_warnings = copy_lone_clip(clip, dest_dir)
                    elapsed = time.perf_counter() - start
                    for w in copy_warnings:
                        self.queue.put(("log", f"warning: {w}"))
                    self.queue.put((
                        "log", f"copied single clip -> {dest_dir / clip.mp4_path.name} ({elapsed:.1f}s)",
                    ))
                    self.queue.put(("merged_group", group))
                    continue

                gap_warning = group_part_index_gap_warning(group)
                if gap_warning:
                    self.queue.put(("log", f"warning: {gap_warning}"))

                start = time.perf_counter()
                result = merge_group(group, dest_dir=dest_dir)
                elapsed = time.perf_counter() - start
                for w in result.warnings:
                    self.queue.put(("log", f"warning: {w}"))
                if not result.ok:
                    self.queue.put((
                        "log", f"FAILED: {', '.join(result.source_files)} -- {result.error} ({elapsed:.1f}s)",
                    ))
                    continue

                self.queue.put(("log", f"merged -> {result.output_path} ({elapsed:.1f}s)"))
                if do_archive:
                    for w in archive_merged_group(group, dest_dir):
                        self.queue.put(("log", f"warning: {w}"))
                self.queue.put(("merged_group", group))

            if do_stills:
                stills = discover_stills(source_dir)
                copied, skipped, still_warnings = copy_stills(
                    stills, dest_dir,
                    on_progress=lambda name, elapsed: self.queue.put(
                        ("log", f"  {name}: copied ({elapsed:.2f}s)"),
                    ),
                )
                for w in still_warnings:
                    self.queue.put(("log", f"warning: {w}"))
                self.queue.put(("log", f"stills: {copied} copied, {skipped} already present"))

            self.queue.put(("merge_done", None))
        except Exception as e:  # noqa: BLE001 -- surfaced to the log, not swallowed
            self.queue.put(("error", str(e)))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, *payload = self.queue.get_nowait()
                if kind == "scan_done":
                    groups, warnings, still_count = payload
                    self.groups = groups
                    for w in warnings:
                        self._log(f"warning: {w}")
                    self._populate_tree()
                    self._log(f"found {len(groups)} group(s), {still_count} still image(s)")
                    self._set_busy(False)
                elif kind == "merged_group":
                    (group,) = payload
                    if group in self.groups:
                        self.groups.remove(group)
                    self._populate_tree()
                elif kind == "merge_done":
                    self._log("merge pass complete")
                    self._set_busy(False)
                elif kind == "log":
                    (message,) = payload
                    self._log(message)
                elif kind == "error":
                    (message,) = payload
                    self._log(f"error: {message}")
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    def _populate_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for group in self.groups:
            summary = group_summary(group)
            warnings = []
            if summary["clip_count"] == 1:
                warnings.append("single clip, will be copied as-is")
            if summary["missing_srt"]:
                warnings.append(f"missing SRT: {', '.join(summary['missing_srt'])}")
            gap_warning = group_part_index_gap_warning(group)
            if gap_warning:
                warnings.append(gap_warning)
            self.tree.insert(
                "", "end",
                values=(
                    summary["clip_count"],
                    summary["start_dt"],
                    f"{summary['total_duration_s']:.1f}",
                    f"{summary['total_size_bytes'] / 1e6:.1f}",
                    "; ".join(warnings),
                ),
            )
        self._set_busy(self.busy)


def main(argv: list[str] | None = None) -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
