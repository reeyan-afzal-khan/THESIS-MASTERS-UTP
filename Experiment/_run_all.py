"""Re-run the executed notebook series end to end, keeping every executed copy.

Runs the notebooks saved in `Experiment/` - the copies whose configuration
produced the reported results, e.g. `06_... (tomasov_full_1002).ipynb` - each in
a fresh Jupyter kernel started in `Experiment/`, and writes the executed copy,
outputs included, to OUT_DIR instead of over the original.  Stage outputs go
wherever DAS_RESULTS_DIR points, so a re-run need never touch `Results/`.

Live view: every line a notebook prints is streamed, as it happens, to the
console and to OUT_DIR/live.log, and the executed copy is saved after each
cell, so it can be opened mid-run.  Follow the log with

    Get-Content OUT_DIR\\live.log -Wait -Tail 40 -Encoding UTF8     # PowerShell
    tail -f OUT_DIR/live.log                                       # Linux / Jetson

Resumable: a notebook whose copy in OUT_DIR finished cleanly is skipped (delete
it to run it again).  Stops at the first failure, because every later stage
reads what the earlier ones wrote.

    $env:DAS_RESULTS_DIR = "..\\Rerun\\Results"          # PowerShell
    python _run_all.py ..\\Rerun\\notebooks             # every notebook, in order
    python _run_all.py ..\\Rerun\\notebooks 03 04       # names starting with 03 / 04
"""
from __future__ import annotations

import asyncio
import csv
import datetime as dt
import os
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, DeadKernelError

HERE = Path(__file__).resolve().parent
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

if sys.platform == "win32":     # zmq needs a selector loop; avoids a warning per kernel
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def keep_awake() -> None:
    """Stop Windows idle-sleeping while this process runs (released on exit).

    A full re-run takes many hours; an idle sleep would stall it mid-notebook.
    A closed lid or a manual sleep still sleeps.
    """
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


class Live:
    """Writes to the console and appends to live.log, flushing every time."""

    def __init__(self, path: Path):
        self.fh = open(path, "a", encoding="utf-8")

    def write(self, text: str) -> None:
        text = ANSI.sub("", text)
        sys.stdout.write(text)
        sys.stdout.flush()
        self.fh.write(text)
        self.fh.flush()


class LiveClient(NotebookClient):
    """A NotebookClient that echoes each output the moment the kernel sends it."""

    live: Live

    def process_message(self, msg, cell, cell_index):
        content, kind = msg["content"], msg["msg_type"]
        if kind == "stream":
            self.live.write(content["text"])
        elif kind in ("display_data", "execute_result"):
            data = content.get("data", {})
            if "image/png" in data:
                self.live.write("[figure]\n")
            elif "text/plain" in data:
                self.live.write(data["text/plain"][:4000] + "\n")
        elif kind == "error":
            self.live.write("\n".join(content.get("traceback", [])) + "\n")
        return super().process_message(msg, cell, cell_index)


def finished(path: Path) -> bool:
    if not path.exists():
        return False
    return nbformat.read(path, as_version=4).metadata.get("rerun", {}).get("status") == "ok"


def run(src: Path, dst: Path, live: Live) -> str:
    nb = nbformat.read(src, as_version=4)
    code = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.strip()]
    t0 = time.time()

    def stamp(status: str) -> None:
        nb.metadata["rerun"] = {
            "status": status,
            "seconds": round(time.time() - t0, 1),
            "finished": dt.datetime.now().isoformat(timespec="seconds"),
            "python": sys.executable,
            "results_dir": os.environ.get("DAS_RESULTS_DIR", "Results/ (default)"),
        }

    def on_cell_start(cell, cell_index):
        if cell_index in code:
            first = next((ln.strip() for ln in cell.source.splitlines() if ln.strip()), "")
            live.write(f"\n--- {src.stem} | cell {code.index(cell_index) + 1}/{len(code)} "
                       f"| {dt.datetime.now():%H:%M:%S} | {first[:60]}\n")

    def on_cell_executed(cell, cell_index, execute_reply):
        stamp("running")                 # partial copy: open it to see outputs so far
        nbformat.write(nb, dst)

    # The "python" in the kernel spec resolves to this interpreter, so the
    # notebooks run in the same environment as this script.
    client = LiveClient(nb, timeout=None, kernel_name="python3",
                        resources={"metadata": {"path": str(HERE)}},
                        on_cell_start=on_cell_start, on_cell_executed=on_cell_executed)
    client.live = live
    status = "ok"
    try:
        client.execute()
    except CellExecutionError:
        status = "failed"
    except DeadKernelError:
        status = "died"                  # e.g. the kernel ran out of memory
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        stamp(status)
        nbformat.write(nb, dst)
    return status


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    out = Path(argv[0]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    live = Live(out / "live.log")
    notebooks = sorted(HERE.glob("[0-9][0-9]_*.ipynb"))
    if argv[1:]:
        notebooks = [p for p in notebooks if any(p.name.startswith(w) for w in argv[1:])]
    live.write(f"\n=== run started {dt.datetime.now():%Y-%m-%d %H:%M} | results -> "
               f"{os.environ.get('DAS_RESULTS_DIR', 'Results/ (default!)')}\n")
    keep_awake()

    log = out / "run_log.csv"
    new_log = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(["notebook", "status", "seconds", "finished"])
        for src in notebooks:
            dst = out / src.name
            if finished(dst):
                live.write(f"skip  {src.name} (already finished)\n")
                continue
            live.write(f"\n=== start {src.name}  {dt.datetime.now():%Y-%m-%d %H:%M}\n")
            status = run(src, dst, live)
            meta = nbformat.read(dst, as_version=4).metadata["rerun"]
            writer.writerow([src.name, status, meta["seconds"], meta["finished"]])
            fh.flush()
            live.write(f"=== {status} {src.name}  {meta['seconds'] / 60:.1f} min\n")
            if status != "ok":
                live.write(f"stopped - see the failing cell in {dst}\n")
                return 1
    live.write("=== all requested notebooks finished\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
