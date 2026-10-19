"""Keep the library run (bench.library) going unattended, e.g. overnight.

Runs here, next to .local, and every CHECK seconds looks after three things:

  the tunnel   ssh -L to the GPU machine, which also runs the bench.gpu worker
               there (stopping any old one first); started again whenever it ends
  the worker   restarted when it uses more than MAX_WORKER_GB
  the run      bench.library, started when it isn't running and hasn't finished
               (it skips finished volumes); stopped while the disk has less than
               MIN_FREE_GB free

and when no volume has finished for STALL seconds, it restarts the worker,
then (still stuck after another STALL) the run. When the run has finished,
it stops the tunnel and exits. What it does goes to stdout:

    nohup setsid .venv/bin/python -m bench.watchdog >> .local/watchdog.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from bench.catalog import LOCAL, ROOT
from bench.library import AVIF_ROOT, VECTORS

CHECK = 60                # seconds between checks
STALL = 20 * 60           # seconds without a finished volume that count as stuck
MAX_WORKER_GB = 8.0       # the worker's committed memory (~3.8 GB normally)
MIN_FREE_GB = 10.0        # free disk under AVIF_ROOT
RUN_LOG, WORKER_LOG = LOCAL / "library.log", LOCAL / "library-worker.log"
LOCAL_PORT, REMOTE_PORT = 8813, 8800
WORKERS = "Get-CimInstance Win32_Process -Filter \\\"name='python.exe'\\\" | Where-Object CommandLine -like '*bench.gpu*'"


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


class Watchdog:
    def __init__(self, ssh_host: str, model: str):
        self.ssh_host, self.model = ssh_host, model
        self.gpu_url = f"http://127.0.0.1:{LOCAL_PORT}"
        self.tunnel: subprocess.Popen | None = None
        self.acted = time.time()  # last restart: stalls count from here or the last finished volume
        self.worker_restarted_for_stall = False

    # ---------------------------------------------------------------- the GPU machine

    def powershell(self, command: str, timeout: int = 60) -> str | None:
        """Output of a PowerShell command on the GPU machine, or None if it can't be reached."""
        try:
            done = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", self.ssh_host,
                                   f'powershell -NoProfile -c "{command}"'], capture_output=True, text=True, check=False,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    def start_tunnel(self) -> None:
        log("starting the tunnel and worker")
        remote = (f'powershell -NoProfile -c "{WORKERS} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"'
                  f" & cd eeffoc && .venv\\Scripts\\python -u -m bench.gpu --host 127.0.0.1 --port {REMOTE_PORT}")
        with WORKER_LOG.open("a") as out:
            self.tunnel = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
                 "-o", "ServerAliveCountMax=4", "-o", "ExitOnForwardFailure=yes",
                 "-L", f"{LOCAL_PORT}:127.0.0.1:{REMOTE_PORT}", self.ssh_host, remote],
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        self.acted = time.time()

    def restart_worker(self, why: str) -> None:
        log(f"restarting the worker: {why}")
        self.powershell(f"{WORKERS} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}")
        self.acted = time.time()  # the tunnel ends with it, and the next check starts both again

    def worker_gb(self) -> float | None:
        """The worker's committed memory, or None if it isn't running or can't be reached."""
        out = self.powershell(f"{WORKERS} | ForEach-Object {{ (Get-Process -Id $_.ProcessId).PagedMemorySize64 }}")
        sizes = [int(n) for n in (out or "").split() if n.isdigit()]
        return max(sizes) / 2**30 if sizes else None

    def worker_answers(self) -> bool:
        try:
            urlopen(f"{self.gpu_url}/model/none", timeout=10)
        except HTTPError as error:
            return error.code == 404  # it's there: it just doesn't have a model called "none"
        except OSError:
            return False
        return True

    # ---------------------------------------------------------------- the run

    def run_pids(self) -> list[int]:
        """bench.library's main process (not its encoders), started by us or anyone else."""
        pids = []
        for proc in Path("/proc").iterdir():
            try:
                args = (proc / "cmdline").read_bytes().split(b"\0")
            except (OSError, ValueError):
                continue
            if proc.name.isdigit() and b"-m" in args and b"bench.library" in args:
                pids.append(int(proc.name))
        return pids

    def start_run(self) -> None:
        log("starting bench.library")
        with RUN_LOG.open("a") as out:
            subprocess.Popen([sys.executable, "-m", "bench.library", "--model", self.model], cwd=ROOT,
                             env={**os.environ, "EEFFOC_GPU": self.gpu_url}, stdin=subprocess.DEVNULL,
                             stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        self.acted = time.time()

    def stop_run(self, why: str) -> None:
        log(f"stopping bench.library: {why}")
        for pid in self.run_pids():
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)  # its encoders too: they share its process group
            except ProcessLookupError:
                pass
        self.acted = time.time()

    def finished(self) -> bool:
        lines = RUN_LOG.read_text().splitlines() if RUN_LOG.exists() else []
        return bool(lines) and lines[-1].startswith("done:") and not self.run_pids()

    def last_volume(self) -> float:
        """When the newest volume's vectors were written."""
        records = (VECTORS / f"{self.model}-fine").rglob("*.json")
        return max((p.stat().st_mtime for p in records), default=0.0)

    # ---------------------------------------------------------------- one check

    def check(self) -> bool:
        """Fix what needs fixing; False once the run has finished."""
        if self.finished():
            log("the run has finished; stopping the tunnel")
            if self.tunnel and self.tunnel.poll() is None:
                self.tunnel.terminate()
            return False
        if self.tunnel is None or self.tunnel.poll() is not None:
            self.start_tunnel()
        free_gb = shutil.disk_usage(AVIF_ROOT if AVIF_ROOT.exists() else Path.home()).free / 2**30
        running = bool(self.run_pids())
        if free_gb < MIN_FREE_GB:
            if running:
                self.stop_run(f"only {free_gb:.0f} GB free")
            return True
        if not running:
            self.start_run()
            return True
        memory = self.worker_gb()
        if memory is not None and memory > MAX_WORKER_GB:
            self.restart_worker(f"using {memory:.1f} GB")
            return True
        stuck = time.time() - max(self.last_volume(), self.acted)
        if stuck > STALL:
            if not self.worker_restarted_for_stall:
                self.restart_worker(f"no volume finished in {stuck / 60:.0f} min"
                                    + ("" if self.worker_answers() else ", and it doesn't answer"))
                self.worker_restarted_for_stall = True
            else:
                self.stop_run(f"still no volume finished in {stuck / 60:.0f} min")
                self.worker_restarted_for_stall = False
        elif time.time() - self.last_volume() < STALL:
            self.worker_restarted_for_stall = False
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", default="koush@100.97.226.123", help="the GPU machine (Windows, with the repo in ~/eeffoc)")
    parser.add_argument("--model", default="manga-ft1")
    args = parser.parse_args()
    watchdog = Watchdog(args.ssh, args.model)
    log(f"watching bench.library, the worker on {args.ssh} and the tunnel (every {CHECK}s)")
    while True:
        try:
            if not watchdog.check():
                return
        except Exception as error:  # noqa: BLE001 - a check that fails mustn't end the watching
            log(f"check failed: {error!r}")
        time.sleep(CHECK)


if __name__ == "__main__":
    main()
