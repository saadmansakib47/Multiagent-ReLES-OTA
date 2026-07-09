"""
Local web control interface for MA-ReLES-OTA.

Run:
    python web_ui.py
"""

from __future__ import annotations

import csv
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "webui"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
LEADERBOARD = RESULTS_DIR / "leaderboard.csv"
DEVICE_CACHE = {"checked_at": 0.0, "data": None}


STATE_LOCK = threading.Lock()
STATE = {
    "running": False,
    "started_at": None,
    "ended_at": None,
    "returncode": None,
    "command": [],
    "logs": [],
    "metrics": {},
    "progress": 0,
    "current_seed": "",
    "error": "",
}
PROCESS = {"handle": None}


def find_project_python() -> str:
    if os.environ.get("RELES_PYTHON"):
        return os.environ["RELES_PYTHON"]

    candidates = [Path(sys.executable)]
    if os.name == "nt":
        candidates.extend([
            ROOT / ".venv" / "Scripts" / "python.exe",
            ROOT / "venv" / "Scripts" / "python.exe",
        ])
    else:
        candidates.extend([
            ROOT / ".venv" / "bin" / "python",
            ROOT / "venv" / "bin" / "python",
        ])

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            result = subprocess.run(
                [str(candidate), "-c", "import pandas, torch, stable_baselines3"],
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
            )
            if result.returncode == 0:
                return str(candidate)
        except Exception:
            continue
    return sys.executable


RUNNER_PYTHON = find_project_python()


def append_log(line: str) -> None:
    clean = line.rstrip("\r\n")
    with STATE_LOCK:
        STATE["logs"].append(clean)
        STATE["logs"] = STATE["logs"][-600:]
        parse_training_line(clean)


def parse_training_line(line: str) -> None:
    metrics = STATE["metrics"]

    seed_match = re.search(r"\[Seed\s+(\d+)/(\d+)\]", line)
    if seed_match:
        STATE["current_seed"] = f"{seed_match.group(1)} / {seed_match.group(2)}"

    if "|" in line:
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            key = parts[0].replace("/", " / ")
            metrics[key] = parts[1]

    total = metrics.get("time / total_timesteps") or metrics.get("total_timesteps")
    if total:
        try:
            STATE["progress"] = min(100, max(STATE["progress"], int(float(str(total).replace(",", "")))))
        except ValueError:
            pass


def get_device_status() -> dict:
    if DEVICE_CACHE["data"] and time.time() - DEVICE_CACHE["checked_at"] < 5:
        return DEVICE_CACHE["data"]

    status = {
        "current": "cpu",
        "cuda_available": False,
        "gpu_name": "",
        "torch_version": "",
        "python": RUNNER_PYTHON,
    }
    try:
        code = (
            "import json, torch; "
            "print(json.dumps({"
            "'torch_version': str(torch.__version__), "
            "'cuda_available': bool(torch.cuda.is_available()), "
            "'current': 'cuda' if torch.cuda.is_available() else 'cpu', "
            "'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''"
            "}))"
        )
        result = subprocess.run(
            [RUNNER_PYTHON, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            status.update(json.loads(result.stdout.strip()))
        else:
            status["error"] = result.stderr.strip()
    except Exception as exc:
        status["error"] = str(exc)
    DEVICE_CACHE["checked_at"] = time.time()
    DEVICE_CACHE["data"] = status
    return status


def read_leaderboard() -> list[dict]:
    if not LEADERBOARD.exists():
        return []
    with open(LEADERBOARD, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def list_charts() -> list[dict]:
    if not CHARTS_DIR.exists():
        return []
    charts = []
    for path in sorted(CHARTS_DIR.glob("*.png"), key=lambda item: item.stat().st_mtime, reverse=True):
        charts.append({
            "name": path.name,
            "url": f"/chart?name={path.name}",
            "modified": path.stat().st_mtime,
        })
    return charts[:8]


def public_state() -> dict:
    with STATE_LOCK:
        state = dict(STATE)
        state["logs"] = list(STATE["logs"])
        state["metrics"] = dict(STATE["metrics"])
    state["leaderboard"] = read_leaderboard()
    state["charts"] = list_charts()
    return state


def bool_arg(value) -> str:
    return "True" if bool(value) else "False"


def build_command(payload: dict) -> list[str]:
    algorithm = payload.get("algorithm", "fp3o")
    compare = payload.get("compare_algorithm", "")
    mode = payload.get("mode", "train")
    command = [
        RUNNER_PYTHON,
        "-u",
        str(ROOT / "main.py"),
        "--mode",
        mode,
        "--algorithm",
        algorithm,
        "--safety",
        bool_arg(payload.get("safety", True)),
        "--n_agents",
        str(int(payload.get("n_agents", 4))),
        "--n_blocks",
        str(int(payload.get("n_blocks", 16))),
        "--timesteps",
        str(int(payload.get("timesteps", 500000))),
        "--seeds",
        str(int(payload.get("seeds", 3))),
        "--n_envs",
        str(int(payload.get("n_envs", 4))),
        "--n_steps",
        str(int(payload.get("n_steps", 256))),
        "--batch_size",
        str(int(payload.get("batch_size", 128))),
        "--n_epochs",
        str(int(payload.get("n_epochs", 10))),
        "--ent_coef",
        str(float(payload.get("ent_coef", 0.01))),
        "--device",
        payload.get("device", "auto"),
        "--bd_mode",
        bool_arg(payload.get("bd_mode", True)),
        "--death_masking",
        bool_arg(payload.get("death_masking", True)),
    ]
    if compare and compare != algorithm:
        command.extend(["--compare_algorithm", compare])
    return command


def run_training(payload: dict) -> tuple[bool, str]:
    with STATE_LOCK:
        if STATE["running"]:
            return False, "A training run is already active."
        STATE.update({
            "running": True,
            "started_at": time.time(),
            "ended_at": None,
            "returncode": None,
            "logs": [],
            "metrics": {},
            "progress": 0,
            "current_seed": "",
            "error": "",
        })
        command = build_command(payload)
        STATE["command"] = command

    def worker() -> None:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            PROCESS["handle"] = process
            assert process.stdout is not None
            for line in process.stdout:
                append_log(line)
            code = process.wait()
            with STATE_LOCK:
                STATE["running"] = False
                STATE["ended_at"] = time.time()
                STATE["returncode"] = code
                if code == 0:
                    STATE["progress"] = 100
                else:
                    STATE["error"] = f"Training exited with code {code}."
        except Exception as exc:
            with STATE_LOCK:
                STATE["running"] = False
                STATE["ended_at"] = time.time()
                STATE["returncode"] = -1
                STATE["error"] = str(exc)
        finally:
            PROCESS["handle"] = None

    threading.Thread(target=worker, daemon=True).start()
    return True, "Training started."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_file(WEB_ROOT / "index.html")
        elif parsed.path.startswith("/static/"):
            self.serve_file(WEB_ROOT / parsed.path.removeprefix("/static/"))
        elif parsed.path == "/api/status":
            self.send_json({"device": get_device_status(), **public_state()})
        elif parsed.path == "/api/results":
            self.send_json({"leaderboard": read_leaderboard(), "charts": list_charts()})
        elif parsed.path == "/chart":
            name = parse_qs(parsed.query).get("name", [""])[0]
            safe_name = Path(unquote(name)).name
            self.serve_file(CHARTS_DIR / safe_name)
        elif parsed.path == "/events":
            self.serve_events()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = {}
        if length:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

        if self.path == "/api/start":
            ok, message = run_training(payload)
            self.send_json({"ok": ok, "message": message}, HTTPStatus.OK if ok else HTTPStatus.CONFLICT)
        elif self.path == "/api/reset":
            with STATE_LOCK:
                if STATE["running"]:
                    self.send_json({"ok": False, "message": "Reset is disabled while training is running."}, HTTPStatus.CONFLICT)
                    return
                STATE.update({
                    "ended_at": None,
                    "returncode": None,
                    "command": [],
                    "logs": [],
                    "metrics": {},
                    "progress": 0,
                    "current_seed": "",
                    "error": "",
                })
            self.send_json({"ok": True, "message": "Interface cleared."})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def serve_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_events(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                data = json.dumps({"device": get_device_status(), **public_state()})
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(1)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            return


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        exc_type, exc, _ = sys.exc_info()
        winerror = getattr(exc, "winerror", None)
        if exc_type in {BrokenPipeError, ConnectionAbortedError, ConnectionResetError} or winerror == 10053:
            return
        super().handle_error(request, client_address)


def main():
    host = "127.0.0.1"
    port = int(os.environ.get("RELES_UI_PORT", "8000"))
    server = QuietThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"MA-ReLES-OTA web interface running at {url}")
    print("Press Ctrl+C to stop the server.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    server.serve_forever()


if __name__ == "__main__":
    main()
