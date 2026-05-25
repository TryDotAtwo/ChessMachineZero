"""Local HTTP dashboard for observing trace-based ChessMachineZero games."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

from chess_machine_zero.chess.board_io import STARTING_FEN
from chess_machine_zero.dashboard.state import CMZDashboardSession, DashboardMoveError
from chess_machine_zero.rng import DEFAULT_SEED


STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/favicon.ico": ("favicon.svg", "image/svg+xml"),
    "/static/favicon.svg": ("favicon.svg", "image/svg+xml"),
    "/static/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
    "/static/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
}


@dataclass(slots=True)
class DashboardApp:
    seed: int = DEFAULT_SEED
    start_fen: str = STARTING_FEN
    max_plies: int = 64
    temperature: float = 0.0
    session: CMZDashboardSession = field(init=False)

    def __post_init__(self) -> None:
        self.session = CMZDashboardSession(
            seed=self.seed,
            start_fen=self.start_fen,
            max_plies=self.max_plies,
            temperature=self.temperature,
        )

    def handle(self, method: str, raw_path: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
        path = urlsplit(raw_path).path
        try:
            if method == "GET" and path == "/api/snapshot":
                return _json_response(200, self.session.snapshot())
            if method == "POST" and path == "/api/reset":
                payload = _decode_json(body)
                return _json_response(200, self._handle_reset(payload))
            if method == "POST" and path == "/api/step":
                payload = _decode_json(body)
                return _json_response(200, self._handle_step(payload))
            if method == "POST" and path == "/api/move":
                payload = _decode_json(body)
                return _json_response(200, self._handle_move(payload))
            if method == "GET" and path in STATIC_ROUTES:
                return _static_response(path)
            return _json_response(404, {"error": {"code": "not_found", "message": f"unknown route: {path}"}})
        except DashboardMoveError as error:
            return _json_response(
                400,
                {
                    "error": {"code": error.code, "message": str(error)},
                    "snapshot": self.session.snapshot(),
                },
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            return _json_response(
                400,
                {
                    "error": {"code": "bad_request", "message": str(error)},
                    "snapshot": self.session.snapshot(),
                },
            )

    def _handle_reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.session.reset(
            start_fen=str(payload.get("fen", STARTING_FEN)),
            seed=_optional_int(payload, "seed"),
            max_plies=_optional_int(payload, "max_plies"),
            temperature=_optional_float(payload, "temperature"),
        )

    def _handle_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        count = int(payload.get("count", 1))
        if not 1 <= count <= 128:
            raise ValueError("count must be within [1, 128]")
        for _ in range(count):
            if self.session.snapshot()["terminal"]["is_terminal"]:
                break
            self.session.step_transformer()
        return self.session.snapshot()

    def _handle_move(self, payload: dict[str, Any]) -> dict[str, Any]:
        move = payload.get("move")
        if not isinstance(move, str):
            raise ValueError("move must be a UCI string")
        auto_reply = bool(payload.get("auto_reply", True))
        self.session.play_human_move(move, auto_reply=auto_reply)
        return self.session.snapshot()


class DashboardHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "ChessMachineZeroDashboard/0.1"

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length) if length else b""
        status, headers, payload = self.server.app.handle(self.command, self.path, body)  # type: ignore[attr-defined]
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def build_server(host: str, port: int, app: DashboardApp) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), DashboardHTTPRequestHandler)
    server.app = app  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local ChessMachineZero dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-plies", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)
    app = DashboardApp(seed=args.seed, max_plies=args.max_plies, temperature=args.temperature)
    server = build_server(args.host, args.port, app)
    print(f"ChessMachineZero dashboard listening on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


def _decode_json(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def _json_response(status: int, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return status, {"content-type": "application/json; charset=utf-8"}, body


def _static_response(path: str) -> tuple[int, dict[str, str], bytes]:
    filename, content_type = STATIC_ROUTES[path]
    static_root = resources.files("chess_machine_zero.dashboard").joinpath("static")
    payload = static_root.joinpath(filename).read_bytes()
    return 200, {"content-type": content_type}, payload


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    if key not in payload or payload[key] is None:
        return None
    return int(payload[key])


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    if key not in payload or payload[key] is None:
        return None
    return float(payload[key])


if __name__ == "__main__":
    raise SystemExit(main())
