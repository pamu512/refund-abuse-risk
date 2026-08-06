"""Minimal claim-path score HTTP API (stdlib) — Downstream can wrap/replace."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from refund_abuse_risk.config import load_operating_point
from refund_abuse_risk.pipeline.score import OrderRiskCache, claim_path_read


@dataclass
class ScoreApiConfig:
    token: str
    cache: OrderRiskCache
    model_version: str = "0.0.0"
    policy_version: str = "0.0.0"
    host: str = "127.0.0.1"
    port: int = 8080


def _unauthorized(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(401)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(b'{"error":"unauthorized"}')


def _json_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    payload: dict[str, Any],
    *,
    model_version: str,
    policy_version: str,
) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("X-Model-Version", model_version)
    handler.send_header("X-Policy-Version", policy_version)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler_class(cfg: ScoreApiConfig) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quieter tests
            return

        def _authorized(self) -> bool:
            auth = self.headers.get("Authorization") or ""
            if auth == f"Bearer {cfg.token}":
                return True
            # Also accept X-Api-Token for simple gateways.
            return (self.headers.get("X-Api-Token") or "") == cfg.token

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/health":
                _json_response(
                    self,
                    200,
                    {"ok": True, "service": "refund-abuse-risk"},
                    model_version=cfg.model_version,
                    policy_version=cfg.policy_version,
                )
                return
            if not self._authorized():
                _unauthorized(self)
                return
            prefix = "/v1/orders/"
            suffix = "/risk"
            if path.startswith(prefix) and path.endswith(suffix):
                order_id = path[len(prefix) : -len(suffix)]
                if not order_id or "/" in order_id:
                    _json_response(
                        self,
                        400,
                        {"error": "invalid_order_id"},
                        model_version=cfg.model_version,
                        policy_version=cfg.policy_version,
                    )
                    return
                try:
                    snap = claim_path_read(order_id, cfg.cache)
                except KeyError:
                    _json_response(
                        self,
                        404,
                        {"error": "not_found", "order_id": order_id},
                        model_version=cfg.model_version,
                        policy_version=cfg.policy_version,
                    )
                    return
                payload = snap.model_dump(mode="json")
                _json_response(
                    self,
                    200,
                    payload,
                    model_version=str(snap.model_version or cfg.model_version),
                    policy_version=str(snap.policy_version or cfg.policy_version),
                )
                return
            _json_response(
                self,
                404,
                {"error": "not_found"},
                model_version=cfg.model_version,
                policy_version=cfg.policy_version,
            )

    return Handler


def serve_forever(cfg: ScoreApiConfig) -> None:
    handler = make_handler_class(cfg)
    httpd = ThreadingHTTPServer((cfg.host, int(cfg.port)), handler)
    print(
        json.dumps(
            {
                "listening": f"http://{cfg.host}:{cfg.port}",
                "health": "/health",
                "risk": "/v1/orders/{order_id}/risk",
                "model_version": cfg.model_version,
                "policy_version": cfg.policy_version,
            }
        ),
        flush=True,
    )
    httpd.serve_forever()


def config_from_env(cache: OrderRiskCache) -> ScoreApiConfig:
    op = load_operating_point()
    token = os.environ.get("SCORE_API_TOKEN") or ""
    if not token:
        raise SystemExit("SCORE_API_TOKEN is required")
    return ScoreApiConfig(
        token=token,
        cache=cache,
        model_version=str(op.get("model_version", "0.0.0")),
        policy_version=str(op.get("policy_version", "0.0.0")),
        host=os.environ.get("SCORE_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("SCORE_API_PORT", "8080")),
    )
