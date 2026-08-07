"""Minimal claim-path score HTTP API (stdlib) — Downstream should terminate TLS."""

from __future__ import annotations

import hmac
import json
import logging
import os
import ssl
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from refund_abuse_risk.config import load_operating_point
from refund_abuse_risk.pipeline.score import OrderRiskCache, claim_path_read

_LOG = logging.getLogger("refund_abuse_risk.serve")


@dataclass
class ScoreApiConfig:
    token: str
    cache: OrderRiskCache
    model_version: str = "0.0.0"
    policy_version: str = "0.0.0"
    host: str = "127.0.0.1"
    port: int = 8080
    include_evidence_default: bool = False
    tls_certfile: str | None = None
    tls_keyfile: str | None = None
    require_tls: bool = False
    audit_log: bool = True
    # Simple per-process rate limit (requests per token window).
    rate_limit_per_minute: int = 120
    _hits: dict[str, list[float]] = field(default_factory=dict)


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


def _rate_limited(cfg: ScoreApiConfig, key: str) -> bool:
    import time

    if cfg.rate_limit_per_minute <= 0:
        return False
    now = time.time()
    window = cfg._hits.setdefault(key, [])
    cfg._hits[key] = [t for t in window if now - t < 60.0]
    if len(cfg._hits[key]) >= int(cfg.rate_limit_per_minute):
        return True
    cfg._hits[key].append(now)
    return False


def make_handler_class(cfg: ScoreApiConfig) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            if cfg.audit_log:
                _LOG.info("%s - %s", self.address_string(), fmt % args)

        def _authorized(self) -> bool:
            expected = cfg.token.encode("utf-8")
            auth = self.headers.get("Authorization") or ""
            if auth.startswith("Bearer "):
                got = auth[len("Bearer ") :].encode("utf-8")
                if hmac.compare_digest(got, expected):
                    return True
            api = (self.headers.get("X-Api-Token") or "").encode("utf-8")
            return bool(api) and hmac.compare_digest(api, expected)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)
            if path == "/health":
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "service": "refund-abuse-risk",
                        "tls": bool(cfg.tls_certfile and cfg.tls_keyfile),
                    },
                    model_version=cfg.model_version,
                    policy_version=cfg.policy_version,
                )
                return
            if not self._authorized():
                if cfg.audit_log:
                    _LOG.warning("unauthorized %s %s", self.command, path)
                _unauthorized(self)
                return
            client = self.client_address[0] if self.client_address else "unknown"
            if _rate_limited(cfg, client):
                _json_response(
                    self,
                    429,
                    {"error": "rate_limited"},
                    model_version=cfg.model_version,
                    policy_version=cfg.policy_version,
                )
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
                    if cfg.audit_log:
                        _LOG.info("miss order_id=%s", order_id)
                    _json_response(
                        self,
                        404,
                        {"error": "not_found", "order_id": order_id},
                        model_version=cfg.model_version,
                        policy_version=cfg.policy_version,
                    )
                    return
                payload = snap.model_dump(mode="json")
                include_ev = cfg.include_evidence_default or (
                    qs.get("include_evidence", ["0"])[0] in {"1", "true", "yes"}
                )
                if not include_ev:
                    payload.pop("evidence_pack", None)
                if cfg.audit_log:
                    _LOG.info(
                        "risk order_id=%s tier=%s evidence=%s",
                        order_id,
                        payload.get("suggested_tier"),
                        include_ev,
                    )
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
    if cfg.require_tls and not (cfg.tls_certfile and cfg.tls_keyfile):
        raise SystemExit(
            "SCORE_API_REQUIRE_TLS=1 but SCORE_API_TLS_CERTFILE / "
            "SCORE_API_TLS_KEYFILE not set"
        )
    if cfg.host in {"0.0.0.0", "::"} and not (cfg.tls_certfile and cfg.tls_keyfile):
        _LOG.warning(
            "binding %s without TLS — terminate TLS at a reverse proxy or set cert/key",
            cfg.host,
        )
    handler = make_handler_class(cfg)
    httpd = ThreadingHTTPServer((cfg.host, int(cfg.port)), handler)
    if cfg.tls_certfile and cfg.tls_keyfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=cfg.tls_certfile, keyfile=cfg.tls_keyfile)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
    else:
        scheme = "http"
        if cfg.host != "127.0.0.1":
            _LOG.warning(
                "serving plain HTTP on %s — use TLS cert/key or localhost-only bind",
                cfg.host,
            )
    print(
        json.dumps(
            {
                "listening": f"{scheme}://{cfg.host}:{cfg.port}",
                "health": "/health",
                "risk": "/v1/orders/{order_id}/risk",
                "tls": scheme == "https",
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return ScoreApiConfig(
        token=token,
        cache=cache,
        model_version=str(op.get("model_version", "0.0.0")),
        policy_version=str(op.get("policy_version", "0.0.0")),
        host=os.environ.get("SCORE_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("SCORE_API_PORT", "8080")),
        include_evidence_default=os.environ.get("SCORE_API_INCLUDE_EVIDENCE", "").lower()
        in {"1", "true", "yes"},
        tls_certfile=os.environ.get("SCORE_API_TLS_CERTFILE") or None,
        tls_keyfile=os.environ.get("SCORE_API_TLS_KEYFILE") or None,
        require_tls=os.environ.get("SCORE_API_REQUIRE_TLS", "").lower()
        in {"1", "true", "yes"},
        audit_log=os.environ.get("SCORE_API_AUDIT_LOG", "1").lower()
        not in {"0", "false", "no"},
        rate_limit_per_minute=int(os.environ.get("SCORE_API_RATE_LIMIT_PER_MINUTE", "120")),
    )
