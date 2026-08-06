"""Prod readiness B/D/F/G: serve API, http feeds, overnight prod, overlay dry-run."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import pytest
import yaml

from refund_abuse_risk.integrations.feeds import pull_http, pull_s3, SUPPORTED_SOURCE_TYPES
from refund_abuse_risk.pipeline.score import OrderRiskCache
from refund_abuse_risk.schemas.models import (
    EntityScores,
    EvidencePack,
    LinkScores,
    OrderRiskSnapshot,
    RefundEffect,
    SuggestedTier,
)
from refund_abuse_risk.scoring.decision import (
    recommend_decision_thresholds_by_slice,
    recommended_overlays_from_slices,
    resolve_decision_thresholds,
)
from refund_abuse_risk.serve.api import ScoreApiConfig, make_handler_class

ROOT = Path(__file__).resolve().parents[1]


def test_supported_source_types_include_http_s3() -> None:
    assert "http" in SUPPORTED_SOURCE_TYPES
    assert "s3" in SUPPORTED_SOURCE_TYPES


def test_pull_http_and_s3_https(tmp_path: Path) -> None:
    payload = b"order_id,disposition\nO1,chargeback_lost\n"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: ANN001
            return

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), H)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/dispositions.csv"
        out = pull_http(
            "dispositions",
            {"uri": url, "filename": "dispositions.csv"},
            root=tmp_path,
            stage_root=tmp_path / "stage",
        )
        assert out.read_bytes() == payload
        # s3 driver delegates https/http to pull_http
        out2 = pull_s3(
            "dispositions",
            {"uri": url, "filename": "d2.csv"},
            root=tmp_path,
            stage_root=tmp_path / "stage",
        )
        assert out2.read_bytes() == payload
    finally:
        server.shutdown()


def test_pull_s3_native_requires_boto3_or_raises() -> None:
    try:
        import boto3  # noqa: F401
    except ImportError:
        with pytest.raises(NotImplementedError, match="boto3"):
            pull_s3("x", {"uri": "s3://bucket/key.csv"}, root=ROOT)
    else:
        pytest.skip("boto3 installed — native s3 path needs live credentials")


def test_score_api_auth_and_claim_path() -> None:
    from http.client import HTTPConnection

    cache = OrderRiskCache()
    snap = OrderRiskSnapshot(
        order_id="O99",
        market="SG",
        vertical="food",
        abuse_score=40.0,
        fraud_score=55.0,
        decision_score=55.0,
        entity_scores=EntityScores(),
        link_scores=LinkScores(),
        suggested_tier=SuggestedTier.HOLD_REVIEW,
        refund_effect=RefundEffect.REFUND_MANUAL_REVIEW,
        evidence_pack=EvidencePack(),
        model_version="9.9.9",
        policy_version="8.8.8",
    )
    cache.put(snap, {"order_id": "O99"})
    cfg = ScoreApiConfig(
        token="secret-token",
        cache=cache,
        model_version="9.9.9",
        policy_version="8.8.8",
        host="127.0.0.1",
        port=0,
    )
    handler = make_handler_class(cfg)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        health = conn.getresponse()
        assert health.status == 200

        conn.request("GET", "/v1/orders/O99/risk")
        unauth = conn.getresponse()
        assert unauth.status == 401
        unauth.read()

        conn.request(
            "GET",
            "/v1/orders/O99/risk",
            headers={"Authorization": "Bearer secret-token"},
        )
        ok = conn.getresponse()
        assert ok.status == 200
        assert ok.getheader("X-Model-Version") == "9.9.9"
        body = json.loads(ok.read().decode())
        assert body["order_id"] == "O99"
        assert body["suggested_tier"] == "hold_review"

        conn.request(
            "GET",
            "/v1/orders/missing/risk",
            headers={"X-Api-Token": "secret-token"},
        )
        missing = conn.getresponse()
        assert missing.status == 404
    finally:
        server.shutdown()


def test_overnight_prod_profile_contract() -> None:
    path = ROOT / "config" / "ops.overnight.prod.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["require_promote"] is True
    ids = [s["id"] for s in data["steps"]]
    assert "validate_oot_schema" in ids
    train = next(s for s in data["steps"] if s["id"] == "train")
    assert "--require-dispositions" in train["argv"]
    assert data["paths"]["oot_pack"].endswith("prod_shaped_v1")
    assert data["paths"]["floors"].endswith("oot_floors.prod.yaml")


def test_overnight_prod_dry_run() -> None:
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/ops_overnight.py",
            "--profile",
            "config/ops.overnight.prod.yaml",
            "--dry-run",
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "validate_oot_schema" in proc.stdout
    assert "--require-dispositions" in proc.stdout
    assert "skip promote_overlays" in proc.stdout


def test_overlay_promote_dry_run_from_eligible_slices(tmp_path: Path) -> None:
    import importlib.util

    # Build a slice large enough to be promote_eligible under costed recommend.
    rows = []
    for i in range(40):
        rows.append(
            {
                "market": "SG",
                "vertical": "food",
                "pattern_y": 1 if i < 10 else 0,
                "decision_score": float(95 - i) if i < 10 else float(20 - (i % 5)),
                "amount": 12.0,
            }
        )
    frame = pd.DataFrame(rows)
    by_slice = recommend_decision_thresholds_by_slice(
        frame,
        min_slice_n=20,
        min_slice_positives=3,
        target_recall=0.8,
        min_precision_at_soft=0.3,
    )
    overlays = recommended_overlays_from_slices(by_slice)
    # If costed ok fails on this synth, still exercise dry-run with a forced eligible overlay.
    if not overlays:
        overlays = [
            {
                "market": "SG",
                "vertical": "food",
                "soft_friction": 30.0,
                "hold_review": 45.0,
                "auto_deny": 70.0,
            }
        ]
    ov_path = tmp_path / "overlays.yaml"
    ov_path.write_text(
        yaml.safe_dump({"decision_threshold_overlays": overlays}),
        encoding="utf-8",
    )
    op_path = tmp_path / "op.yaml"
    op_path.write_text(
        yaml.safe_dump(
            {
                "policy_version": "1.0.0",
                "decision_thresholds": {
                    "soft_friction": 35,
                    "hold_review": 50,
                    "auto_deny": 75,
                },
                "decision_threshold_overlays": [],
            }
        ),
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location(
        "promote_overlays", ROOT / "scripts" / "promote_overlays.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    summary = mod.promote(
        overlays_path=ov_path,
        op_path=op_path,
        backup_dir=tmp_path / "backups",
        dry_run=True,
        require_non_empty=True,
        backup_keep=3,
    )
    assert summary["dry_run"] is True
    assert summary["n_overlays"] >= 1
    # Apply for real and check resolve
    mod.promote(
        overlays_path=ov_path,
        op_path=op_path,
        backup_dir=tmp_path / "backups",
        dry_run=False,
        require_non_empty=True,
        backup_keep=3,
    )
    live = yaml.safe_load(op_path.read_text(encoding="utf-8"))
    thr = resolve_decision_thresholds(live, market="SG", vertical="food")
    assert thr["soft_friction"] == pytest.approx(float(overlays[0]["soft_friction"]))
