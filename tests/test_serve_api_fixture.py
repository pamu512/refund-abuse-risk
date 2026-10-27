"""#3.2: serve_api must score an in-repo fixture, not replay --cache-jsonl."""

from __future__ import annotations

import json
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path
from threading import Thread

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "oot_packs" / "prod_shaped_v1"
PLANTED_DECISION = 12.34
PLANTED_ORDER = "PS-0000"

_PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "CUTOVER.md",
    ROOT / "docs" / "MANUAL.md",
    ROOT / "docs" / "OPS_RUNBOOK.md",
    ROOT / "scripts" / "serve_api.py",
    ROOT / "src" / "refund_abuse_risk" / "serve" / "api.py",
)


def _planted_snapshot_line(order_id: str = PLANTED_ORDER) -> str:
    return json.dumps(
        {
            "order_id": order_id,
            "market": "XX",
            "vertical": "food",
            "abuse_score": 1.0,
            "fraud_score": 1.0,
            "decision_score": PLANTED_DECISION,
            "entity_scores": {},
            "link_scores": {},
            "suggested_tier": "auto_approve",
            "refund_effect": "refund_auto_grant",
            "reason_codes": ["PLANTED_CACHE"],
            "hard_gated": False,
            "model_version": "planted",
            "policy_version": "planted",
        }
    )


@pytest.fixture(scope="module")
def scored_fixture_cache():
    planted = ROOT / "data" / "score_cache.jsonl"
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text(_planted_snapshot_line() + "\n", encoding="utf-8")
    try:
        from scripts.serve_api import score_fixture_dir

        return score_fixture_dir(FIXTURE)
    finally:
        planted.unlink(missing_ok=True)


def test_score_fixture_dir_runs_real_stack_and_ignores_cache_jsonl(
    scored_fixture_cache,
) -> None:
    cache = scored_fixture_cache
    orders = pd.read_csv(FIXTURE / "orders.csv")
    snaps = cache.all_snapshots()
    assert len(snaps) == len(orders)
    snap = cache.get(PLANTED_ORDER)
    assert snap is not None
    assert snap.decision_score != PLANTED_DECISION
    assert snap.abuse_score != 1.0
    assert snap.fraud_score != 1.0
    assert snap.market == "SG"
    assert snap.vertical == "food"
    assert "PLANTED_CACHE" not in snap.reason_codes
    assert 0.0 <= float(snap.decision_score) <= 100.0
    assert 0.0 <= float(snap.abuse_score) <= 100.0
    assert 0.0 <= float(snap.fraud_score) <= 100.0
    assert snap.model_version != "planted"
    assert snap.suggested_tier.value
    assert isinstance(snap.reason_codes, list)


def test_score_fixture_dir_requires_core_csvs(tmp_path: Path) -> None:
    from scripts.serve_api import score_fixture_dir

    with pytest.raises(FileNotFoundError, match="orders.csv"):
        score_fixture_dir(tmp_path)


def test_serve_cli_defaults_to_committed_fixture_not_cache_jsonl() -> None:
    from scripts import serve_api

    parser = serve_api.build_parser()
    help_text = parser.format_help()
    assert "--data-dir" in help_text
    assert "--cache-jsonl" not in help_text
    args = parser.parse_args([])
    assert Path(args.data_dir).resolve() == FIXTURE.resolve()
    assert not hasattr(args, "cache_jsonl")
    doc = f"{serve_api.__doc__ or ''}\n{help_text}"
    lowered = doc.lower()
    assert "score http api" not in lowered
    assert "live score http" not in lowered
    assert "not a live scorer" in lowered or "not live scoring" in lowered
    assert "fixture" in lowered or "precompute" in lowered


def test_public_docs_do_not_market_cache_replay_as_live_score() -> None:
    for path in _PUBLIC_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "score HTTP API" not in text, path
        assert "--cache-jsonl" not in text, path
        if path.name == "README.md":
            assert "minimal `serve_api.py`" not in text
            assert "minimal serve_api" not in text.lower()
        if "serve_api.py" in text:
            lowered = text.lower()
            assert "fixture" in lowered or "precompute" in lowered, path


def test_serve_http_returns_fixture_precompute(scored_fixture_cache) -> None:
    from refund_abuse_risk.serve.api import ScoreApiConfig, make_handler_class

    cache = scored_fixture_cache
    expected = cache.get(PLANTED_ORDER)
    assert expected is not None
    cfg = ScoreApiConfig(
        token="fixture-token",
        cache=cache,
        model_version=expected.model_version,
        policy_version=expected.policy_version,
        host="127.0.0.1",
        port=0,
    )
    server = HTTPServer(("127.0.0.1", 0), make_handler_class(cfg))
    port = server.server_address[1]
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            "GET",
            f"/v1/orders/{PLANTED_ORDER}/risk",
            headers={"Authorization": "Bearer fixture-token"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        body = json.loads(resp.read().decode())
        assert body["order_id"] == PLANTED_ORDER
        assert body["decision_score"] == pytest.approx(expected.decision_score)
        assert body["decision_score"] != PLANTED_DECISION
        assert body["market"] == "SG"
        assert "PLANTED_CACHE" not in body.get("reason_codes", [])
    finally:
        server.shutdown()
