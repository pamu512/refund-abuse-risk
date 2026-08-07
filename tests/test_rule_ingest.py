"""Tests for proposed-rule ingest (shadow-by-default, merge-tighten, gates)."""

from __future__ import annotations

from pathlib import Path

import yaml

from refund_abuse_risk.config import load_rule_ingest
from refund_abuse_risk.ops.rule_ingest import (
    apply_gates,
    coerce_live_policy,
    live_allowed,
    merge_tighten_overlay,
    append_rule_if_new,
    parse_proposals,
    run_ingest,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _base_cfg(tmp_path: Path, **overrides: object) -> dict:
    op = {
        "policy_version": "1.0.0",
        "decision_thresholds": {
            "soft_friction": 40,
            "hold_review": 55,
            "auto_deny": 80,
        },
        "decision_threshold_overlays": [
            {
                "market": "SG",
                "vertical": "food",
                "soft_friction": 40,
                "hold_review": 55,
                "auto_deny": 80,
            }
        ],
    }
    effect = {
        "version": "0.1.0",
        "enabled": True,
        "rules": [{"id": "existing_rule", "mode": "shadow", "when": {}, "effect": "refund_step_up"}],
        "challenge_rules": [],
    }
    op_path = tmp_path / "operating_point.yaml"
    effect_path = tmp_path / "effect_rules.yaml"
    _write_yaml(op_path, op)
    _write_yaml(effect_path, effect)
    cfg = {
        "version": 1,
        "enabled": True,
        "live_enabled": False,
        "sources": ["proposed/*.yaml"],
        "gates": {
            "min_z": 3.0,
            "min_recon_z": 0.0,
            "min_edge_orders": 0,
            "max_proposals": 50,
            "max_overlay_proposals": 50,
            "max_challenge_proposals": 50,
        },
        "targets": {
            "operating_point": str(op_path),
            "effect_rules": str(effect_path),
            "backup_dir": str(tmp_path / "backups"),
            "backup_keep": 5,
        },
    }
    cfg.update(overrides)
    return cfg


def test_load_rule_ingest_default_disabled() -> None:
    cfg = load_rule_ingest()
    assert cfg["enabled"] is False
    assert cfg["live_enabled"] is False
    assert float(cfg["gates"]["min_z"]) == 3.0


def test_disabled_is_noop(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, enabled=False)
    prop_dir = tmp_path / "proposed"
    _write_yaml(
        prop_dir / "one.yaml",
        {
            "proposed_rules": [
                {
                    "kind": "decision_threshold_overlay",
                    "market": "SG",
                    "vertical": "food",
                    "soft_friction": 30,
                    "mode": "live",
                }
            ]
        },
    )
    summary = run_ingest(cfg=cfg, root=tmp_path, dry_run=False)
    assert summary["skipped"] is True
    assert summary["reason"] == "disabled"
    op = yaml.safe_load((tmp_path / "operating_point.yaml").read_text(encoding="utf-8"))
    assert op["decision_threshold_overlays"][0]["soft_friction"] == 40


def test_shadow_write_coerces_live(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, enabled=True, live_enabled=False)
    _write_yaml(
        tmp_path / "proposed" / "one.yaml",
        {
            "proposed_rules": [
                {
                    "kind": "effect_rule",
                    "id": "new_shadow_rule",
                    "mode": "live",
                    "auto_enforce": True,
                    "when": {"min_decision_score": 90},
                    "effect": "refund_block",
                    "reason_code": "EFFECT_TEST",
                }
            ]
        },
    )
    summary = run_ingest(cfg=cfg, root=tmp_path, dry_run=False, env={})
    assert summary["skipped"] is False
    assert summary["n_accepted"] == 1
    assert summary["accepted"][0]["mode"] == "shadow"
    effect = yaml.safe_load((tmp_path / "effect_rules.yaml").read_text(encoding="utf-8"))
    new = [r for r in effect["rules"] if r["id"] == "new_shadow_rule"][0]
    assert new["mode"] == "shadow"
    assert (tmp_path / "backups").exists()
    assert list((tmp_path / "backups").glob("effect_rules.*.yaml"))


def test_live_flag_allows_mode_live(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, enabled=True, live_enabled=False)
    _write_yaml(
        tmp_path / "proposed" / "one.yaml",
        {
            "proposed_rules": [
                {
                    "kind": "challenge_rule",
                    "id": "live_challenge",
                    "mode": "live",
                    "when": {"tiers": ["soft_friction"]},
                    "challenge": "payment_verify",
                    "reason_code": "CHALLENGE_TEST",
                }
            ]
        },
    )
    summary = run_ingest(
        cfg=cfg, root=tmp_path, dry_run=False, env={"INGEST_RULES_LIVE": "1"}
    )
    assert summary["live_allowed"] is True
    assert summary["accepted"][0]["mode"] == "live"
    effect = yaml.safe_load((tmp_path / "effect_rules.yaml").read_text(encoding="utf-8"))
    assert effect["challenge_rules"][-1]["mode"] == "live"


def test_merge_tighten_never_loosens() -> None:
    op = {
        "decision_thresholds": {
            "soft_friction": 40,
            "hold_review": 55,
            "auto_deny": 80,
        },
        "decision_threshold_overlays": [
            {
                "market": "SG",
                "vertical": "food",
                "soft_friction": 40,
                "hold_review": 55,
                "auto_deny": 80,
            }
        ],
    }
    op2, meta = merge_tighten_overlay(
        op,
        {
            "kind": "decision_threshold_overlay",
            "market": "SG",
            "vertical": "food",
            "soft_friction": 50,
        },
    )
    assert meta["accepted"] is True
    ov = op2["decision_threshold_overlays"][0]
    assert ov["soft_friction"] == 40  # looser 50 rejected / min stays 40

    op3, meta3 = merge_tighten_overlay(
        op2,
        {
            "kind": "decision_threshold_overlay",
            "market": "sg",
            "vertical": "FOOD",
            "soft_friction": 30,
        },
    )
    assert meta3["accepted"] is True
    assert op3["decision_threshold_overlays"][0]["soft_friction"] == 30


def test_skip_existing_effect_id() -> None:
    rules = [{"id": "existing_rule", "mode": "shadow", "when": {}, "effect": "refund_step_up"}]
    rules2, appended = append_rule_if_new(
        rules,
        {
            "id": "existing_rule",
            "mode": "live",
            "when": {"min_decision_score": 1},
            "effect": "refund_block",
        },
    )
    assert appended is False
    assert rules2 == rules

    rules3, appended3 = append_rule_if_new(
        rules,
        {
            "id": "brand_new",
            "mode": "shadow",
            "when": {},
            "effect": "refund_manual_review",
            "reason_code": "X",
        },
    )
    assert appended3 is True
    assert len(rules3) == 2


def test_gates_drop_low_z_and_cap() -> None:
    props = [
        {"kind": "decision_threshold_overlay", "z": 2.0, "market": "SG", "vertical": "food"},
        {"kind": "decision_threshold_overlay", "z": 4.0, "market": "ID", "vertical": "food"},
        {"kind": "decision_threshold_overlay", "z": 5.0, "market": "MY", "vertical": "food"},
        {"kind": "challenge_rule", "id": "c1", "z": 4.0},
    ]
    kept, skipped = apply_gates(
        props,
        {
            "min_z": 3.0,
            "min_recon_z": 0.0,
            "min_edge_orders": 0,
            "max_proposals": 2,
            "max_overlay_proposals": 1,
            "max_challenge_proposals": 50,
        },
    )
    assert any("min_z" in str(s.get("skip_reason")) for s in skipped)
    assert len(kept) == 2
    assert kept[0]["market"] == "ID"
    # second overlay hits max_overlay_proposals; challenge fills remaining max_proposals slot
    assert kept[1]["kind"] == "challenge_rule"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, enabled=True)
    _write_yaml(
        tmp_path / "proposed" / "one.yaml",
        {
            "proposed_rules": [
                {
                    "kind": "decision_threshold_overlay",
                    "market": "SG",
                    "vertical": "food",
                    "suggested": {"soft_friction_delta": -5},
                    "mode": "shadow",
                }
            ]
        },
    )
    summary = run_ingest(cfg=cfg, root=tmp_path, dry_run=True, env={})
    assert summary["n_accepted"] == 1
    assert summary["dry_run"] is True
    assert not (tmp_path / "backups").exists()
    op = yaml.safe_load((tmp_path / "operating_point.yaml").read_text(encoding="utf-8"))
    assert op["decision_threshold_overlays"][0]["soft_friction"] == 40


def test_live_allowed_helpers() -> None:
    assert live_allowed({"live_enabled": False}, env={}) is False
    assert live_allowed({"live_enabled": True}, env={}) is True
    assert live_allowed({"live_enabled": False}, env={"INGEST_RULES_LIVE": "1"}) is True
    out = coerce_live_policy(
        [{"mode": "live", "auto_enforce": True}],
        live_ok=False,
    )
    assert out[0]["mode"] == "shadow"
    assert out[0]["auto_enforce"] is False


def test_parse_proposed_rules_and_typed_lists() -> None:
    doc = {
        "proposed_rules": [{"kind": "decision_threshold_overlay", "market": "SG", "vertical": "food"}],
        "rules": [{"id": "r1", "when": {}, "effect": "refund_block"}],
    }
    props = parse_proposals(doc, source_path="x.yaml")
    kinds = {p["kind"] for p in props}
    assert "decision_threshold_overlay" in kinds
    assert "effect_rule" in kinds


def test_default_config_cli_disabled_exit_zero() -> None:
    import json
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ingest_proposed_rules.py"),
            "--config",
            str(ROOT / "config" / "rule_ingest.default.yaml"),
            "--dry-run",
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["skipped"] is True
    assert summary["reason"] == "disabled"
