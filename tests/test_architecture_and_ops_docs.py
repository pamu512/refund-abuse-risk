"""Architecture / ops surface invariants (supports Architecture & Ops/docs ≥ 9.2)."""

from __future__ import annotations

from pathlib import Path

import yaml

from refund_abuse_risk import config as cfg

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PACKAGES = {
    "schemas",
    "features",
    "graph",
    "labels",
    "model",
    "scoring",
    "pipeline",
    "training",
    "integrations",
    "baselines",
    "control_plane",
}

REQUIRED_DOCS = {
    "docs/ARCHITECTURE.md",
    "docs/OPS_RUNBOOK.md",
    "docs/MANUAL.md",
    "docs/CUTOVER.md",
}


def test_package_map_directories_exist() -> None:
    base = ROOT / "src" / "refund_abuse_risk"
    missing = sorted(p for p in REQUIRED_PACKAGES if not (base / p).is_dir())
    assert not missing, f"missing packages: {missing}"


def test_required_ops_docs_exist_and_cross_link() -> None:
    for rel in REQUIRED_DOCS:
        path = ROOT / rel
        assert path.is_file(), rel
        text = path.read_text(encoding="utf-8")
        assert len(text) > 200, rel
    arch = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "Extension points" in arch
    assert "Intentional boundaries" in arch
    ops = (ROOT / "docs/OPS_RUNBOOK.md").read_text(encoding="utf-8")
    assert "Promote decision tree" in ops
    assert "ops.overnight.yaml" in ops


def test_all_default_configs_load() -> None:
    loaders = [
        cfg.load_policy,
        cfg.load_label_weights,
        cfg.load_operating_point,
        cfg.load_guardrails,
        cfg.load_behavior_baselines,
        cfg.load_bipartite_anomaly,
        cfg.load_disposition_labels,
        cfg.load_effect_rules,
        cfg.load_refund_budget,
        cfg.load_sdk_ingest,
        cfg.load_head_hyperparams,
        cfg.load_vertical_policy,
        cfg.load_feeds,
    ]
    for load in loaders:
        data = load()
        assert isinstance(data, dict) and data, load.__name__


def test_overnight_profile_contract() -> None:
    path = ROOT / "config" / "ops.overnight.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert "paths" in data and "steps" in data
    ids = [s["id"] for s in data["steps"]]
    assert ids[:4] == ["pull_feeds", "train", "eval_oot", "backtest"]
    assert "list_hil" in ids
    assert "promote_overlays" in ids
    assert data["steps"][-1].get("optional") is True
    for step in data["steps"]:
        assert step.get("argv"), step
        for part in step["argv"]:
            if "{" in part:
                key = part.strip("{}")
                assert key in data["paths"], f"placeholder {part} missing from paths"


def test_ops_overnight_dry_run() -> None:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/ops_overnight.py", "--dry-run"],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "pull_feeds" in proc.stdout
    assert "list_hil" in proc.stdout
    assert "done" in proc.stdout


def test_promote_ok_from_metrics_contract(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ops_overnight", ROOT / "scripts" / "ops_overnight.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    missing = tmp_path / "missing.json"
    assert mod.promote_ok_from_metrics(missing) is False

    bad = tmp_path / "bad.json"
    bad.write_text('{"honesty": {"promote_ok": false}}', encoding="utf-8")
    assert mod.promote_ok_from_metrics(bad) is False

    good = tmp_path / "good.json"
    good.write_text('{"honesty": {"promote_ok": true}}', encoding="utf-8")
    assert mod.promote_ok_from_metrics(good) is True
