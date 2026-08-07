"""Ingest machine-proposed rules into OP overlays / effect_rules (shadow by default)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from refund_abuse_risk.config import load_yaml

# OP ladder keys (proposal may use deny as alias for auto_deny).
_LADDER_KEYS = ("soft_friction", "hold_review", "auto_deny")
_DELTA_KEYS = {
    "soft_friction_delta": "soft_friction",
    "hold_review_delta": "hold_review",
    "auto_deny_delta": "auto_deny",
    "deny_delta": "auto_deny",
}
_KIND_OVERLAY = "decision_threshold_overlay"
_KIND_EFFECT = "effect_rule"
_KIND_CHALLENGE = "challenge_rule"
_KNOWN_KINDS = {_KIND_OVERLAY, _KIND_EFFECT, _KIND_CHALLENGE}
_TYPED_LIST_KINDS = {
    "decision_threshold_overlays": _KIND_OVERLAY,
    "rules": _KIND_EFFECT,
    "challenge_rules": _KIND_CHALLENGE,
}


def live_allowed(cfg: dict[str, Any], *, env: dict[str, str] | None = None) -> bool:
    env = env if env is not None else os.environ
    if str(env.get("INGEST_RULES_LIVE", "")).strip() in {"1", "true", "TRUE", "yes"}:
        return True
    return bool(cfg.get("live_enabled"))


def discover_proposal_files(sources: list[Any], *, root: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for raw in sources or []:
        pattern = str(raw)
        base = Path(pattern)
        if base.is_absolute():
            matches = sorted(base.parent.glob(base.name)) if any(c in pattern for c in "*?[") else (
                [base] if base.is_file() else []
            )
        else:
            matches = sorted(root.glob(pattern))
        for path in matches:
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
    return out


def _stable_id(proposal: dict[str, Any], *, source_path: str) -> str:
    existing = proposal.get("id")
    if existing is not None and str(existing).strip():
        return str(existing).strip()
    market = str(proposal.get("market") or "").upper()
    vertical = str(proposal.get("vertical") or "").lower()
    kind = str(proposal.get("kind") or "unknown")
    source = str(proposal.get("source") or Path(source_path).stem)
    return f"{source}|{market}|{vertical}|{kind}"


def parse_proposals(doc: Any, *, source_path: str) -> list[dict[str, Any]]:
    """Flatten proposed_rules / typed lists / single kind object."""
    out: list[dict[str, Any]] = []

    def _attach(item: dict[str, Any], *, kind: str | None = None) -> None:
        prop = dict(item)
        if kind and not prop.get("kind"):
            prop["kind"] = kind
        prop["source_path"] = source_path
        if prop.get("kind") in {_KIND_EFFECT, _KIND_CHALLENGE} or prop.get("id"):
            prop["id"] = _stable_id(prop, source_path=source_path)
        elif prop.get("kind") == _KIND_OVERLAY and not prop.get("id"):
            prop["id"] = _stable_id(prop, source_path=source_path)
        out.append(prop)

    if isinstance(doc, list):
        for item in doc:
            if isinstance(item, dict):
                _attach(item)
        return out

    if not isinstance(doc, dict):
        return out

    if isinstance(doc.get("proposed_rules"), list):
        for item in doc["proposed_rules"]:
            if isinstance(item, dict):
                _attach(item)

    for key, kind in _TYPED_LIST_KINDS.items():
        items = doc.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                _attach(item, kind=kind)

    # Single proposal object with kind at top level (and not a wrapper doc).
    if doc.get("kind") in _KNOWN_KINDS and "proposed_rules" not in doc:
        has_typed = any(isinstance(doc.get(k), list) for k in _TYPED_LIST_KINDS)
        if not has_typed:
            _attach(doc)

    return out


def _gate_numeric(proposal: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in proposal and proposal[key] is not None:
            try:
                return float(proposal[key])
            except (TypeError, ValueError):
                continue
        suggested = proposal.get("suggested")
        if isinstance(suggested, dict) and key in suggested and suggested[key] is not None:
            try:
                return float(suggested[key])
            except (TypeError, ValueError):
                continue
    return None


def apply_gates(
    proposals: list[dict[str, Any]],
    gates: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (kept, skipped) where skipped entries have reason."""
    min_z = float(gates.get("min_z") or 0.0)
    min_recon_z = float(gates.get("min_recon_z") or 0.0)
    min_edge_orders = int(gates.get("min_edge_orders") or 0)
    max_proposals = int(gates.get("max_proposals") or 50)
    max_overlay = int(gates.get("max_overlay_proposals") or max_proposals)
    max_challenge = int(gates.get("max_challenge_proposals") or max_proposals)

    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    n_overlay = 0
    n_challenge = 0

    for prop in proposals:
        kind = str(prop.get("kind") or "")
        if kind and kind not in _KNOWN_KINDS:
            skipped.append({**prop, "skip_reason": f"unknown_kind:{kind}"})
            continue

        z = _gate_numeric(prop, ("z", "node_z", "edge_z"))
        if z is not None and z < min_z:
            skipped.append({**prop, "skip_reason": f"min_z:{z}<{min_z}"})
            continue

        recon_z = _gate_numeric(prop, ("recon_z",))
        if min_recon_z > 0 and recon_z is not None and recon_z < min_recon_z:
            skipped.append({**prop, "skip_reason": f"min_recon_z:{recon_z}<{min_recon_z}"})
            continue

        edge_n = _gate_numeric(prop, ("edge_n", "n_test_day", "n_orders"))
        if min_edge_orders > 0 and edge_n is not None and edge_n < min_edge_orders:
            skipped.append(
                {**prop, "skip_reason": f"min_edge_orders:{edge_n}<{min_edge_orders}"}
            )
            continue

        if kind == _KIND_OVERLAY and n_overlay >= max_overlay:
            skipped.append({**prop, "skip_reason": "max_overlay_proposals"})
            continue
        if kind == _KIND_CHALLENGE and n_challenge >= max_challenge:
            skipped.append({**prop, "skip_reason": "max_challenge_proposals"})
            continue
        if len(kept) >= max_proposals:
            skipped.append({**prop, "skip_reason": "max_proposals"})
            continue

        kept.append(prop)
        if kind == _KIND_OVERLAY:
            n_overlay += 1
        elif kind == _KIND_CHALLENGE:
            n_challenge += 1

    return kept, skipped


def coerce_live_policy(
    proposals: list[dict[str, Any]],
    *,
    live_ok: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for prop in proposals:
        item = dict(prop)
        if not live_ok:
            item["mode"] = "shadow"
            if item.get("auto_enforce"):
                item["auto_enforce"] = False
        else:
            mode = str(item.get("mode") or "shadow").lower()
            item["mode"] = mode if mode in {"shadow", "live"} else "shadow"
        out.append(item)
    return out


def _normalize_mv(market: Any, vertical: Any) -> tuple[str, str]:
    return str(market or "").upper(), str(vertical or "").lower()


def _abs_thresholds_from_proposal(proposal: dict[str, Any]) -> dict[str, float]:
    raw: dict[str, Any] = {}
    for key in ("soft_friction", "hold_review", "auto_deny", "deny"):
        if key in proposal and proposal[key] is not None:
            dest = "auto_deny" if key == "deny" else key
            raw[dest] = float(proposal[key])
    return raw


def _apply_deltas(
    base: dict[str, float],
    proposal: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    suggested = proposal.get("suggested") if isinstance(proposal.get("suggested"), dict) else {}
    deltas: dict[str, Any] = {}
    for src, dest in _DELTA_KEYS.items():
        if src in proposal and proposal[src] is not None:
            deltas[dest] = proposal[src]
        elif src in suggested and suggested[src] is not None:
            deltas[dest] = suggested[src]

    rejected: list[str] = []
    out = dict(base)
    for key, delta_raw in deltas.items():
        try:
            delta = float(delta_raw)
        except (TypeError, ValueError):
            rejected.append(key)
            continue
        if key not in base:
            continue
        new_val = float(base[key]) + delta
        # Tighten-only: resulting value must be ≤ base (negative deltas expected).
        if new_val > float(base[key]):
            rejected.append(key)
            continue
        out[key] = new_val
    return out, rejected


def merge_tighten_overlay(
    op: dict[str, Any],
    proposal: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge one overlay proposal into OP. Returns (op, meta)."""
    market, vertical = _normalize_mv(proposal.get("market"), proposal.get("vertical"))
    if not market and not vertical:
        return op, {"accepted": False, "reason": "missing_market_vertical"}

    global_thr = {
        k: float(v)
        for k, v in (op.get("decision_thresholds") or {}).items()
        if k in _LADDER_KEYS and v is not None
    }
    # Compat: some OPs may only have auto_deny under decision_thresholds.
    overlays = [dict(x) for x in (op.get("decision_threshold_overlays") or []) if isinstance(x, dict)]
    existing_idx: int | None = None
    existing: dict[str, Any] | None = None
    for idx, ov in enumerate(overlays):
        om, ovv = _normalize_mv(ov.get("market"), ov.get("vertical"))
        if om == market and ovv == vertical:
            existing_idx = idx
            existing = ov
            break

    base = dict(global_thr)
    if existing is not None:
        for key in _LADDER_KEYS:
            if key in existing and existing[key] is not None:
                base[key] = float(existing[key])

    abs_vals = _abs_thresholds_from_proposal(proposal)
    rejected: list[str] = []
    proposed_vals: dict[str, float] = {}

    if abs_vals:
        for key, val in abs_vals.items():
            if key in base and val > float(base[key]):
                # Absolute looser than base → reject that key (do not loosen).
                rejected.append(key)
            else:
                proposed_vals[key] = val
    else:
        proposed_vals, delta_rej = _apply_deltas(base, proposal)
        rejected.extend(delta_rej)
        # Only keep keys that changed or are ladder keys from deltas/base write intent.
        delta_targets = set()
        suggested = proposal.get("suggested") if isinstance(proposal.get("suggested"), dict) else {}
        for src, dest in _DELTA_KEYS.items():
            if src in proposal or src in suggested:
                delta_targets.add(dest)
        proposed_vals = {k: v for k, v in proposed_vals.items() if k in delta_targets}

    if not proposed_vals and not abs_vals:
        # No absolute and no usable deltas.
        if rejected:
            return op, {"accepted": False, "reason": "loosen_rejected", "rejected_keys": rejected}
        return op, {"accepted": False, "reason": "no_thresholds"}

    if existing is None:
        new_ov: dict[str, Any] = {
            "market": market,
            "vertical": vertical,
            **{k: proposed_vals[k] for k in _LADDER_KEYS if k in proposed_vals},
        }
        mode = str(proposal.get("mode") or "shadow")
        if mode:
            new_ov["mode"] = mode
        overlays.append(new_ov)
        op = dict(op)
        op["decision_threshold_overlays"] = overlays
        return op, {"accepted": True, "action": "append", "rejected_keys": rejected}

    merged = dict(existing)
    merged["market"] = market
    merged["vertical"] = vertical
    for key, val in proposed_vals.items():
        if key in merged and merged[key] is not None:
            merged[key] = min(float(merged[key]), float(val))
        else:
            merged[key] = float(val)
    mode = str(proposal.get("mode") or merged.get("mode") or "shadow")
    if mode:
        merged["mode"] = mode
    overlays[existing_idx] = merged  # type: ignore[index]
    op = dict(op)
    op["decision_threshold_overlays"] = overlays
    return op, {"accepted": True, "action": "merge_tighten", "rejected_keys": rejected}


def append_rule_if_new(
    rules: list[dict[str, Any]],
    proposal: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    rid = str(proposal.get("id") or "").strip()
    if not rid:
        return rules, False
    existing_ids = {str(r.get("id")) for r in rules if isinstance(r, dict)}
    if rid in existing_ids:
        return rules, False
    entry = {
        "id": rid,
        "mode": str(proposal.get("mode") or "shadow"),
        "when": dict(proposal.get("when") or {}),
        "reason_code": proposal.get("reason_code"),
    }
    if "effect" in proposal:
        entry["effect"] = proposal["effect"]
    if "challenge" in proposal:
        entry["challenge"] = proposal["challenge"]
    # Preserve extra when/reason fields already copied; drop None reason_code.
    if entry.get("reason_code") is None:
        entry.pop("reason_code", None)
    return list(rules) + [entry], True


def _import_backup_helpers():
    """Load promote_overlays backup/bump without making scripts a package."""
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "promote_overlays.py"
    spec = importlib.util.spec_from_file_location("promote_overlays", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.backup_op, mod.bump_policy_version


def _resolve_target(path_str: str, *, root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else root / p


def run_ingest(
    *,
    cfg: dict[str, Any],
    root: Path,
    dry_run: bool = False,
    strict: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    if not bool(cfg.get("enabled")):
        return {
            "skipped": True,
            "reason": "disabled",
            "dry_run": bool(dry_run),
            "accepted": [],
            "skipped_proposals": [],
        }

    sources = list(cfg.get("sources") or [])
    gates = dict(cfg.get("gates") or {})
    targets = dict(cfg.get("targets") or {})
    op_path = _resolve_target(
        str(targets.get("operating_point") or "config/operating_point.default.yaml"),
        root=root,
    )
    effect_path = _resolve_target(
        str(targets.get("effect_rules") or "config/effect_rules.default.yaml"),
        root=root,
    )
    backup_dir = _resolve_target(str(targets.get("backup_dir") or "config/backups"), root=root)
    backup_keep = int(targets.get("backup_keep") or 5)

    files = discover_proposal_files(sources, root=root)
    proposals: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for path in files:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — surface in summary / strict
            parse_errors.append(f"{path}: {exc}")
            continue
        proposals.extend(parse_proposals(raw, source_path=str(path)))

    if parse_errors and strict:
        return {
            "ok": False,
            "error": "parse_failure",
            "parse_errors": parse_errors,
            "dry_run": bool(dry_run),
        }

    gated, gate_skipped = apply_gates(proposals, gates)
    live_ok = live_allowed(cfg, env=env_map)
    gated = coerce_live_policy(gated, live_ok=live_ok)

    op = load_yaml(op_path) if op_path.is_file() else {}
    effect = load_yaml(effect_path) if effect_path.is_file() else {}
    rules = [dict(x) for x in (effect.get("rules") or []) if isinstance(x, dict)]
    challenge_rules = [
        dict(x) for x in (effect.get("challenge_rules") or []) if isinstance(x, dict)
    ]

    accepted: list[dict[str, Any]] = []
    skipped_proposals: list[dict[str, Any]] = list(gate_skipped)
    op_dirty = False
    effect_dirty = False

    for prop in gated:
        kind = str(prop.get("kind") or "")
        if kind == _KIND_OVERLAY:
            op, meta = merge_tighten_overlay(op, prop)
            if meta.get("accepted"):
                op_dirty = True
                accepted.append(
                    {
                        "id": prop.get("id"),
                        "kind": kind,
                        "market": prop.get("market"),
                        "vertical": prop.get("vertical"),
                        "mode": prop.get("mode"),
                        "action": meta.get("action"),
                    }
                )
            else:
                skipped_proposals.append({**prop, "skip_reason": meta.get("reason")})
        elif kind == _KIND_EFFECT:
            if not prop.get("id"):
                skipped_proposals.append({**prop, "skip_reason": "missing_id"})
                continue
            rules, appended = append_rule_if_new(rules, prop)
            if appended:
                effect_dirty = True
                accepted.append(
                    {"id": prop.get("id"), "kind": kind, "mode": prop.get("mode"), "action": "append"}
                )
            else:
                skipped_proposals.append({**prop, "skip_reason": "id_exists"})
        elif kind == _KIND_CHALLENGE:
            if not prop.get("id"):
                skipped_proposals.append({**prop, "skip_reason": "missing_id"})
                continue
            challenge_rules, appended = append_rule_if_new(challenge_rules, prop)
            if appended:
                effect_dirty = True
                accepted.append(
                    {"id": prop.get("id"), "kind": kind, "mode": prop.get("mode"), "action": "append"}
                )
            else:
                skipped_proposals.append({**prop, "skip_reason": "id_exists"})
        else:
            skipped_proposals.append({**prop, "skip_reason": "unknown_kind"})

    unknown_or_gate = [
        s
        for s in skipped_proposals
        if str(s.get("skip_reason") or "").startswith("unknown_kind")
        or str(s.get("skip_reason") or "").startswith("min_")
        or str(s.get("skip_reason") or "") in {"max_proposals", "max_overlay_proposals", "max_challenge_proposals"}
    ]
    strict_fail = bool(strict and unknown_or_gate)

    summary: dict[str, Any] = {
        "ok": not strict_fail,
        "skipped": False,
        "dry_run": bool(dry_run),
        "live_allowed": live_ok,
        "n_sources": len(files),
        "n_proposals": len(proposals),
        "n_accepted": len(accepted),
        "n_skipped": len(skipped_proposals),
        "accepted": accepted,
        "skipped_proposals": [
            {
                "id": s.get("id"),
                "kind": s.get("kind"),
                "skip_reason": s.get("skip_reason"),
                "source_path": s.get("source_path"),
            }
            for s in skipped_proposals
        ],
        "parse_errors": parse_errors,
        "op_path": str(op_path),
        "effect_rules_path": str(effect_path),
        "op_mutated": op_dirty,
        "effect_rules_mutated": effect_dirty,
    }

    if dry_run or (not op_dirty and not effect_dirty):
        if strict_fail:
            summary["error"] = "strict_skipped_proposals"
        return summary

    backup_op, bump_policy_version = _import_backup_helpers()
    if op_dirty:
        backup = backup_op(op_path, backup_dir, keep=backup_keep, prefix="operating_point")
        op["policy_version"] = bump_policy_version(op.get("policy_version"))
        op_path.write_text(
            yaml.safe_dump(op, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        summary["op_backup"] = str(backup)
        summary["policy_version"] = op["policy_version"]
    if effect_dirty:
        backup = backup_op(effect_path, backup_dir, keep=backup_keep, prefix="effect_rules")
        effect = dict(effect)
        effect["rules"] = rules
        effect["challenge_rules"] = challenge_rules
        effect_path.write_text(
            yaml.safe_dump(effect, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        summary["effect_rules_backup"] = str(backup)

    if strict_fail:
        summary["error"] = "strict_skipped_proposals"
    return summary


def summary_to_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, default=str) + "\n"
