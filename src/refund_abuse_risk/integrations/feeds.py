"""Hybrid production feed pull → stage → apply (local_dir, sqlite, http, s3)."""

from __future__ import annotations

import json
import shutil
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import yaml

from refund_abuse_risk.config import (
    default_config_dir,
    load_disposition_labels,
    load_sdk_ingest,
    load_yaml,
)
from refund_abuse_risk.integrations.ops_ingest import (
    load_ops_snapshot_file,
    write_ops_snapshot_sidecar,
)
from refund_abuse_risk.integrations.sdk_ingest import apply_sdk_signals_to_orders
from refund_abuse_risk.labels.dispositions import apply_dispositions_to_orders

ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_SOURCE_TYPES = {"local_dir", "sqlite", "http", "s3"}


def load_feeds_config(config_dir: Path | None = None) -> dict[str, Any]:
    return load_yaml((config_dir or default_config_dir()) / "feeds.default.yaml")


def _resolve_path(path: str | Path, *, root: Path = ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root / p)


def _pick_latest(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No extract files matched")
    return max(paths, key=lambda p: p.stat().st_mtime)


def pull_local_dir(source: dict[str, Any], *, root: Path = ROOT) -> Path:
    uri = _resolve_path(source.get("uri") or "", root=root)
    pattern = str(source.get("glob") or "*")
    if not uri.is_dir():
        raise FileNotFoundError(f"Feed uri is not a directory: {uri}")
    matches = sorted(uri.glob(pattern))
    files = [p for p in matches if p.is_file()]
    return _pick_latest(files)


def pull_sqlite(
    feed_name: str,
    source: dict[str, Any],
    *,
    root: Path = ROOT,
    stage_root: Path | None = None,
) -> Path:
    """
    Run ``source.query`` against ``source.uri`` sqlite DB; write extract file.

    ``format``: csv (default) | jsonl | json
    """
    db_path = _resolve_path(source.get("uri") or "", root=root)
    query = str(source.get("query") or "").strip()
    if not query:
        raise ValueError(f"Feed {feed_name}: sqlite source requires query")
    if not db_path.is_file():
        raise FileNotFoundError(f"Feed {feed_name}: sqlite db not found: {db_path}")
    fmt = str(source.get("format") or "csv").strip().lower()
    pull_root = stage_root or _resolve_path("data/feeds/staging", root=root)
    out_dir = pull_root / "_pull" / feed_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(query, conn)
    if fmt == "csv":
        out = out_dir / "extract.csv"
        frame.to_csv(out, index=False)
    elif fmt == "jsonl":
        out = out_dir / "extract.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for row in frame.to_dict(orient="records"):
                f.write(json.dumps(row, default=str) + "\n")
    elif fmt == "json":
        out = out_dir / "extract.json"
        # ops-shaped: single-row metrics → envelope; else list
        if len(frame) == 1 and "metrics" not in frame.columns:
            payload = {
                "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": f"sqlite:{db_path.name}",
                "metrics": {
                    k: float(v)
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    else v
                    for k, v in frame.iloc[0].to_dict().items()
                    if k not in {"as_of", "source"}
                },
            }
            out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        else:
            out.write_text(
                json.dumps(frame.to_dict(orient="records"), indent=2, default=str),
                encoding="utf-8",
            )
    else:
        raise ValueError(f"Feed {feed_name}: unsupported sqlite format {fmt!r}")
    return out


def pull_http(
    feed_name: str,
    source: dict[str, Any],
    *,
    root: Path = ROOT,
    stage_root: Path | None = None,
) -> Path:
    """
    GET ``source.uri`` (http/https); write bytes to staging extract.

    Optional ``headers`` map; ``filename`` overrides extract name (else from URL path).
    """
    uri = str(source.get("uri") or "").strip()
    if not uri:
        raise ValueError(f"Feed {feed_name}: http source requires uri")
    headers = {str(k): str(v) for k, v in (source.get("headers") or {}).items()}
    timeout = float(source.get("timeout_seconds") or 60)
    req = urllib.request.Request(uri, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            content_type = str(resp.headers.get("Content-Type") or "")
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(
            f"Feed {feed_name}: HTTP {exc.code} for {uri}"
        ) from exc
    except urllib.error.URLError as exc:
        raise FileNotFoundError(f"Feed {feed_name}: HTTP error for {uri}: {exc}") from exc

    name = str(source.get("filename") or "").strip()
    if not name:
        path_name = Path(urlparse(uri).path).name or "extract.bin"
        name = path_name
        if "." not in name:
            if "json" in content_type:
                name = "extract.json"
            elif "csv" in content_type:
                name = "extract.csv"
            else:
                name = "extract.bin"
    pull_root = stage_root or _resolve_path("data/feeds/staging", root=root)
    out_dir = pull_root / "_pull" / feed_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    out.write_bytes(body)
    return out


def pull_s3(
    feed_name: str,
    source: dict[str, Any],
    *,
    root: Path = ROOT,
    stage_root: Path | None = None,
) -> Path:
    """
    Pull from S3.

    - ``uri: s3://bucket/key`` requires optional extra ``boto3`` (pip install boto3)
    - ``uri: https://...`` (pre-signed or public) delegates to ``pull_http``
    """
    uri = str(source.get("uri") or "").strip()
    if not uri:
        raise ValueError(f"Feed {feed_name}: s3 source requires uri")
    if uri.startswith("http://") or uri.startswith("https://"):
        return pull_http(feed_name, source, root=root, stage_root=stage_root)
    if not uri.startswith("s3://"):
        raise ValueError(f"Feed {feed_name}: s3 uri must be s3:// or https:// (got {uri!r})")

    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise NotImplementedError(
            f"Feed {feed_name}: s3:// URIs require boto3 "
            "(pip install boto3) or use a pre-signed https:// uri"
        ) from exc

    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Feed {feed_name}: invalid s3 uri {uri!r}")
    region = source.get("region")
    client_kw: dict[str, Any] = {}
    if region:
        client_kw["region_name"] = str(region)
    client = boto3.client("s3", **client_kw)
    pull_root = stage_root or _resolve_path("data/feeds/staging", root=root)
    out_dir = pull_root / "_pull" / feed_name
    out_dir.mkdir(parents=True, exist_ok=True)
    name = str(source.get("filename") or Path(key).name or "extract.bin")
    out = out_dir / name
    client.download_file(bucket, key, str(out))
    return out


def pull_feed(
    feed_name: str,
    feed_cfg: dict[str, Any],
    *,
    root: Path = ROOT,
    stage_root: Path | None = None,
) -> Path:
    source = feed_cfg.get("source") or {}
    stype = str(source.get("type") or "").strip().lower()
    if stype == "local_dir":
        return pull_local_dir(source, root=root)
    if stype == "sqlite":
        return pull_sqlite(feed_name, source, root=root, stage_root=stage_root)
    if stype == "http":
        return pull_http(feed_name, source, root=root, stage_root=stage_root)
    if stype == "s3":
        return pull_s3(feed_name, source, root=root, stage_root=stage_root)
    raise NotImplementedError(
        f"Feed {feed_name}: source.type={stype!r} not implemented yet "
        f"(supported: {sorted(SUPPORTED_SOURCE_TYPES)})."
    )


def stage_extract(
    feed_name: str,
    extract: Path,
    *,
    stage_root: Path,
    as_of: str | None = None,
) -> Path:
    stamp = as_of or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest_dir = stage_root / feed_name / stamp
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / extract.name
    shutil.copy2(extract, dest)
    return dest


def validate_required_columns(path: Path, required: list[str]) -> None:
    if not required:
        return
    suffix = path.suffix.lower()
    if suffix == ".csv":
        cols = set(pd.read_csv(path, nrows=0).columns.astype(str))
    elif suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            line = next((ln.strip() for ln in f if ln.strip()), "")
        if not line:
            raise ValueError(f"Empty jsonl: {path}")
        cols = set(json.loads(line).keys())
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            cols = set(data[0].keys()) if isinstance(data[0], dict) else set()
        elif isinstance(data, dict):
            cols = set(data.keys()) | set((data.get("metrics") or {}).keys())
        else:
            cols = set()
    else:
        return
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")


def _load_sdk_events(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and "events" in data:
            return pd.DataFrame(data["events"])
        raise ValueError("JSON must be a list or {events: [...]}")
    return pd.read_csv(path)


def apply_staged_feed(
    feed_name: str,
    staged: Path,
    feed_cfg: dict[str, Any],
    feeds_cfg: dict[str, Any],
    *,
    root: Path = ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    apply_cfg = feed_cfg.get("apply") or {}
    kind = str(apply_cfg.get("kind") or feed_name)
    summary: dict[str, Any] = {
        "feed": feed_name,
        "kind": kind,
        "staged": str(staged),
        "dry_run": bool(dry_run),
        "applied": False,
    }

    if kind == "dispositions":
        orders_path = _resolve_path(feeds_cfg.get("orders_path") or "data/orders.csv", root=root)
        out_path = _resolve_path(
            feeds_cfg.get("orders_labeled_out") or "data/orders.labeled.csv", root=root
        )
        if dry_run:
            summary["would_write"] = str(out_path)
            summary["dispositions_rows"] = int(len(pd.read_csv(staged)))
            return summary
        orders = pd.read_csv(orders_path)
        labeled = apply_dispositions_to_orders(
            orders,
            pd.read_csv(staged),
            load_disposition_labels(),
            as_of=apply_cfg.get("as_of"),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        labeled.to_csv(out_path, index=False)
        summary["applied"] = True
        summary["out"] = str(out_path)
        summary["dispositions_applied"] = int(labeled.attrs.get("dispositions_applied", 0))
        return summary

    if kind == "sdk_events":
        orders_path = _resolve_path(feeds_cfg.get("orders_path") or "data/orders.csv", root=root)
        # Prefer enriching labeled if present.
        labeled = _resolve_path(
            feeds_cfg.get("orders_labeled_out") or "data/orders.labeled.csv", root=root
        )
        target = labeled if labeled.exists() else orders_path
        if dry_run:
            summary["would_write"] = str(target)
            summary["events_rows"] = int(len(_load_sdk_events(staged)))
            return summary
        orders = pd.read_csv(target)
        events = _load_sdk_events(staged)
        enriched = apply_sdk_signals_to_orders(orders, events, load_sdk_ingest())
        target.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(target, index=False)
        summary["applied"] = True
        summary["out"] = str(target)
        summary["sdk_events_applied"] = int(enriched.attrs.get("sdk_events_applied", 0))
        return summary

    if kind == "ops_snapshot":
        sidecar = _resolve_path(feeds_cfg.get("ops_sidecar") or "data/ops_snapshot.json", root=root)
        snapshot = load_ops_snapshot_file(staged)
        if dry_run:
            summary["would_write"] = str(sidecar)
            summary["snapshot"] = snapshot
            return summary
        if not snapshot:
            raise ValueError(f"No usable ops metrics in {staged}")
        write_ops_snapshot_sidecar(sidecar, snapshot)
        summary["applied"] = True
        summary["out"] = str(sidecar)
        summary["snapshot"] = snapshot
        return summary

    raise ValueError(f"Unknown apply.kind={kind!r} for feed {feed_name}")


def run_feeds(
    feeds_cfg: dict[str, Any] | None = None,
    *,
    only: list[str] | None = None,
    stage_only: bool = False,
    dry_run: bool = False,
    as_of: str | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    cfg = feeds_cfg if feeds_cfg is not None else load_feeds_config()
    stage_root = _resolve_path(cfg.get("stage_root") or "data/feeds/staging", root=root)
    feeds = cfg.get("feeds") or {}
    names = list(only) if only else list(feeds.keys())
    report: dict[str, Any] = {
        "ok": True,
        "stage_only": bool(stage_only),
        "dry_run": bool(dry_run),
        "feeds": {},
    }
    for name in names:
        if name not in feeds:
            report["ok"] = False
            report["feeds"][name] = {"ok": False, "error": "unknown feed"}
            continue
        fcfg = feeds[name]
        if not fcfg.get("enabled", True):
            report["feeds"][name] = {"ok": True, "skipped": True, "reason": "disabled"}
            continue
        try:
            extract = pull_feed(name, fcfg, root=root, stage_root=stage_root)
            validate_required_columns(extract, list(fcfg.get("required_columns") or []))
            staged = stage_extract(name, extract, stage_root=stage_root, as_of=as_of)
            entry: dict[str, Any] = {
                "ok": True,
                "extract": str(extract),
                "staged": str(staged),
            }
            if not stage_only:
                entry["apply"] = apply_staged_feed(
                    name, staged, fcfg, cfg, root=root, dry_run=dry_run
                )
            report["feeds"][name] = entry
        except Exception as exc:  # noqa: BLE001 — surface per-feed errors in report
            report["ok"] = False
            report["feeds"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return report


def write_overlays_yaml(overlays: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {"decision_threshold_overlays": overlays},
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return path
