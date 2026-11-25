"""Tamper-Evident SQLite WAL Clinical Audit Log & Reversible PHI Token Vault."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

GENESIS_HASH = "GENESIS_HEALTHSCRIBEFHIR_SQLITE_0000"


class SQLiteClinicalAuditStore:
    """Append-only SHA-256 hash-chained SQLite WAL store for HIPAA audit events and PHI vault tokens."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        default_dir = Path(os.environ.get("HEALTHSCRIBE_RUNTIME_DIR", "/tmp/healthscribefhir-runtime"))
        default_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else (default_dir / "clinical_audit.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clinical_audit_log (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL UNIQUE,
                        timestamp_utc TEXT NOT NULL,
                        encounter_id TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        verdict TEXT NOT NULL,
                        duration_ms REAL NOT NULL,
                        details_json TEXT NOT NULL,
                        prev_hash TEXT NOT NULL,
                        sha256_hash TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS phi_token_vault (
                        token TEXT PRIMARY KEY,
                        category TEXT NOT NULL,
                        raw_value TEXT NOT NULL,
                        created_utc TEXT NOT NULL
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _compute_hash(payload: Dict[str, Any], prev_hash: str) -> str:
        canonical = json.dumps({"payload": payload, "prev_hash": prev_hash}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append_event(
        self,
        encounter_id: str,
        action_type: str,
        verdict: str,
        duration_ms: float,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT seq, sha256_hash FROM clinical_audit_log ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                next_seq = (int(row["seq"]) + 1) if row else 1
                prev_hash = str(row["sha256_hash"]) if row else GENESIS_HASH
                event_id = f"CLN-AUD-{next_seq:05d}"
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                details_str = json.dumps(details, sort_keys=True, separators=(",", ":"))
                payload = {
                    "event_id": event_id,
                    "timestamp_utc": ts,
                    "encounter_id": encounter_id,
                    "action_type": action_type,
                    "verdict": verdict,
                    "duration_ms": round(float(duration_ms), 3),
                    "details_json": details_str,
                }
                digest = self._compute_hash(payload, prev_hash)
                conn.execute(
                    """
                    INSERT INTO clinical_audit_log (
                        event_id, timestamp_utc, encounter_id, action_type, verdict,
                        duration_ms, details_json, prev_hash, sha256_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        ts,
                        encounter_id,
                        action_type,
                        verdict,
                        payload["duration_ms"],
                        details_str,
                        prev_hash,
                        digest,
                    ),
                )
                conn.commit()
                return {**payload, "prev_hash": prev_hash, "sha256_hash": digest}
            finally:
                conn.close()

    def store_vault_tokens(self, mappings: List[Dict[str, str]]) -> None:
        if not mappings:
            return
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO phi_token_vault (token, category, raw_value, created_utc)
                    VALUES (?, ?, ?, ?)
                    """,
                    [(m["token"], m["category"], m["raw_value"], ts) for m in mappings],
                )
                conn.commit()
            finally:
                conn.close()

    def resolve_vault_token(self, token: str) -> Optional[Dict[str, str]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT token, category, raw_value, created_utc FROM phi_token_vault WHERE token = ?",
                    (token,),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def list_recent_events(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT event_id, timestamp_utc, encounter_id, action_type, verdict,
                           duration_ms, details_json, prev_hash, sha256_hash
                    FROM clinical_audit_log
                    ORDER BY seq DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                out: List[Dict[str, Any]] = []
                for r in rows:
                    d = dict(r)
                    d["details"] = json.loads(d.pop("details_json") or "{}")
                    out.append(d)
                return out
            finally:
                conn.close()

    def verify_chain_integrity(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT seq, event_id, timestamp_utc, encounter_id, action_type, verdict,
                           duration_ms, details_json, prev_hash, sha256_hash
                    FROM clinical_audit_log
                    ORDER BY seq ASC
                    """
                ).fetchall()
                expected_prev = GENESIS_HASH
                for r in rows:
                    if r["prev_hash"] != expected_prev:
                        return {
                            "valid": False,
                            "checked_rows": len(rows),
                            "broken_at_event_id": r["event_id"],
                            "reason": "prev_hash_mismatch",
                            "verify_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                        }
                    payload = {
                        "event_id": r["event_id"],
                        "timestamp_utc": r["timestamp_utc"],
                        "encounter_id": r["encounter_id"],
                        "action_type": r["action_type"],
                        "verdict": r["verdict"],
                        "duration_ms": round(float(r["duration_ms"]), 3),
                        "details_json": r["details_json"],
                    }
                    recomputed = self._compute_hash(payload, expected_prev)
                    if recomputed != r["sha256_hash"]:
                        return {
                            "valid": False,
                            "checked_rows": len(rows),
                            "broken_at_event_id": r["event_id"],
                            "reason": "sha256_digest_mismatch",
                            "verify_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                        }
                    expected_prev = r["sha256_hash"]
                return {
                    "valid": True,
                    "checked_rows": len(rows),
                    "genesis_hash": GENESIS_HASH,
                    "head_hash": expected_prev,
                    "verify_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }
            finally:
                conn.close()

    def reset(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM clinical_audit_log;")
                conn.execute("DELETE FROM phi_token_vault;")
                conn.execute("DELETE FROM sqlite_sequence WHERE name='clinical_audit_log';")
                conn.commit()
            finally:
                conn.close()
