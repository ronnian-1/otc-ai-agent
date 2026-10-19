"""Session-local evidence from executed local tools, never from chat text.

Query scopes are exact: neither fuzzy searches nor comparisons become a global
catalog. A local JSON version and a short TTL bound reuse, not real-time truth.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from app.tools.data_loader import SKU_DATA_PATH


PRODUCT_FIELDS = (
    "name", "category", "spec", "price", "stock", "delivery",
    "suitable_crowd", "taboo", "product_desc",
)
SKU_RE = re.compile(r"S\d{3,}", re.I)

# Cache timestamps describe our observation/reuse window, not business freshness.
LOCAL_EVIDENCE_SCOPE = {
    "data_updated_at": None,
    "business_freshness": "unverified",
    "expiry_meaning": "local_cache_reuse_only",
    "cannot_verify": ["live_stock", "live_promotion", "platform_policy", "order_status"],
}


@dataclass(frozen=True)
class ToolFact:
    sku: str
    field: str
    value: Any
    source_tool: str
    query_args: dict[str, Any]
    observed_at: float
    expires_at: float
    source_path: str
    source_version: str
    success: bool
    missing: bool
    source_kind: str = "local_json_snapshot"


@dataclass(frozen=True)
class ToolQuery:
    source_tool: str
    query_args: dict[str, Any]
    status: str
    facts: tuple[ToolFact, ...]
    result: dict[str, Any] | None
    observed_at: float
    expires_at: float
    source_version: str | None
    error: str | None = None


class SessionFacts:
    """Owned by one SessionMemory. Only the tool-execution boundary writes here.

    Failure/missing results replace the previous record in the same query scope.
    Diagnostic records may be inspected after expiry; fact reads always validate
    both age and source version. Returned records are copies.
    """

    def __init__(
        self, *, source_path: Path = SKU_DATA_PATH, ttl_seconds: float = 300,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.source_path = Path(source_path)
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._queries: dict[str, ToolQuery] = {}
        self._lock = RLock()

    @staticmethod
    def _key(tool: str, args: dict[str, Any]) -> str:
        return json.dumps([tool, args], sort_keys=True, ensure_ascii=False)

    def source_version(self) -> str | None:
        try:
            return hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        except OSError:
            return None

    @staticmethod
    def _rows(tool: str, args: dict[str, Any], result: Any) -> list[dict[str, Any]]:
        """Validate tool-specific ownership before assigning any field to a SKU."""
        if not isinstance(result, dict) or "error" in result:
            raise ValueError("Tool failed or returned a non-object result")
        if tool == "search_product":
            query = args.get("query")
            if set(args) != {"query"} or not isinstance(query, str) or not query.strip():
                raise ValueError("Unsupported search conditions")
            rows = result.get("results")
            if result.get("query") != query or not isinstance(rows, list):
                raise ValueError("Malformed search response")
            exact = query.strip().upper() if SKU_RE.fullmatch(query.strip()) else None
            seen = set()
            for row in rows:
                sku = row.get("sku_id") if isinstance(row, dict) else None
                if not isinstance(sku, str) or not SKU_RE.fullmatch(sku):
                    raise ValueError("Missing or invalid SKU identity")
                if sku.upper() in seen or (exact and sku.upper() != exact):
                    raise ValueError("Search SKU does not match query ownership")
                seen.add(sku.upper())
            return rows
        if tool == "compare_product":
            raw = args.get("sku_ids")
            if set(args) != {"sku_ids"} or not isinstance(raw, str):
                raise ValueError("Unsupported comparison conditions")
            ids = [s.strip().upper() for s in raw.split(",") if s.strip()]
            returned = result.get("sku_ids")
            if (len(ids) != 2 or len(set(ids)) != 2
                    or not all(SKU_RE.fullmatch(s) for s in ids)
                    or not isinstance(returned, list)
                    or [str(s).upper() for s in returned] != ids):
                raise ValueError("Comparison order or identity mismatch")
            rows = []
            basic = result.get("basic_difference", {})
            if not isinstance(basic, dict):
                raise ValueError("Malformed comparison fields")
            for sku, side in zip(ids, ("sku_a", "sku_b")):
                row = {"sku_id": sku}
                for field in PRODUCT_FIELDS[:6]:
                    block = basic.get(field, {})
                    if not isinstance(block, dict):
                        raise ValueError("Malformed basic field")
                    row[field] = block.get(side)
                for field, block_name in (
                    ("suitable_crowd", "suitable_crowd_difference"),
                    ("taboo", "taboo_difference"),
                    ("product_desc", "product_position_difference"),
                ):
                    block = result.get(block_name, {})
                    if not isinstance(block, dict) or not isinstance(block.get(side, {}), dict):
                        raise ValueError("Malformed comparison side")
                    entry = block.get(side, {})
                    if entry and str(entry.get("sku_id", "")).upper() != sku:
                        raise ValueError("Comparison field belongs to another SKU")
                    row[field] = entry.get("value")
                rows.append(row)
            return rows
        raise ValueError("Unregistered fact source")

    def record_result(
        self, tool: str, args: dict[str, Any], result: Any, *,
        version_before: str | None, version_after: str | None,
        execution_error: str | None = None,
    ) -> ToolQuery:
        """Called after actual execution, including exceptions and parse failures."""
        now = self._clock()
        status, error, facts = "success", execution_error, []
        try:
            if execution_error:
                raise ValueError(execution_error)
            if not version_before or version_before != version_after:
                raise ValueError("Local source unavailable or changed during execution")
            rows = self._rows(tool, args, result)
            if not rows:
                status = "empty"
            for row in rows:
                for field in PRODUCT_FIELDS:
                    value = row.get(field)
                    missing = (not isinstance(value, (str, int, float))
                               or isinstance(value, bool)
                               or (isinstance(value, str) and not value.strip()))
                    if missing:
                        status = "partial"
                    facts.append(ToolFact(
                        sku=row["sku_id"].upper(), field=field,
                        value=None if missing else deepcopy(value), source_tool=tool,
                        query_args=deepcopy(args), observed_at=now,
                        expires_at=now + self.ttl_seconds, source_path=str(self.source_path),
                        source_version=version_before, success=not missing, missing=missing,
                    ))
        except ValueError as exc:
            status, error, facts = "error", str(exc), []
        record = ToolQuery(
            source_tool=tool, query_args=deepcopy(args), status=status, facts=tuple(facts),
            result=deepcopy(result) if status in ("success", "partial") else None,
            observed_at=now, expires_at=now + self.ttl_seconds,
            source_version=version_after, error=error,
        )
        with self._lock:
            self._queries[self._key(tool, args)] = record
        return deepcopy(record)

    def get_query(self, tool: str, args: dict[str, Any]) -> ToolQuery | None:
        """Diagnostics, including failures; not an unchecked cache read."""
        with self._lock:
            return deepcopy(self._queries.get(self._key(tool, args)))

    def _fresh(self, query: ToolQuery) -> bool:
        return (query.observed_at <= self._clock() < query.expires_at
                and query.source_version is not None
                and query.source_version == self.source_version())

    def get_fact(
        self, sku: str, field: str, *, tool: str, args: dict[str, Any],
    ) -> ToolFact | None:
        query = self.get_query(tool, args)
        if query is not None and self._fresh(query):
            return next((f for f in query.facts
                         if f.sku == sku.upper() and f.field == field and f.success), None)
        return None

    def valid_query_facts(self, tool: str, args: dict[str, Any]) -> tuple[ToolFact, ...]:
        """Read successful fields in one exact, still-valid tool query scope."""
        query = self.get_query(tool, args)
        if query is None or not self._fresh(query):
            return ()
        return tuple(f for f in query.facts if f.success and not f.missing)

    def reusable_result(self, tool: str, args: dict[str, Any]) -> ToolQuery | None:
        """Only complete exact-SKU searches; compare/fuzzy/missing always requery.

        The caller additionally bypasses reuse for stock/delivery, explicit
        freshness requests, and fields without a supported local fact contract.
        """
        query_text = args.get("query")
        if (tool != "search_product" or set(args) != {"query"}
                or not isinstance(query_text, str)
                or not SKU_RE.fullmatch(query_text.strip())):
            return None
        query = self.get_query(tool, args)
        if query is not None and query.status == "success" and self._fresh(query):
            return query
        return None

    def discard_queries(self, scopes) -> None:
        """Quarantine only query scopes touched by an aborted invocation."""
        with self._lock:
            for tool, args in scopes:
                self._queries.pop(self._key(tool, args), None)

    def clear(self) -> None:
        with self._lock:
            self._queries.clear()
