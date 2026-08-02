from __future__ import annotations

from collections import defaultdict
from typing import Any


def link_key(kind: str, *parts: str) -> str:
    return f"{kind}:" + "|".join(str(p) for p in parts)


class EntityGraphIndex:
    """In-memory index of open orders by entity/link for risk-change rescoring."""

    def __init__(self) -> None:
        self._by_key: dict[str, set[str]] = defaultdict(set)
        self._order_keys: dict[str, set[str]] = defaultdict(set)

    def keys_for_order(self, order: dict[str, Any]) -> set[str]:
        user_id = str(order.get("user_id", ""))
        driver_id = str(order.get("driver_id", ""))
        vendor_id = str(order.get("vendor_id", ""))
        device_id = str(order.get("device_id", ""))
        keys = {
            link_key("user", user_id),
            link_key("driver", driver_id),
            link_key("vendor", vendor_id),
            link_key("device", device_id),
            link_key("ud", user_id, driver_id),
            link_key("uv", user_id, vendor_id),
            link_key("vd", vendor_id, driver_id),
            link_key("uvd", user_id, vendor_id, driver_id),
        }
        return {k for k in keys if not k.endswith(":") and not k.endswith("|")}

    def index_order(self, order_id: str, order: dict[str, Any]) -> None:
        keys = self.keys_for_order(order)
        self._order_keys[order_id] = keys
        for key in keys:
            self._by_key[key].add(order_id)

    def remove_order(self, order_id: str) -> None:
        for key in self._order_keys.pop(order_id, set()):
            self._by_key[key].discard(order_id)

    def orders_for_key(self, key: str) -> set[str]:
        return set(self._by_key.get(key, set()))

    def orders_touched_by_keys(self, keys: set[str]) -> set[str]:
        out: set[str] = set()
        for key in keys:
            out |= self._by_key.get(key, set())
        return out
