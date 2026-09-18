"""markdown_store.py — LangGraph-compatible store backed by a markdown file.

Implements .search() and .get() so it works with GoAT's
_retrieve_memory_from_store() helper. GoAT reads only — never writes.

Markdown format expected:
    # User Memory Store

    ## preferences        ← namespace tuple ("preferences",)

    ### user_123          ← key within namespace
    - style: detailed and scientific
    - diet: vegetarian

    ## memory             ← namespace tuple ("memory",)

    ### user_123
    - User prefers biryani when hungry
    ...

    ## context            ← namespace tuple ("context",)

    ### user_123
    - Upcoming flight: BLR to BOM
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class StoreItem:
    """A single item retrieved from the store — mimics LangGraph's Item."""
    key: str
    value: Any
    namespace: tuple


class MarkdownStore:
    """Read-only LangGraph-compatible store backed by a markdown file.

    Supports:
        - .search(namespace, query=...) → list[StoreItem]
        - .get(namespace, key) → StoreItem | None
        - .put(namespace, key, value) → writes back to markdown (for testing)

    The markdown file is re-read on every call so external edits are
    always visible (no caching).
    """

    def __init__(self, path: str | Path = "memory.md"):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Memory file not found: {self.path}")

    # ── Read operations ──────────────────────────────────────────────────────

    def search(
        self,
        namespace: tuple,
        *,
        query: str = "",
        limit: int = 10,
    ) -> list[StoreItem]:
        """Search for items in a namespace.

        If query is provided, only returns items whose key or value
        contains the query string (case-insensitive substring match).

        Args:
            namespace: Namespace tuple, e.g. ("preferences",) or ("memory",).
            query: Optional search query for substring matching.
            limit: Max number of items to return.

        Returns:
            List of StoreItem objects.
        """
        data = self._parse_markdown()
        ns_key = self._namespace_key(namespace)
        section = data.get(ns_key, {})

        results = []
        # Split query into tokens for word-level matching
        query_tokens = query.lower().split() if query else []

        for item_key, item_value in section.items():
            # Word-level match: any query token must appear in key or value
            if query_tokens:
                searchable = f"{item_key} {item_value}".lower()
                if not any(token in searchable for token in query_tokens):
                    continue
            results.append(StoreItem(key=item_key, value=item_value, namespace=namespace))
            if len(results) >= limit:
                break

        return results

    def get(self, namespace: tuple, key: str) -> Optional[StoreItem]:
        """Get a specific item by namespace and key.

        Args:
            namespace: Namespace tuple, e.g. ("preferences",).
            key: Item key within the namespace.

        Returns:
            StoreItem if found, None otherwise.
        """
        data = self._parse_markdown()
        ns_key = self._namespace_key(namespace)
        section = data.get(ns_key, {})
        value = section.get(key)
        if value is not None:
            return StoreItem(key=key, value=value, namespace=namespace)
        return None

    def put(self, namespace: tuple, key: str, value: dict) -> None:
        """Write an item to the markdown file.

        This is provided for compatibility with LangGraph's store interface
        and for testing. It appends/updates the section in the markdown file.

        Args:
            namespace: Namespace tuple.
            key: Item key.
            value: Value dict to store.
        """
        data = self._parse_markdown()
        ns_key = self._namespace_key(namespace)

        if ns_key not in data:
            data[ns_key] = {}

        data[ns_key][key] = value
        self._write_markdown(data)

    # ── Parsing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _namespace_key(namespace: tuple) -> str:
        """Convert a namespace tuple to a string key for internal storage."""
        return "/".join(namespace)

    def _parse_markdown(self) -> dict[str, dict[str, Any]]:
        """Parse the markdown file into a nested dict.

        Returns:
            {namespace_str: {key: value}} where value is either a dict
            (if parsed from key-value lines) or a list of strings.
        """
        content = self.path.read_text(encoding="utf-8")
        data: dict[str, dict[str, Any]] = {}
        current_ns = None
        current_key = None

        for line in content.splitlines():
            # ## preferences  →  namespace = ("preferences",)
            h2_match = re.match(r"^##\s+(.+)$", line)
            if h2_match:
                current_ns = h2_match.group(1).strip()
                current_key = None
                if current_ns not in data:
                    data[current_ns] = {}
                continue

            # ### user_123  →  key within namespace
            h3_match = re.match(r"^###\s+(.+)$", line)
            if h3_match and current_ns:
                current_key = h3_match.group(1).strip()
                if current_key not in data[current_ns]:
                    data[current_ns][current_key] = {"_lines": []}
                continue

            # - key: value  →  parse as key-value pair
            kv_match = re.match(r"^-\s+(.+?):\s+(.+)$", line)
            if kv_match and current_ns and current_key:
                k, v = kv_match.group(1).strip(), kv_match.group(2).strip()
                entry = data[current_ns][current_key]
                if isinstance(entry, dict) and "_lines" in entry:
                    entry[k] = v
                continue

            # - plain text line  →  store as list item
            bullet_match = re.match(r"^-\s+(.+)$", line)
            if bullet_match and current_ns and current_key:
                entry = data[current_ns][current_key]
                if isinstance(entry, dict) and "_lines" in entry:
                    entry["_lines"].append(bullet_match.group(1).strip())
                continue

        # Clean up: convert entries with only _lines to simple list,
        # and merge _lines with key-value pairs
        for ns in data:
            for key in data[ns]:
                entry = data[ns][key]
                if isinstance(entry, dict) and "_lines" in entry:
                    lines = entry.pop("_lines")
                    if not entry:
                        # Only had plain lines — store as list of strings
                        data[ns][key] = lines
                    else:
                        # Had both key-value pairs and plain lines
                        # Keep as dict with a "_notes" field for the lines
                        if lines:
                            entry["_notes"] = lines

        return data

    def _write_markdown(self, data: dict[str, dict[str, Any]]) -> None:
        """Write the data dict back to the markdown file."""
        lines = ["# User Memory Store", ""]

        for ns_key, items in data.items():
            lines.append(f"## {ns_key}")
            lines.append("")
            for item_key, value in items.items():
                lines.append(f"### {item_key}")
                if isinstance(value, list):
                    for v in value:
                        lines.append(f"- {v}")
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if k == "_notes" and isinstance(v, list):
                            for note in v:
                                lines.append(f"- {note}")
                        else:
                            lines.append(f"- {k}: {v}")
                lines.append("")
            lines.append("")

        self.path.write_text("\n".join(lines), encoding="utf-8")
