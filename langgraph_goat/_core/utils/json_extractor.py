import json
import re


def _try_repair_json(text: str) -> str:
    """Attempt common repairs for LLM-generated JSON.

    Fixes: trailing commas, single quotes, unescaped newlines,
    missing closing brace, and smart quotes.
    """
    # Replace smart quotes with straight quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Remove trailing commas before } or ] (common LLM mistake)
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Replace single-quoted strings with double-quoted strings
    # Only match single-quoted strings that look like JSON values/keys
    text = re.sub(r"'([^']*)'", r'"\1"', text)

    # Fix missing comma between key-value pairs (common in small LLMs)
    # e.g.  "key1": "val1"  "key2": "val2"  -> add comma
    text = re.sub(r'"\s*\n\s*"', '",\n  "', text)

    # Count braces — add missing closing brace if needed
    open_braces = text.count("{")
    close_braces = text.count("}")
    if open_braces > close_braces:
        text = text.rstrip() + ("}" * (open_braces - close_braces))

    return text


def extractJson(text: str) -> dict | None:
    """Find the first valid JSON object in the LLM's response.

    Tries (in order):
      1. Direct parse of stripped text
      2. JSON inside ```json ... ``` fences
      3. Outermost balanced braces
    Each attempt first tries raw json.loads, then retries after
    running _try_repair_json() to fix common small-LLM mistakes.

    Returns None on total failure — caller decides fallback.
    """
    stripped = text.strip()

    # ── Attempt 1: direct parse ──────────────────────────────────────────
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_try_repair_json(stripped))
        except json.JSONDecodeError:
            pass

    # ── Attempt 2: ```json fenced block ──────────────────────────────────
    # Non-greedy to catch the first fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_try_repair_json(candidate))
        except json.JSONDecodeError:
            pass

    # ── Attempt 3: outermost balanced braces (greedy) ────────────────────
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                    try:
                        return json.loads(_try_repair_json(candidate))
                    except json.JSONDecodeError:
                        # Last resort: regex-extract the two known keys
                        return _extract_keys_regex(candidate)

    # ── Attempt 4: last-resort regex extraction ──────────────────────────
    return _extract_keys_regex(text)



def _extract_keys_regex(text: str) -> dict | None:
    """Last-resort: position-based extraction of known explanation keys.

    Used when json.loads fails even after repair. Instead of using a
    non-greedy regex (which truncates at the first quote+comma inside a
    value), this finds the byte positions of the two known keys and takes
    everything between them as the value.

    Handles: "key": "value", 'key': 'value', key: value, key=value
    """
    # Find positions of the two known keys (with or without quotes)
    user_pattern = re.compile(r'["\']?user_explanation["\']?\s*[:=]\s*', re.IGNORECASE)
    dev_pattern = re.compile(r'["\']?developer_explanation["\']?\s*[:=]\s*', re.IGNORECASE)

    user_match = user_pattern.search(text)
    dev_match = dev_pattern.search(text)

    if not user_match and not dev_match:
        return None

    result: dict[str, str] = {}

    # Helper: extract value from a start position to the next key or end
    def _extract_value(start: int, other_key_pos: int | None) -> str:
        """Extract a value starting at `start`, ending at `other_key_pos` or end of text."""
        if other_key_pos is not None and other_key_pos > start:
            raw = text[start:other_key_pos]
        else:
            raw = text[start:]

        # Strip trailing comma, whitespace, and closing braces
        raw = raw.rstrip()
        # Remove a trailing comma if present
        if raw.endswith(","):
            raw = raw[:-1]
        # Remove trailing closing brace(s)
        raw = raw.rstrip("}").rstrip()
        if raw.endswith(","):
            raw = raw[:-1]

        raw = raw.strip()

        # Remove surrounding quotes (single or double) if the entire value is quoted
        if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
            raw = raw[1:-1]
        elif len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
            raw = raw[1:-1]

        # Unescape common escape sequences
        raw = raw.replace('\\"', '"').replace("\\'", "'").replace("\\n", "\n")

        return raw.strip()

    user_start = user_match.end() if user_match else None
    dev_start = dev_match.end() if dev_match else None

    # Determine value boundaries based on which key comes first
    if user_start is not None and dev_start is not None:
        if user_start < dev_start:
            # user comes first, its value ends at dev key
            result["user_explanation"] = _extract_value(user_start, dev_match.start())
            result["developer_explanation"] = _extract_value(dev_start, None)
        else:
            # dev comes first, its value ends at user key
            result["developer_explanation"] = _extract_value(dev_start, user_match.start())
            result["user_explanation"] = _extract_value(user_start, None)
    elif user_start is not None:
        result["user_explanation"] = _extract_value(user_start, None)
    elif dev_start is not None:
        result["developer_explanation"] = _extract_value(dev_start, None)

    if result:
        return result
    return None




