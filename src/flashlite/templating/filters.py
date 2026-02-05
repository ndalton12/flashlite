"""Custom Jinja filters for prompt templating."""

import json
from typing import Any

from jinja2 import Environment


def json_encode(value: Any, indent: int | None = None) -> str:
    """
    Encode a value as JSON string.

    Usage in template: {{ data | json }}
    """
    return json.dumps(value, indent=indent, ensure_ascii=False)


def json_encode_pretty(value: Any) -> str:
    """
    Encode a value as pretty-printed JSON.

    Usage in template: {{ data | json_pretty }}
    """
    return json.dumps(value, indent=2, ensure_ascii=False)


def truncate_words(value: str, max_words: int, suffix: str = "...") -> str:
    """
    Truncate string to a maximum number of words.

    Usage in template: {{ text | truncate_words(100) }}
    """
    words = value.split()
    if len(words) <= max_words:
        return value
    return " ".join(words[:max_words]) + suffix


def truncate_chars(value: str, max_chars: int, suffix: str = "...") -> str:
    """
    Truncate string to a maximum number of characters.

    Usage in template: {{ text | truncate_chars(500) }}
    """
    if len(value) <= max_chars:
        return value
    return value[: max_chars - len(suffix)] + suffix


def escape_xml(value: str) -> str:
    """
    Escape XML special characters.

    Usage in template: {{ text | escape_xml }}
    """
    replacements = [
        ("&", "&amp;"),
        ("<", "&lt;"),
        (">", "&gt;"),
        ('"', "&quot;"),
        ("'", "&apos;"),
    ]
    result = value
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def strip_tags(value: str) -> str:
    """
    Remove XML/HTML tags from string.

    Usage in template: {{ html_content | strip_tags }}
    """
    import re

    return re.sub(r"<[^>]+>", "", value)


def bullet_list(items: list[str], bullet: str = "- ") -> str:
    """
    Format a list as bullet points.

    Usage in template: {{ items | bullet_list }}
    """
    return "\n".join(f"{bullet}{item}" for item in items)


def numbered_list(items: list[str], start: int = 1) -> str:
    """
    Format a list as numbered items.

    Usage in template: {{ items | numbered_list }}
    """
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=start))


def wrap_xml(value: str, tag: str) -> str:
    """
    Wrap content in XML tags.

    Usage in template: {{ content | wrap_xml('context') }}
    Produces: <context>content</context>
    """
    return f"<{tag}>{value}</{tag}>"


def indent_text(value: str, spaces: int = 2, first_line: bool = True) -> str:
    """
    Indent text by a number of spaces.

    Usage in template: {{ text | indent_text(4) }}
    """
    prefix = " " * spaces
    lines = value.split("\n")
    if first_line:
        return "\n".join(prefix + line for line in lines)
    else:
        return lines[0] + "\n" + "\n".join(prefix + line for line in lines[1:])


def default_if_empty(value: Any, default: Any) -> Any:
    """
    Return default if value is empty/falsy.

    Usage in template: {{ maybe_empty | default_if_empty('N/A') }}
    """
    return value if value else default


def register_default_filters(env: Environment) -> None:
    """Register all default filters with a Jinja environment."""
    env.filters["json"] = json_encode
    env.filters["json_pretty"] = json_encode_pretty
    env.filters["truncate_words"] = truncate_words
    env.filters["truncate_chars"] = truncate_chars
    env.filters["escape_xml"] = escape_xml
    env.filters["strip_tags"] = strip_tags
    env.filters["bullet_list"] = bullet_list
    env.filters["numbered_list"] = numbered_list
    env.filters["wrap_xml"] = wrap_xml
    env.filters["indent_text"] = indent_text
    env.filters["default_if_empty"] = default_if_empty
