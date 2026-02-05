"""Tests for templating functionality."""

from pathlib import Path

import pytest

from flashlite.templating import TemplateEngine, TemplateRegistry
from flashlite.templating.filters import (
    bullet_list,
    json_encode,
    numbered_list,
    truncate_chars,
    truncate_words,
    wrap_xml,
)
from flashlite.types import TemplateError


class TestTemplateRegistry:
    """Tests for TemplateRegistry."""

    def test_register_and_get(self) -> None:
        registry = TemplateRegistry()
        registry.register("test", "Hello, {{ name }}!")

        template = registry.get("test")
        result = template.render(name="World")
        assert result == "Hello, World!"

    def test_get_nonexistent_raises(self) -> None:
        registry = TemplateRegistry()
        with pytest.raises(TemplateError, match="not found"):
            registry.get("nonexistent")

    def test_render_shortcut(self) -> None:
        registry = TemplateRegistry()
        registry.register("greet", "Hi {{ name }}!")
        result = registry.render("greet", {"name": "Alice"})
        assert result == "Hi Alice!"

    def test_has_template(self) -> None:
        registry = TemplateRegistry()
        registry.register("exists", "content")
        assert registry.has("exists")
        assert not registry.has("missing")

    def test_list_templates(self) -> None:
        registry = TemplateRegistry()
        registry.register("prompts.system", "a")
        registry.register("prompts.user", "b")
        registry.register("other", "c")

        all_templates = registry.list_templates()
        assert set(all_templates) == {"prompts.system", "prompts.user", "other"}

        prompts_only = registry.list_templates(prefix="prompts.")
        assert set(prompts_only) == {"prompts.system", "prompts.user"}

    def test_get_hash(self) -> None:
        registry = TemplateRegistry()
        registry.register("test", "Hello!")

        hash1 = registry.get_hash("test")
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

        # Same content = same hash
        registry.register("test2", "Hello!")
        assert registry.get_hash("test2") == hash1

    def test_load_from_directory(self, temp_template_dir: Path) -> None:
        registry = TemplateRegistry()
        count = registry.load_from_directory(temp_template_dir)

        assert count == 3  # greeting.jinja, system.jinja, nested/deep.jinja
        assert registry.has("greeting")
        assert registry.has("system")
        assert registry.has("nested.deep")

    def test_load_with_prefix(self, temp_template_dir: Path) -> None:
        registry = TemplateRegistry()
        registry.load_from_directory(temp_template_dir, prefix="prompts")

        assert registry.has("prompts.greeting")
        assert registry.has("prompts.nested.deep")


class TestTemplateEngine:
    """Tests for TemplateEngine."""

    def test_render_inline_template(self) -> None:
        engine = TemplateEngine()
        result = engine.render("Hello, {{ name }}!", {"name": "World"})
        assert result == "Hello, World!"

    def test_render_registered_template(self) -> None:
        engine = TemplateEngine()
        engine.register("greet", "Hi {{ name }}!")
        result = engine.render("greet", {"name": "Alice"})
        assert result == "Hi Alice!"

    def test_render_from_directory(self, temp_template_dir: Path) -> None:
        engine = TemplateEngine(template_dir=temp_template_dir)
        result = engine.render("greeting", {"name": "Bob", "place": "Wonderland"})
        assert result == "Hello, Bob! Welcome to Wonderland."

    def test_strict_mode_raises_on_missing_var(self) -> None:
        engine = TemplateEngine(strict=True)
        with pytest.raises(TemplateError, match="Missing"):
            engine.render("Hello, {{ name }}!")

    def test_get_variables(self) -> None:
        engine = TemplateEngine()
        engine.register("test", "Hello {{ name }}, you are {{ age }} years old")

        variables = engine.get_variables("test")
        assert variables == {"name", "age"}

    def test_custom_filter(self) -> None:
        engine = TemplateEngine()
        engine.add_filter("double", lambda x: x * 2)

        result = engine.render("{{ value | double }}", {"value": 5})
        assert result == "10"

    def test_global_variable(self) -> None:
        engine = TemplateEngine()
        engine.add_global("app_name", "MyApp")

        result = engine.render("Welcome to {{ app_name }}!")
        assert result == "Welcome to MyApp!"


class TestFilters:
    """Tests for custom Jinja filters."""

    def test_json_encode(self) -> None:
        result = json_encode({"key": "value"})
        assert result == '{"key": "value"}'

    def test_json_encode_with_indent(self) -> None:
        result = json_encode({"a": 1}, indent=2)
        assert "  " in result  # Has indentation

    def test_truncate_words(self) -> None:
        text = "one two three four five"
        assert truncate_words(text, 3) == "one two three..."
        assert truncate_words(text, 10) == text  # No truncation needed

    def test_truncate_chars(self) -> None:
        text = "Hello World"
        assert truncate_chars(text, 8) == "Hello..."
        assert truncate_chars(text, 50) == text

    def test_bullet_list(self) -> None:
        items = ["apple", "banana", "cherry"]
        result = bullet_list(items)
        assert result == "- apple\n- banana\n- cherry"

    def test_numbered_list(self) -> None:
        items = ["first", "second", "third"]
        result = numbered_list(items)
        assert result == "1. first\n2. second\n3. third"

    def test_wrap_xml(self) -> None:
        result = wrap_xml("content", "tag")
        assert result == "<tag>content</tag>"

    def test_filters_in_template(self) -> None:
        engine = TemplateEngine()

        # Test json filter
        result = engine.render("{{ data | json }}", {"data": [1, 2, 3]})
        assert result == "[1, 2, 3]"

        # Test bullet_list filter
        result = engine.render("{{ items | bullet_list }}", {"items": ["a", "b"]})
        assert result == "- a\n- b"
