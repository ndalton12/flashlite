"""Template registry for managing prompt templates."""

import hashlib
from pathlib import Path
from typing import Any

from jinja2 import Template

from ..types import TemplateError


class TemplateRegistry:
    """
    Registry for managing prompt templates.

    Supports:
    - Loading templates from files
    - Registering templates by name
    - Namespaced template names (e.g., 'prompts.system.default')
    - Template versioning via content hashing
    """

    def __init__(self) -> None:
        self._templates: dict[str, Template] = {}
        self._sources: dict[str, str] = {}  # name -> original source
        self._hashes: dict[str, str] = {}  # name -> content hash

    def register(self, name: str, template: Template | str) -> None:
        """
        Register a template by name.

        Args:
            name: Template name (can use dots for namespacing)
            template: Jinja Template object or template string
        """
        if isinstance(template, str):
            source = template
            template = Template(template)
        else:
            source = template.source if hasattr(template, "source") else ""

        self._templates[name] = template
        self._sources[name] = source
        self._hashes[name] = self._compute_hash(source)

    def get(self, name: str) -> Template:
        """
        Get a template by name.

        Args:
            name: Template name

        Returns:
            The Jinja Template

        Raises:
            TemplateError: If template not found
        """
        if name not in self._templates:
            raise TemplateError(f"Template not found: {name}")
        return self._templates[name]

    def render(self, name: str, variables: dict[str, Any] | None = None) -> str:
        """
        Render a template by name.

        Args:
            name: Template name
            variables: Variables to pass to template

        Returns:
            Rendered template string
        """
        template = self.get(name)
        return template.render(**(variables or {}))

    def has(self, name: str) -> bool:
        """Check if a template exists."""
        return name in self._templates

    def list_templates(self, prefix: str | None = None) -> list[str]:
        """
        List registered template names.

        Args:
            prefix: Optional prefix to filter by (e.g., 'prompts.')

        Returns:
            List of template names
        """
        names = list(self._templates.keys())
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return sorted(names)

    def get_hash(self, name: str) -> str:
        """
        Get the content hash of a template.

        Useful for reproducibility tracking.
        """
        if name not in self._hashes:
            raise TemplateError(f"Template not found: {name}")
        return self._hashes[name]

    def get_source(self, name: str) -> str:
        """Get the original source of a template."""
        if name not in self._sources:
            raise TemplateError(f"Template not found: {name}")
        return self._sources[name]

    def _compute_hash(self, source: str) -> str:
        """Compute SHA256 hash of template source."""
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    def load_from_directory(
        self,
        directory: Path | str,
        prefix: str = "",
        extensions: tuple[str, ...] = (".jinja", ".j2", ".txt", ".md"),
    ) -> int:
        """
        Load all templates from a directory.

        Template names are derived from file paths:
        - 'prompts/system/default.jinja' -> 'system.default'
        - With prefix='prompts': 'prompts.system.default'

        Args:
            directory: Directory to load from
            prefix: Prefix to add to template names
            extensions: File extensions to load

        Returns:
            Number of templates loaded
        """
        directory = Path(directory)
        if not directory.exists():
            raise TemplateError(f"Template directory not found: {directory}")

        count = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in extensions:
                # Convert path to template name
                relative = file_path.relative_to(directory)
                # Remove extension and convert path separators to dots
                name_parts = list(relative.parts)
                name_parts[-1] = relative.stem  # Remove extension from last part
                name = ".".join(name_parts)

                if prefix:
                    name = f"{prefix}.{name}"

                # Load and register
                source = file_path.read_text()
                self.register(name, source)
                count += 1

        return count

    def clear(self) -> None:
        """Remove all registered templates."""
        self._templates.clear()
        self._sources.clear()
        self._hashes.clear()
