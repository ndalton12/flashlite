"""Jinja template engine for prompt templating."""

from pathlib import Path
from typing import Any

from jinja2 import (
    BaseLoader,
    Environment,
    FileSystemLoader,
    StrictUndefined,
    Template,
    TemplateNotFound,
    Undefined,
    UndefinedError,
)

from ..types import TemplateError
from .filters import register_default_filters
from .registry import TemplateRegistry


class RegistryLoader(BaseLoader):
    """Jinja loader that loads templates from a TemplateRegistry."""

    def __init__(self, registry: TemplateRegistry):
        self.registry = registry

    def get_source(
        self, environment: Environment, template: str
    ) -> tuple[str, str | None, Any]:
        if not self.registry.has(template):
            raise TemplateNotFound(template)
        source = self.registry.get_source(template)
        return source, template, lambda: True


class TemplateEngine:
    """
    Main template engine for rendering prompts.

    Features:
    - File-based templates from a directory
    - In-memory template registry
    - Custom filters for common operations
    - Strict undefined checking (errors on missing variables)
    - Variable validation
    """

    def __init__(
        self,
        template_dir: Path | str | None = None,
        strict: bool = True,
    ):
        """
        Initialize the template engine.

        Args:
            template_dir: Optional directory to load templates from
            strict: If True, raise error on undefined variables
        """
        self.registry = TemplateRegistry()
        self._template_dir = Path(template_dir) if template_dir else None

        # Create loaders
        loaders: list[BaseLoader] = [RegistryLoader(self.registry)]
        if self._template_dir and self._template_dir.exists():
            loaders.append(FileSystemLoader(str(self._template_dir)))

        # Create Jinja environment
        # Using ChoiceLoader to try registry first, then filesystem
        from jinja2 import ChoiceLoader

        self.env = Environment(
            loader=ChoiceLoader(loaders),
            undefined=StrictUndefined if strict else Undefined,
            autoescape=False,  # Don't HTML-escape (we're making prompts, not HTML)
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register default filters
        register_default_filters(self.env)

        # Load templates from directory if provided
        if self._template_dir and self._template_dir.exists():
            self.registry.load_from_directory(self._template_dir)

    def render(
        self,
        template: str,
        variables: dict[str, Any] | None = None,
        validate: bool = True,
    ) -> str:
        """
        Render a template with variables.

        Args:
            template: Template name or inline template string
            variables: Variables to pass to the template
            validate: If True, validate that all variables are provided

        Returns:
            Rendered template string

        Raises:
            TemplateError: If template not found or rendering fails
        """
        variables = variables or {}

        try:
            # Try to get from registry/filesystem first
            try:
                tpl = self.env.get_template(template)
            except TemplateNotFound:
                # Treat as inline template string
                tpl = self.env.from_string(template)

            if validate:
                self._validate_variables(tpl, variables)

            return tpl.render(**variables)

        except UndefinedError as e:
            raise TemplateError(f"Missing template variable: {e}") from e
        except Exception as e:
            raise TemplateError(f"Template rendering failed: {e}") from e

    def render_string(
        self,
        template_string: str,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """
        Render an inline template string.

        Args:
            template_string: The template source string
            variables: Variables to pass to the template

        Returns:
            Rendered string
        """
        variables = variables or {}
        try:
            tpl = self.env.from_string(template_string)
            return tpl.render(**variables)
        except UndefinedError as e:
            raise TemplateError(f"Missing template variable: {e}") from e
        except Exception as e:
            raise TemplateError(f"Template rendering failed: {e}") from e

    def register(self, name: str, template: str) -> None:
        """
        Register an in-memory template.

        Args:
            name: Template name
            template: Template source string
        """
        self.registry.register(name, template)

    def get_variables(self, template: str) -> set[str]:
        """
        Get the variables used in a template.

        Note: This uses AST analysis and may not catch all dynamic variables.

        Args:
            template: Template name or inline template string

        Returns:
            Set of variable names
        """
        from jinja2 import meta

        try:
            try:
                self.env.get_template(template)  # Validate template exists
                source = self.registry.get_source(template)
            except TemplateNotFound:
                source = template

            ast = self.env.parse(source)
            return meta.find_undeclared_variables(ast)
        except Exception:
            return set()

    def _validate_variables(self, template: Template, variables: dict[str, Any]) -> None:
        """Validate that all required variables are provided."""
        from jinja2 import meta

        if not hasattr(template, "source"):
            return

        try:
            ast = self.env.parse(template.source)
            required = meta.find_undeclared_variables(ast)
            provided = set(variables.keys())
            missing = required - provided

            if missing:
                raise TemplateError(
                    f"Missing required template variables: {', '.join(sorted(missing))}"
                )
        except TemplateError:
            raise
        except Exception:
            # If we can't parse, let rendering catch the error
            pass

    def add_filter(self, name: str, func: Any) -> None:
        """Add a custom filter to the environment."""
        self.env.filters[name] = func

    def add_global(self, name: str, value: Any) -> None:
        """Add a global variable available in all templates."""
        self.env.globals[name] = value
