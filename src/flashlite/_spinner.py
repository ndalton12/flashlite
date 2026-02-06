"""Terminal spinner for user-visible progress during async waits.

Provides a lightweight, non-blocking spinner that renders to stderr
when — and only when — the output is an interactive terminal.  Multiple
concurrent ``Spinner`` instances (e.g. from ``complete_many``) are
gracefully collapsed so only one animation is visible at a time.
"""

import asyncio
import sys
import time

# Braille-dot frames — smooth and compact.
_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_INTERVAL = 0.08  # seconds between frame updates

# ANSI helpers
_CYAN = "\033[36m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_CLEAR_LINE = "\r\033[K"

# Module-level guard — only one spinner renders at a time.
_active: bool = False


class Spinner:
    """Async context manager that shows a terminal spinner on stderr.

    The spinner only appears when stderr is a TTY **and** no other
    ``Spinner`` is already active, making it safe for concurrent use
    inside ``complete_many``.

    Args:
        message: Text displayed next to the spinner.
        delay: Grace period (seconds) before the spinner appears.
            If the wrapped operation finishes within this window the
            spinner is never rendered, avoiding flicker for fast calls.
    """

    def __init__(self, message: str = "Working...", *, delay: float = 0.3) -> None:
        self.message = message
        self.delay = delay
        self._task: asyncio.Task[None] | None = None
        self._owns_active = False
        self._start: float = 0.0

    # -- internal -----------------------------------------------------

    async def _render(self) -> None:
        """Background coroutine that draws frames until cancelled."""
        await asyncio.sleep(self.delay)
        idx = 0
        while True:
            elapsed = time.monotonic() - self._start
            frame = _FRAMES[idx % len(_FRAMES)]
            sys.stderr.write(
                f"{_CLEAR_LINE}{_CYAN}{frame}{_RESET} {self.message} "
                f"{_DIM}({elapsed:.1f}s){_RESET}"
            )
            sys.stderr.flush()
            idx += 1
            await asyncio.sleep(_INTERVAL)

    @staticmethod
    def _clear() -> None:
        sys.stderr.write(_CLEAR_LINE)
        sys.stderr.flush()

    # -- context manager ----------------------------------------------

    async def __aenter__(self) -> "Spinner":
        global _active  # noqa: PLW0603
        if sys.stderr.isatty() and not _active:
            _active = True
            self._owns_active = True
            self._start = time.monotonic()
            self._task = asyncio.create_task(self._render())
        return self

    async def __aexit__(self, *_: object) -> None:
        global _active  # noqa: PLW0603
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._clear()
        if self._owns_active:
            _active = False
