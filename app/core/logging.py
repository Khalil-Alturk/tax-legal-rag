"""Logging setup with a filter that redacts the Google API key from every record."""

import logging
import re

# Google API keys: classic "AIza" + 35 chars, or the newer "AQ." prefixed format.
_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{35}|AQ\.[0-9A-Za-z_\-]{20,}")


class RedactSecretsFilter(logging.Filter):
    """Replace anything that looks like an API key (or the configured key) with ***."""

    def __init__(self, secret: str | None = None) -> None:
        super().__init__()
        self._secret = secret

    def _redact(self, text: str) -> str:
        if self._secret:
            text = text.replace(self._secret, "***")
        return _KEY_PATTERN.sub("***", text)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redact(record.getMessage())
        record.args = None
        return True


def setup_logging(level: str = "INFO", secret: str | None = None) -> None:
    """Configure root logging once; safe to call multiple times."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    if not any(getattr(h, "_rag_handler", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler._rag_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    for handler in root.handlers:
        handler.filters = [f for f in handler.filters if not isinstance(f, RedactSecretsFilter)]
        handler.addFilter(RedactSecretsFilter(secret))
