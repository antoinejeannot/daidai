import functools
import logging
import os
from logging import LoggerAdapter


class StructuredLoggerAdapter(LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.pop("extra", {})
        # Move any non-standard kwargs to the context
        for key in list(kwargs.keys()):
            if key not in ["exc_info", "stack_info", "stacklevel"]:
                extra[key] = kwargs.pop(key)

        # Format in structlog style
        if extra:
            context_str = " ".join(f"{k}={v!r}" for k, v in extra.items())
            msg = f"{msg} {context_str}" if msg else context_str

        return msg, kwargs


@functools.cache
def get_logger(name):
    logger = logging.getLogger(name)

    # Only configure if needed
    if not logger.hasHandlers() and not logger.parent.hasHandlers():
        log_level_name = os.environ.get("DAIDAI_LOG_LEVEL", "WARNING").upper()
        log_level = getattr(logging, log_level_name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)

        logger.setLevel(log_level)
        logger.addHandler(handler)

    return StructuredLoggerAdapter(logger, {})
