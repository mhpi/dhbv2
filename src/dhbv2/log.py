"""Config for debug logging in ngen runtime.

Inspired by NOAA-OWP/lstm logging @Austin Raney.
"""

import logging
import logging.config

log = logging.getLogger("bmi.dhbv2")


def configure_logging(level: str = 'info') -> None:
    """Configure logging for dhbv2 module."""
    if level == 'info':
        level = logging.INFO
    elif level == 'debug':
        level = logging.DEBUG
    else:
        level = logging.ERROR
    logging.config.dictConfig(logging_config)
    log.setLevel(level)


logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "incremental": False,
    "formatters": {
        "short": {
            "format": "[%(levelname)s|%(name)s]: %(message)s",
            "datefmt": "%Y-%m-%d T%H:%M:%S",
        },
        "verbose": {
            "format": "[%(levelname)s|%(name)s|%(asctime)s|%(module)s.%(funcName)s:%(lineno)d]: %(message)s",
            "datefmt": "%Y-%m-%d T%H:%M:%S",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "short",
        },
        "stderr": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "verbose",
            "level": "WARNING",
        },
    },
    "loggers": {
        "root": {
            "handlers": ["stdout", "stderr"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
