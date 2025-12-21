import os
import sys
import argparse

import jax
import jax.numpy as jnp
import random
import numpy as np
import inspect
import logging
from loguru import logger
from dotenv import load_dotenv

import parameters

load_dotenv()

logger.remove()

# Set random seeds for reproducibility
# JAX uses explicit PRNGKey which is handled in the trainer
random.seed(0)
np.random.seed(0)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

if __name__ == "__main__":
    args = parameters.parser.parse_args()
    if args.distributed:

        import jax
        jax.distributed.initialize()

    logger.add(
        sys.stderr,
        format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> |"
        "<level>{level: ^8}</level>| "
        "<magenta>({name}:{line})</magenta> <level>{message}</level>",
        level=("DEBUG" if args.verbose > 0 else "INFO"),
        colorize=True,
        enqueue=True,
        filter=lambda x: x["extra"].get("task", "") != "plot",
    )

    # Log JAX device information
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"JAX process count: {jax.process_count()}")
    logger.info(f"JAX process index: {jax.process_index()}")

    from commands import execute
    execute(args)
