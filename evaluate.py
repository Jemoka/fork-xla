#!/usr/bin/env python3
"""
Evaluation script for running all benchmarks on a trained model checkpoint.

Usage:
    python evaluate.py --checkpoint /path/to/checkpoint
    python evaluate.py --checkpoint /path/to/checkpoint --evals hellaswag piqa
"""

import click
import json
import jax

from loguru import logger
from commands import configure

from trainer.finetuner import Finetuner, AVAILABLE_EVALS
from evals import (
    Evaluator,
    Blimp,
    GSM8k,
    HellaSwag,
    PIQA,
    ARCEasy,
    ARCChallenge,
    Lambada,
)

@click.command()
@click.option(
    "--checkpoint",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to model checkpoint directory",
)
@click.option(
    "--evals",
    "-e",
    multiple=True,
    default=None,
    help="Specific evaluations to run (default: all). Can specify multiple times.",
)
@click.option(
    "--encoding",
    default="gpt2",
    help="Tokenizer encoding to use (default: gpt2)",
)
@click.option(
    "--truncate/--no-truncate",
    default=True,
    help="Whether to truncate datasets for batch alignment (default: True)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Optional JSON file to write results to",
)
@click.option(
    "--shard-into",
    type=int,
    default=1,
    help="split model into this many splits",
)
@click.option(
    "--per-device-batch-size",
    type=int,
    default=4,
    help="how many sequences fit per GPU",
)
def main(checkpoint, evals, encoding, truncate, output, shard_into, per_device_batch_size):
    """Run evaluations on a trained model checkpoint."""

    # Determine which evals to run
    if evals:
        eval_names = list(evals)
    else:
        eval_names = list(AVAILABLE_EVALS.keys())

    # Validate eval names
    for name in eval_names:
        if name.lower() not in AVAILABLE_EVALS:
            raise click.BadParameter(
                f"Unknown evaluation: {name}. Available: {list(AVAILABLE_EVALS.keys())}"
            )

    logger.info(f"EVAL | Loading checkpoint from {checkpoint}")
    logger.info(f"EVAL | Device count: {jax.device_count()}")
    logger.info(f"EVAL | Process count: {jax.process_count()}")
    logger.info(f"EVAL | Process index: {jax.process_index()}")

    # build arguments
    args = configure(
        "evaluator",
        shard_into=shard_into,
        per_device_batch_size=per_device_batch_size,
        evals=eval_names
    )

    # Load the trainer from checkpoint
    trainer = Finetuner.from_checkpoint(
        checkpoint,
        disable_wandb=True,
        distributed=jax.process_count() > 1,
    )

    logger.info(f"EVAL | Model loaded successfully")
    logger.info(f"EVAL | Running evaluations: {eval_names}")

    # Create evaluator and run
    evaluator = trainer.evaluator
    results = evaluator(encoding, trainer, truncate=truncate)

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("=" * 60)

    # Save results to JSON if requested
    if output and jax.process_index() == 0:
        with open(output, "w") as f:
            json.dump(
                {
                    "checkpoint": checkpoint,
                    "encoding": encoding,
                    "truncate": truncate,
                    "results": {k: float(v) for k, v in results.items()},
                },
                f,
                indent=2,
            )
        logger.info(f"EVAL | Results saved to {output}")

    return results


if __name__ == "__main__":
    main()
