from loguru import logger

from trainer import Pretrainer, Midtrainer
from parameters import parser

from pathlib import Path

import signal


def pretrain(args):
    if args.warm_start and (Path(str(args.warm_start))/"config.json").exists():
        # by default, the from_pretrained function disables
        # whatever wandb settings was there b/c we usually
        # use this to load an existing model, but when we are
        # actually training, we want to actually enable it
        trainer = Pretrainer.from_pretrained(args.warm_start, disable_wandb=False, distributed=args.distributed)
    else:
        trainer = Pretrainer(args, distributed=args.distributed)

    # hook a signal to checkponit on preemption
    def checkpoint_on_preemption(signum, frame):
        if signum in [signal.SIGUSR1, signal.SIGUSR2, signal.SIGTERM]:
            trainer.save(str(trainer.recovery_dir))
            raise KeyboardInterrupt(
                f"Caught signal {signal.Signals(signum).name}, "
                "byeee!"
            )

    signal.signal(signal.SIGUSR1, checkpoint_on_preemption)
    signal.signal(signal.SIGUSR2, checkpoint_on_preemption)
    signal.signal(signal.SIGTERM, checkpoint_on_preemption)

    # and train
    trainer.train()


def midtrain(args, midtrain):
    if args.warm_start and (Path(str(args.warm_start))/"config.json").exists():
        # by default, the from_pretrained function disables
        # whatever wandb settings was there b/c we usually
        # use this to load an existing model, but when we are
        # actually training, we want to actually enable it
        trainer = Midtrainer.from_checkpoint(args.warm_start, disable_wandb=False, distributed=args.distributed)
    else:
        trainer = Midtrainer.from_pretrained(midtrain, args, distributed=args.distributed, disable_wandb=False)

    # hook a signal to checkponit on preemption
    def checkpoint_on_preemption(signum, frame):
        if signum in [signal.SIGUSR1, signal.SIGUSR2, signal.SIGTERM]:
            trainer.save(str(trainer.recovery_dir))
            raise KeyboardInterrupt(
                f"Caught signal {signal.Signals(signum).name}, "
                "byeee!"
            )

    signal.signal(signal.SIGUSR1, checkpoint_on_preemption)
    signal.signal(signal.SIGUSR2, checkpoint_on_preemption)
    signal.signal(signal.SIGTERM, checkpoint_on_preemption)

    # and train
    trainer.train()

@logger.catch
def execute(args):
    if args.midtrain is not None:
        midtrain(args, args.midtrain)
    else:
        pretrain(args)

def configure(experiment, **kwargs):
    """configure a run from arguments

    Arguments
    ----------
        experiment : str
                experiment name
        kwargs : dict
                arguments to configure the run

    Returns
    -------
        SimpleNamespace
                configuration object
    """

    # listcomp grossery to parse input string into arguments that's
    # readable by argparse

    try:
        return parser.parse_args(([str(experiment)]+
        [j for k,v in kwargs.items() for j in (([f"--{k}", str(v)] if not isinstance(v, list) else [f"--{k}"] + v)
        if not isinstance(v, bool) else [f"--{k}"])]))
    except SystemExit as e:
        logger.error("unrecognized arguments found in configure: {}", kwargs)
        return None

