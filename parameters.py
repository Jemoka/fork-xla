import argparse

parser = argparse.ArgumentParser(prog='fork')

# logistics
parser.add_argument("experiment", help="name for the experiment", type=str)
parser.add_argument('-v', '--verbose', action='count', default=0, help="log level")
parser.add_argument("--wandb", default=False, action="store_true", help="whether to use wandb")
parser.add_argument("--warm_start", default=None, type=str, help="recover trainer from this path")
parser.add_argument("--local-rank", "--local_rank", default=0, type=int, help="the local rank of this run")
parser.add_argument("--flops_promised", default=989e12, type=float, help="how many flops does our hardware promise (for MFU measurements); default to H100")

# intervals
parser.add_argument("--report_interval", default=32, type=int, help="save to wandb every this many steps")
parser.add_argument("--plot_interval", default=1024, type=int, help="checkpoint every this many steps")
parser.add_argument("--checkpoint_interval", default=10240, type=int, help="checkpoint every this many steps")
parser.add_argument("--validation_interval", default=1024, type=int, help="validate every this many steps")

# dataset
parser.add_argument("--data_file", help="the .toml to the data spec", type=str, default="./experiments/data/pretrain.toml")
parser.add_argument("--out_dir", help="directory to save checkpoints and outputs", type=str, default="output")

# scaling
parser.add_argument("--total_steps", help="effective gradient step count", type=int, default=1_000_000)
parser.add_argument("--batch_size", help="effective batch size", type=int, default=480)
parser.add_argument("--per_device_batch_size", help="how many batches fits on one gpu", type=int, default=8)
parser.add_argument("--validation_steps", help="number of steps to validate, *per device*", type=int, default=128)

# hyperparameters
parser.add_argument("--lr", help="learning rate", type=float, default=2.5e-4)
parser.add_argument("--warmup_pct", help="percentage of steps to warmup", type=float, default=0.005)
parser.add_argument("--decay_pct", help="percentage of steps to decay", type=float, default=0.01)

## optimizer configuration
parser.add_argument("--optimizer", type=str, choices=["muon", "adamw"], default="adamw", help="optimizer")
parser.add_argument("--weight_decay", type=float, default=1e-1, help="AdamW weight decay")
parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1 parameter") 
parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2 parameter")
parser.add_argument("--muon_scale", help="scale lr by this much for muon'd matricies", type=float, default=1.0)
parser.add_argument("--adamw_scalar_scale", help="scale lr by this much for adamw'd scalars", type=float, default=1.0)
parser.add_argument("--adamw_embd_scale", help="scale lr by this much for adamw'd embeddings", type=float, default=20.0)

# GPT model construction arguments
parser.add_argument("--block_size", help="context length", type=int, default=512)
parser.add_argument("--vocab_size", help="vocabulary size", type=int, default=50304)
parser.add_argument("--n_head", help="number of attention heads", type=int, default=16)
parser.add_argument("--n_embd", help="embedding size", type=int, default=2048)
parser.add_argument("--dropout", help="dropout rate", type=float, default=0.0)
parser.add_argument("--no_bias", help="do not use bias in linear layers", dest="bias", action="store_false", default=True)

# Forking token construction arguments
parser.add_argument("--max_block_size", help="context length, after expansion", type=int, default=2048) # thought budget = max_block_size - block_size
parser.add_argument('--plan', nargs='+', help='layer plan; a number of \'fork\'/\'regular\'')
parser.add_argument("--merge_killed_tokens", default=False, action="store_true", help="merge killed tokens to the rightmost token")
parser.add_argument("--averaging_method", type=str, choices=["residual", "logit", "rightmost"], default="residual", help="method for averaging forked tokens")


