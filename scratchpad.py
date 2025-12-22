"""
scratchpad.py
A place to test code snippets and experiment with new ideas.
"""

import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> |"
    "<level>{level: ^8}</level>| "
    "<magenta>({name}:{line})</magenta> <level>{message}</level>",
    level="DEBUG",
    colorize=True,
    enqueue=True,
    filter=lambda x: x["extra"].get("task", "") != "plot",
)

from trainer import Trainer
from commands import configure
# !nvidia-smi

args = configure(
    "test",
    flops_promised=275e12,
    report_interval=16,
    # validation_interval=1,
    data_file="/home/houjun/data/recipes/pretrain.toml",
    block_size=512,
    estimate_mfu=False,
    max_block_size=1024,
    plan = [
        "regular",
        "regular",
        "regular",
        "regular",

        "regular",
        "regular",
        "regular",
        "regular",

        "regular",
        "regular",
        "regular",
        "regular",

        "regular",
        "regular",
        "regular",
        "regular",

        "regular",
        "regular",
        "regular",
        "regular",

        "regular",
        "regular",
        "regular",
        "regular",

        "regular",
        "regular",
        "regular",
        "regular",

        "regular",
        "regular",
        "regular",
        "regular",

        "regular",
        "regular",
        "regular",
        "regular",
    ],
    out_dir="/home/houjun/checkpoints",
    # per_device_batch_size=48, for h200s
    # per_device_batch_size=20, for a100s
    per_device_batch_size=4, # for tpu-v4
    # batch_size=64, # so we don't insanely accumulate
    shard_into=4,
    validation_steps=1024
)

# from flywheel import MemmapDataset

# data = MemmapDataset(args, "/home/houjun/data-local/datasets/pes2o/")
# x,y = data.get_batch(1024)

# import tiktoken
# enc = tiktoken.get_encoding("gpt2")

# print(enc.decode_batch(x)[14])

# args.validation_steps
# spec = parse_dataset_spec("/juice2/scr2/houjun/fork/experiments/data/pretrain.toml", args)
# x,y =spec.get_batch(3)
# x
# y

# # !cat /etc/hostname
# args.data_file = "./experiments/data/pretrain.toml"
# print(jax.devices())

trainer = Trainer(args)
trainer.train()
# self = trainer
# self = trainer

# valid_step = self.make_valid_step()
# # valid_step
# result = valid_step(self.state)
# print(result)
# print(result)
# print(result)


# import time
# a = time.time()
# x,y = trainer.batch()
# b = time.time()

# b-a

# x.shape

# trainer.train()

# self = trainer

# from pathlib import Path
# Path(trainer.recovery_dir).exists()
# trainer.recovery_dir
# !ls '/home/houjun/bubbles/checkpoints/test'
# trainer.train()

# x,y= trainer.batch()
# print(x)
# type(x)

# print(jax.devices())

# print(trainer.model)
# # !git fetch && git checkout 829d37fb44eff4d597b3fea92ac42fc1d9290c86

# # 26435936ef15a93b74e43dbfb5f2339fdad2c5a8

# # f944d935b765bfe81d876391fc6dc3f8e269f136

# # import os
# # import numpy as np

# # data = np.memmap(
# #     os.path.join("/sphinx/u/houjun/dataset/fineweb", "train.bin"), dtype=np.uint16, mode="r"
# # ).shape)
# # data[70_000_000_000]
# # print(data.shape)
# # 131 B
# # data[71072000000]

# # tmp = np.memmap(
# #     os.path.join("/sphinx/u/houjun/dataset/pes2o_large", "train.bin"), dtype=np.uint16, mode="r"
# # )

# # tmp[]
# # batch = tmp[500000000:500000000+72].reshape(8,9)

# # batch_size = 8
# # cut_batch = batch[(~(batch == 0).all(axis=-1))]
# # np.concat([cut_batch, cut_batch], axis=0)
# # if cut_batch.shape[0] < batch_size:
# #     new_batch = 
    

# # mask = np.zeros_like(data, dtype=bool)
# # mask[len(data) - np.argmax(data[::-1] != 0):] = True

# # mask

# # (data == 0)



# # w d

# # # trainer.train()
# # # x,y = trainer.batch()

# # # 
# # # enc.decode_batch(x.tolist())
# # # 1+1
# # # x
# # # y
# # # !ls 
# # # !realpath ~/nlp/fork/experiments/data/pretrain.toml
# # # self = trainer._vomit()

# # # # !cat /etc/hostname

# # for i in range(10):
# #     break
# # else:
# #     print("bchicken")



# # # # idx = torch.randint(128,(9, self.config.block_size)).cuda()
# # # # !git fetch && git checkout efcca7d57f2a17df00347b3f6d3dc8d1b0e9712e

# #  # fd4d5d85b447d7abc56f8e6ae5d1bba767779efa


# # # from flywheel import MemmapDataset, Sampling, Strategy
# # # sampling = Strategy(args, [
# # #     Sampling(
# # #         MemmapDataset(args, "/sphinx/u/houjun/dataset/pes2o_new"),


# # #         0.5
# # #     ),
# # #     Sampling(
# # #         MemmapDataset(args, "/sphinx/u/houjun/dataset/openwebtext"),
# # #         0.5
# # #     )
# # # ])


# # import tiktoken
# # enc = tiktoken.get_encoding("gpt2")

# # # enc.decode_batch(sampling.get_batch(2, deterministic_key=4)[0].tolist())
    



