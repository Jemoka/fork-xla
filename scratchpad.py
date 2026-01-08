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

from trainer import Finetuner, Pretrainer
from commands import configure
import os
import json
from argparse import Namespace
# !nvidia-smi
# !git fetch && git checkout 33a608457c7f95d23fa8aa9f8125067b80486c23
# 7cd57d8f5e0a69a4803a5432d58839c84a0fb7e7

# c124aede21d21cf1d6be0a56fb026ef7f1eed66d
# 1+1

# 506598785e400efa232f53f83a6a4c90350cd15b
args = configure(
    "test",
    flops_promised=275e12,
    report_interval=1,
    shard_into=1,
    plan=["regular", "regular", "regular", "regular"]
)

trainer = Pretrainer.from_pretrained("/sphinx/u/houjun/checkpoints/fork/jax/pretrain/best")

# 
self = SillyFinetuner.from_pretrained("/sphinx/u/houjun/checkpoints/fork/jax/pretrain/old/final_pretrain_1_9b_baseline/checkpoint/184320", args)
self
# self.model
from tiktoken import get_encoding
enc = get_encoding("gpt2")
# self = Finetuner(args)

# # enc.encode("<|endoftext|>")
# # eos_token = enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
# # eos_token
# res = self.generate(enc.encode_batch(["The Federal Reserve said last Tuesday that"])) 
# # enc.encode_batch(["I'm a big ol' chicken, but", "3+12"])
# print(enc.decode_batch(res.tolist())[0])
# enc.decode_batch(x.tolist())[0]

# import jax
# x,y,_ = trainer.batch()
# x,y = jax.device_put((x[:2],y[:2]))
# trainer.state.apply_fn({"params": trainer.state.params}, x, y, padding_mask=None)[1]
# l
# # print(enc.decode_batch(x.tolist())[0])

# l


# self.pad(enc.encode_batch(["I'm a big ol' chicken, but", "3+12"])[0]).shape
# res.shape
# prompts = enc.encode_batch(["I'm a big ol' chicken, but", "3+12"])

# prompts=

# import jax.

# x, mask = pad(prompts)

# # from tiktoken import get_encoding

# args = configure(
#     "test",
#     flops_promised=275e12,
#     report_interval=1
# )
# #     validation_interval=10,
# #     data_file="/juice2/scr2/houjun/fork-xla/experiments/data/pretrain.toml",
# #     total_steps=8500,
# #     per_device_batch_size=32, #for h200s
# #     shard_into=1, # h100
# #     plan=["regular", "regular", "fork", "regular", "regular", "regular", "fork", "regular", "regular"],
# # #     # per_device_batch_size=20, for a100s
# # #     per_device_batch_size=32, # for tpu-v4
# # #     # batch_size=64, # so we don't insanely accumulate
# # #     # shard_into=4,
# # #     validation_steps=1024
# # )

# # # !git fetch && git checkout a198ce933867b686644b56103ecdfc59daca1c43
# # # f812346b1fb05c7704e64b38bc8ce4893dfe45c6
# # # 3263cec199374c73bed21ff7dd670a066ae1477b

# # # 1f73ce74cd3c29958c85fcdfd758e0fa577af029

# # # feec32b5d35c67261af7da168b25c8a051d969ed
# # # b540e313c012a5b9251f903424abc0584ef2c26a
# # # 31d18d0529e87e1547addd561ea388283047308a
# # # d6a7b71b34f5b468fb147676e63e60688d4be140
# # # 1ac1b0df8b1224a74fb942892f2448d1c3a16ea3
# # # 60191eedf68f4fa60fecb6195a7eda8b6058554b
# # #0d24782fed313b9ff9ac7b11f566263495b1504e

# #  # a060bb2ff75ed8069d1bc1f0e0b89952c235e090
# # # ecb2fb01ee4cd15aa2f081985e6ff83310089f37

# # # da77456784a9dcbfdf1ce40676317e6eb3c37a06
# # # path = "/sphinx/u/houjun/checkpoints/fork/jax/pretrain/final_pretrain_1_9b_baseline/checkpoint/184320"

# # trainer = Pretrainer(args)
# # trainer.train()

# # x.shape
# # # padding_mask = p[:8,-128:]
# # # padding_mask

# # from types import SimpleNamespace
# # self = SimpleNamespace()
# # x,y,p = trainer.batch()
# # padding_mask = p[:8, :256]

# # qkv = jax.random.normal(jax.random.PRNGKey(8), (8,512,512*3))
# # cumulative_scores = jnp.zeros((8,512))
# # token_index = jnp.arange(256).repeat(2)[None].repeat(8, axis=0)
# # self.n_head = 16
# # self.config = args

# # token_index.shape

# # # # x.shape
# # # # y.shape
# # # # trainer.state.params
# # # # !git log
# # # def chicken(params):
# # #     return trainer.state.apply_fn({"params": params}, x[:2,:],y[:2,:], p[:2,:])[1]

# # # self.state.apply_fn({"params": self.state.params}, x[:2,:],y[:2,:], p[:2,:])
# # # print(self.state.apply_fn({"params": self.state.params}, x[:1,:],y[:1,:], p[:1,:])[0].sum())
# # # logits = self.state.apply_fn({"params": self.state.params}, x[:1,:],y[:1,:], p[:1,:])[0]
# # # targets = y[:1,:]
# # # # self.state.apply_fn({"params": self.state.params}, x[0][p[0]][None, :],y[0][y[0] != -1][None, :])[1]

# # # x[0][p[0]].shape
# # # y[0][y[0] != -1][1:].shape
# # # y[0]



# # # x[:1,:]

# # # import jax
# # # jax.value_and_grad(chicken)(trainer.state.params)

# # # params = trainer.state.params

# # # from model import key_padding_bias, causal_bias
# # # key_padding_bias(p[:2,:]).shape

# # # mask = causal_bias(trainer.args.max_block_size)
# # # mask
# # # mask = mask[:,:,:512,:512]


# # # p[:2,:]
# # # casual_pad_bias(512, p[:2,:])[0]

# # # key_padding_bias(p[:2,:]).shape
# # # mask.shape
# # # (mask + key_padding_bias(p[:2,:])).shape

# # # .shape

# # # trainer.state.apply_fn({"params": params}, x[:2,:],y[:2,:],padding_mask=p[:2,:])




# # # # !cat /juice2/scr2/houjun/fork-xla/experiments/data/midtrain.toml

# # # # !git status
# # # # 1+1

# # # # self = SimpleNamespace()
# # # # path = "/sphinx/u/houjun/dataset/smoltalk"
# # # # with open(os.path.join(path, "config.json"), "r") as df:
# # # #     data = json.load(df)
# # # # args = Namespace(**data.get("config", {}))
# # # # args.shard_into = 1
# # # # args.wandb = False
# # # # args.data_file="/juice2/scr2/houjun/fork-xla/experiments/data/pretrain.toml"
# # # # args.out_dir="/sphinx/u/houjun/checkpoints/fork/jax/midtrain/final_pretrain_1_9b_baseline/checkpoint/184320"
# # # # self = Trainer(args)
# # # # self.load("/sphinx/u/houjun/checkpoints/fork/jax/pretrain/final_pretrain_1_9b_baseline/checkpoint/18432")

# # # #     plan = [
# # # #         "regular",
# # # #         "regular",
# # # #         "regular",
# # # #         "regular",

# # # #         "regular",
# # # #         "regular",
# # # #         "regular",
# # # #         "regular",

# # # #         "regular",
# # # #         "regular",
# # # #         "regular",
# # # #         "regular",

# # # #         "regular",
# # # #         "regular",
# # # #         "regular",
# # # #         "regular",

# # # #         "regular",
# # # #         "regular",
# # # #         "regular",
# # # #         "regular",

# # # #         "regular",
# # # #         "regular",
# # # #         "regular",
# # # #         "regular",

# # # #         "regular",
# # # #         "regular",
# # # #         "regular",
# # # #         "regular",

# # # #         "regular",
# # # #         "regular",
# # # #         "regular",
# # # #         "regular",

# # # #         "regular",
# # # #         "regular",
# # # #         "regular",
# # # #         "regular",
# # # #     ],
# # # #     out_dir="/sphinx/u/houjun/checkpoints/fork",
# # # #     # per_device_batch_size=48, for h200s
# # # #     # per_device_batch_size=20, for a100s
# # # #     per_device_batch_size=32, # for tpu-v4
# # # #     # batch_size=64, # so we don't insanely accumulate
# # # #     shard_into=1, # h100
# # # #     # shard_into=4,
# # # #     validation_steps=1024
# # # # )

# # # # # from types import SimpleNamespace
# # # # # args.block_size = 64
# # # # # args.n_embd = 512
# # # # # self = SimpleNamespace(config=args)
# # # # # args.plan=["regular", "regular", "fork", "regular", "regular", "fork", "regular", "regular"]

# # # # # # args.averaging_method = "rightmost"

# # # # # model = Thoughtbubbles(args)
# # # # # variables = model.init(
# # # # #     jax.random.PRNGKey(8),
# # # # #     jnp.ones((3, 3)).astype(jnp.int16)
# # # # # )

# # # # # # model.apply(variables, jnp.ones((1,3)).astype(jnp.int16), deterministic=True)[0].argmax(axis=-1)

# # # # # # import jax

# # # # # @jax.jit
# # # # # def test(values):
# # # # #     a,b = model.apply(values, jnp.ones((8,3)).astype(jnp.int16), deterministic=True)
# # # # #     return a.sum()

# # # # # jax.value_and_grad(test)(variables)
# # # # # test()

# # # # # # logits, loss = 
# # # # # test(jnp.ones((1,3)).astype(jnp.int16))
# # # # # test(.astype(jnp.int16))
# # # # # test(jnp.ones((3,3)).astype(jnp.int16))
# # # # # # loss


# # # # # def f_single(x1):  # x1: (T,)
# # # # #     logits, _ = model.apply(variables, x1[None, :], deterministic=True)
# # # # #     return logits[0]

# # # # # def f_batch(xB):   # xB: (B,T)
# # # # #     logits, _ = model.apply(variables, xB, deterministic=True)
# # # # #     return logits

# # # # # xB = jnp.ones((3, 3), jnp.int16)

# # # # # y_v = jax.vmap(f_single)(xB).astype(jnp.float32)
# # # # # y_b = f_batch(xB).astype(jnp.float32)

# # # # # print("maxabs all:", jnp.max(jnp.abs(y_v - y_b)))
# # # # # print("maxabs row0:", jnp.max(jnp.abs(y_v[0] - y_b[0])))

# # # # # rel = jnp.max(jnp.abs(y_v - y_b)) / (jnp.max(jnp.abs(y_v)) + 1e-12)
# # # # # print(rel)



# # # # # !ls
# # # # # !pushd ./output && pwd -P && popd
# # # # # !rm -rf output

# # # # # # from flywheel import MemmapDataset

# # # # # # data = MemmapDataset(args, "/home/houjun/data-local/datasets/pes2o/")
# # # # # # x,y = data.get_batch(1024)

# # # # # # import tiktoken
# # # # # # enc = tiktoken.get_encoding("gpt2")

# # # # # # print(enc.decode_batch(x)[14])

# # # # # # args.validation_steps
# # # # # # spec = parse_dataset_spec("/juice2/scr2/houjun/fork/experiments/data/pretrain.toml", args)
# # # # # # x,y =spec.get_batch(3)
# # # # # # x
# # # # # # y

# # # # # # # !cat /etc/hostname
# # # # # # args.data_file = "./experiments/data/pretrain.toml"
# # # # # # print(jax.devices())

# # # # # trainer = Trainer(args)
# # # # # import time
# # # # # a = time.time()
# # # # # x,y = trainer.batch()
# # # # # b = time.time()
# # # # # print(b-a)
# # # # # trainer.async_dl_cache.get("train")
# # # # # # # trainer.train()
# # # # # # strategy = trainer.data_strategy


# # # # # # res = AsyncStrategy(strategy, {
# # # # # #     "batch_size": 64,
# # # # # #     "split": "train",
# # # # # #     "deterministic_key": None
# # # # # # })

# # # # # # time.sleep()

# # # # # # for _ in range(10):
# # # # # #     a = time.time()
# # # # # #     res.get_batch()
# # # # # #     b = time.time()
# # # # # #     print(b-a)

# # # # # # print("----")

# # # # # # for _ in range(10):
# # # # # #     a = time.time()
# # # # # #     strategy.get_batch(64)
# # # # # #     b = time.time()
# # # # # #     print(b-a)




# # # # # # import time
# # # # # # # from types import SimpleNamespace
# # # # # # # self = SimpleNamespace()
# # # # # # # kwargs = {
# # # # # # #     "batch_size": 18,
# # # # # # #     "split": "train",
# # # # # # #     "deterministic_key": 8
# # # # # # # }


# # # # # # # Async

# # # # # # # res = trainer.make_valid_step()(trainer.state)
# # # # # # # trainer.args.estimate_mfu
# # # # # # # self = trainer
# # # # # # # self = trainer

# # # # # # # valid_step = self.make_valid_step()
# # # # # # # # valid_step
# # # # # # # result = valid_step(self.state)
# # # # # # # print(result)
# # # # # # # print(result)
# # # # # # # print(result)


# # # # # # # import time
# # # # # # # a = time.time()
# # # # # # # x,y = trainer.batch()
# # # # # # # b = time.time()

# # # # # # # b-a

# # # # # # # x.shape

# # # # # # # trainer.train()

# # # # # # # self = trainer

# # # # # # # from pathlib import Path
# # # # # # # Path(trainer.recovery_dir).exists()
# # # # # # # trainer.recovery_dir
# # # # # # # !ls '/home/houjun/bubbles/checkpoints/test'
# # # # # # # trainer.train()

# # # # # # # x,y= trainer.batch()
# # # # # # # print(x)
# # # # # # # type(x)

# # # # # # # print(jax.devices())

# # # # # # # print(trainer.model)
# # # # # # # # !git fetch && git checkout 038532e14a345b14fc2b900da09ca7d3a91be178

# # # # # # # # 26435936ef15a93b74e43dbfb5f2339fdad2c5a8

# # # # # # # # f944d935b765bfe81d876391fc6dc3f8e269f136

# # # # # # # # import os
# # # # # # # # import numpy as np

# # # # # # # # data = np.memmap(
# # # # # # # #     os.path.join("/sphinx/u/houjun/dataset/fineweb", "train.bin"), dtype=np.uint16, mode="r"
# # # # # # # # ).shape)
# # # # # # # # data[70_000_000_000]
# # # # # # # # print(data.shape)
# # # # # # # # 131 B
# # # # # # # # data[71072000000]

# # # # # # # # tmp = np.memmap(
# # # # # # # #     os.path.join("/sphinx/u/houjun/dataset/pes2o_large", "train.bin"), dtype=np.uint16, mode="r"
# # # # # # # # )

# # # # # # # # tmp[]
# # # # # # # # batch = tmp[500000000:500000000+72].reshape(8,9)

# # # # # # # # batch_size = 8
# # # # # # # # cut_batch = batch[(~(batch == 0).all(axis=-1))]
# # # # # # # # np.concat([cut_batch, cut_batch], axis=0)
# # # # # # # # if cut_batch.shape[0] < batch_size:
# # # # # # # #     new_batch = 
    

# # # # # # # # mask = np.zeros_like(data, dtype=bool)
# # # # # # # # mask[len(data) - np.argmax(data[::-1] != 0):] = True

# # # # # # # # mask

# # # # # # # # (data == 0)



# # # # # # # # w d

# # # # # # # # trainer.train()
# # # # # # # # # x,y = trainer.batch()

# # # # # # # # # 
# # # # # # # # # enc.decode_batch(x.tolist())
# # # # # # # # # 1+1
# # # # # # # # # x
# # # # # # # # # y
# # # # # # # # # !ls 
# # # # # # # # # !realpath ~/nlp/fork/experiments/data/pretrain.toml
# # # # # # # # # self = trainer._vomit()

# # # # # # # # # # !cat /etc/hostname

# # # # # # # # for i in range(10):
# # # # # # # #     break
# # # # # # # # else:
# # # # # # # #     print("bchicken")



# # # # # # # # # # idx = torch.randint(128,(9, self.config.block_size)).cuda()
# # # # # # # # # # !git fetch && git checkout 038532e14a345b14fc2b900da09ca7d3a91be178

# # # # # # d924d2f8f297626c08837a099eb83a56b42dcee0

# # # # # # efcca7d57f2a17df00347b3f6d3dc8d1b0e9712e

# # # # # # # # #  # fd4d5d85b447d7abc56f8e6ae5d1bba767779efa


# # # # # # # # # # from flywheel import MemmapDataset, Sampling, Strategy
# # # # # # # # # # sampling = Strategy(args, [
# # # # # # # # # #     Sampling(
# # # # # # # # # #         MemmapDataset(args, "/sphinx/u/houjun/dataset/pes2o_new"),


# # # # # # # # # #         0.5
# # # # # # # # # #     ),
# # # # # # # # # #     Sampling(
# # # # # # # # # #         MemmapDataset(args, "/sphinx/u/houjun/dataset/openwebtext"),
# # # # # # # # # #         0.5
# # # # # # # # # #     )
# # # # # # # # # # ])


# # # # # # # # # import tiktoken
# # # # # # # # # enc = tiktoken.get_encoding("gpt2")

# # # # # # # # # # enc.decode_batch(sampling.get_batch(2, deterministic_key=4)[0].
