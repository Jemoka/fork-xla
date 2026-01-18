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

from trainer import Pretrainer, Finetuner
from commands import configure
import os
import json
from argparse import Namespace

# !nvidia-smi
# !git fetch && git checkout 2105fdc37aca243b5de80cbf8b498be3956a08d1
#998d1b83fb398bc9b4e762721284d76d050b2de4
# 506114370c82e1eb497051ea759b1e284e723722
# 33a608457c7f95d23fa8aa9f8125067b80486c23
# 7cd57d8f5e0a69a4803a5432d58839c84a0fb7e7

# c124aede21d21cf1d6be0a56fb026ef7f1eed66d
# 1+1

# 506598785e400efa232f53f83a6a4c90350cd15b
args = configure(
    "test",
    flops_promised=275e12,
    report_interval=1,
    shard_into=1,
    data_file="/juice2/scr2/houjun/fork-xla/experiments/data/gsm8k.toml",
    evals=["gsm8k"]
)

# 1+1
# trainer = Finetuner.from_pretrained("/sphinx/u/houjun/checkpoints/fork/jax/pretrain/best", args)


trainer = Finetuner.from_pretrained(
    "/sphinx/u/houjun/checkpoints/fork/jax/pretrain/final_pretrain_1_9b_fork/best",
    args
)

val_metrics = trainer.evaluator(
    "gpt2",
    lambda x: trainer.generate(
        x,
        num_tokens=64,
        temperature=0.0,
        pad_to=256
    ),
    logger=lambda x:logger.info(x),
    batch_size=trainer.per_device_batch_size
)
scores = val_metrics.values()
score = sum(scores) / len(scores)
indx = 8
val_metrics["train/tokens"] = (
    (((indx+1) // trainer.accumulate_steps)*
     trainer.args.batch_size*trainer.args.block_size)
)

from evals.gsm8k import GSM8k
from evals.eval import Evaluator


eval = Evaluator([GSM8k()])
result = eval("gpt2", lambda x: generate(x, num_tokens=64, temperature=0.0, pad_to=128), logger=lambda x:logger.info(x), batch_size=16)

trainer.Trai


def generate(prompts, num_tokens=128, temperature=1.0, top_p=0.9, pad_to=None):
    input, input_mask = pad(prompts, pad_token=0, pad_to=pad_to)
    trainer.key, key = jax.random.split(trainer.key)

    output = trainer._Finetuner__autoregress_jit(
        trainer.state, key, input, input_mask, num_tokens, float(temperature), top_p
    )

    return jax.device_get(output)


import json

with open("/sphinx/u/houjun/checkpoints/fork/jax/midtrain/test_20480_midtrain_fork/best/mmm.json", 'w') as df:
    json.dump(result, df, indent=4)


from tiktoken import get_encoding
enc = get_encoding("gpt2")
res = trainer.generate(enc.encode_batch(["fuck the fucking "]), num_tokens=64, temperature=0.7, top_p=0.8)
print(enc.decode_batch(res)[0])

# enc.encode("<|endoftext|>")


# x,y,mask = trainer.batch()
# decoded = enc.decode_batch(x.tolist())
# encoded = enc.encode_batch([i.split("\n")[0].replace("!","") for i in decoded])
# [len(i) for i in encoded]


# def generate(prompts, num_tokens=128, temperature=1.0, top_p=0.9):
#     __autoregress_jit = jax.jit(
#         _autoregress,
#         in_shardings=(
#             trainer.state_sharding,
#             None,
#             NamedSharding(trainer.mesh, P("batch", None)),
#             NamedSharding(trainer.mesh, P("batch", None))
#         ),
#         out_shardings=None,
#         static_argnames=("num_tokens", "temperature", "top_p"),
#     )

#     input, input_mask = trainer.pad(prompts, pad_token=0)
#     trainer.key, key = jax.random.split(trainer.key)

#     output = __autoregress_jit(
#         trainer.state, key, input, input_mask, num_tokens, float(temperature), top_p
#     )

#     return jax.device_get(output)


# results = trainer.generate(encoded, num_tokens=32)

# results


# 1+1
# 1
# 1+1
# # enc
# # 1+1
# # y
# # x


# # !git fetch && git checkout 104ce8fe5b9f7f2a0b3e210fd95d1da1ce7d862b
# # 40d14e4b0fb3418d8ef2f356e330097aa6b58830
# # !git log
# # 
# self = SillyFinetuner.from_pretrained("/sphinx/u/houjun/checkpoints/fork/jax/pretrain/old/final_pretrain_1_9b_baseline/checkpoint/184320", args)
# # self
# # self.model
# from tiktoken import get_encoding
# enc = get_encoding("gpt2")
# prompts = enc.encode_batch(["In research, NLP---which stands for"])
# # print(enc.decode_batch(res.tolist())[0])

# from flywheel import MemmapDataset, PaddedDataset, Strategy, Sampling
# ds = PaddedDataset(trainer.args, "/sphinx/u/houjun/dataset/gsm8k_aug/")
# ds = Strategy(trainer.args, [Sampling(ds, 1.0)])


# def generate(prompts, num_tokens=128, temperature=1.0, top_p=0.9):
#     __autoregress_jit = jax.jit(
#         _autoregress,
#         in_shardings=(trainer.state_sharding, None, None, None),
#         out_shardings=None,
#         static_argnames=("num_tokens", "temperature", "top_p"),
#     )

#     input, input_mask = pad(prompts, pad_token=0)
#     trainer.key, key = jax.random.split(trainer.key)

#     output = __autoregress_jit(
#         trainer.state, key, input, input_mask, num_tokens, float(temperature), top_p
#     )

#     return jax.device_get(output)

# result = generate(prompts, temperature=1.0, top_p=0.95)
# print(enc.decode(result.tolist()[0]))
# result

# autoregress = jax.jit(
#     _autoregress,
#     in_shardings=(trainer.state_sharding, None, None),
#     out_shardings=None,
#     static_argnames=("num_tokens", "temperature"),
# )

# input, input_mask = trainer.pad(prompts, pad_token=0)
# output = jax.device_get(autoregress(self.state, input, input_mask, num_tokens, temperature))


# # # enc.encode("<|endoftext|>")
# # # eos_token = enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
# # # eos_token

# # enc.encode_batch(["I'm a big ol' chicken, but", "3+12"])
# # enc.decode_batch(x.tolist())[0]




# trainer.
# enc.decode_batch(x[0])
# path = "/sphinx/u/houjun/checkpoints/fork/jax/pretrain/new/61440"
# import jax.numpy as jnp
# mask1 = causal_bias(1024)[:,:,:128,:128]
# mask2 = causal_bias(2048)[:,:,:128,:128]
# (mask1 == mask2).all()



# x,y,_ = trainer.batch()
# l2,r2 = trainer.state.apply_fn(
#     {"params": trainer.state.params},
#     x[:2], y[:2], padding_mask=None, deterministic=True
# )
# l32,r32 = trainer.state.apply_fn(
#     {"params": trainer.state.params},
#     x[:32], y[:32], padding_mask=None, deterministic=True
# )

# trainer.train()
# l2[:,18]
# l32[:,18]

# l2
# l32

# # l2[0, 2]
# l32[8, 2]
# l64[8, 2]



# from jax import random as jax_random
# key = jax_random.PRNGKey(0)

# x = jnp.ones((1, args.block_size, args.n_embd), dtype=jnp.int32)
# cumulative_scores = jnp.ones((1, args.block_size), dtype=jnp.int32)
# token_index = jnp.ones((1, args.block_size), dtype=jnp.int32)

# block = Block(trainer.args)
# variables = block.init(key, x, cumulative_scores, token_index)

# x = jax.random.normal(jax.random.PRNGKey(8), (8, args.block_size, args.n_embd))
# cumulative_scores = jnp.ones((8, args.block_size), dtype=jnp.int32)
# token_index = jnp.arange(args.block_size).repeat(1)[None].repeat(8, axis=0)


# x8 = x[:8]
# cumulative_scores8 = cumulative_scores[:8]
# token_index8 = token_index[:8]

# x2 = x[:2]
# cumulative_scores2 = cumulative_scores[:2]
# token_index2 = token_index[:2]

# import numpy as np
# np.set_printoptions(edgeitems=8, linewidth=128, 
#     formatter=dict(float=lambda x: "%.3g" % x))
# print((block.apply(variables, x8, cumulative_scores8, token_index8)[0][:2]==block.apply(variables, x2, cumulative_scores2, token_index2)[0])[0])
# block.apply(variables, x2, cumulative_scores2, token_index2)[0]
# # (block.apply(variables, x8, cumulative_scores8, token_index8)[:2]==block.apply(variables, x2, cumulative_scores2, token_index2))[0].shape


# token_index = token_index8
# block_size = args.block_size
# token_counts = jnp.zeros((*token_index.shape[:-1], block_size), dtype=token_index.dtype)

# token_counts.at[..., token_index].add(1)
# token_counts

# partial_rotations = jnp.cumsum(
#     jnp.take_along_axis(1 / (token_counts + 1e-10), token_index, axis=-1),
#     axis=-1
# )


# trainer.best_val_score_
# trainer.args
# l8,r8 = trainer.state.apply_fn(
#     {"params": trainer.state.params},
#     x[:8], y[:8], padding_mask=None, deterministic=True
# )
# l32,r32 = trainer.state.apply_fn(
#     {"params": trainer.state.params},
#     x[:32], y[:32], padding_mask=None, deterministic=True
# )
# l64,r64 = trainer.state.apply_fn(
#     {"params": trainer.state.params},
#     x[:64], y[:64], padding_mask=None, deterministic=True
# )
# l120,r120 = trainer.state.apply_fn(
#     {"params": trainer.state.params},
#     x[:120], y[:120], padding_mask=None, deterministic=True
# )

# l120[:,8]


# l32[:32,:3]
# l120[:32,:3]



# l8
# l32
# l64
# l120

# import jax.numpy as jnp

# logits_f32 = l8.astype(jnp.float32)
# logits_flat = logits_f32.reshape(-1, logits_f32.shape[-1])
# targets_flat = y[:8].reshape(-1)

# mask = targets_flat != -1
# logits_masked = logits_flat
# targets_masked = jnp.where(mask, targets_flat, 0)

# import jax
# (jax.nn.log_softmax(logits_masked, axis=-1)*jax.nn.one_hot(targets_masked, trainer.args.vocab_size))

# .shape
# (jax.nn.log_softmax(logits_masked, axis=-1) *
#  )

# mask.sum().clip(min=1)

# l8.argmax(axis=-1).shape
# l8.shape
# enc.decode_batch(l8.argmax(axis=-1).tolist())

# targets_flat

# # Mask out ignore index (-1) AND padding tokens
# mask = targets_flat != -1
# logits_masked = logits_flat
# targets_masked = jnp.where(mask, targets_flat, 0)

# # Compute cross entropy in float32
# loss = -jnp.sum(
#     (jax.nn.log_softmax(logits_masked, axis=-1) *
#         jax.nn.one_hot(targets_masked, self.config.vocab_size))*mask[:,None]
# ) / mask.sum().clip(min=1)

# vs = trainer.make_valid_step()
# vs(trainer.state)

# self = trainer


# model


# from orbax.checkpoint.checkpoint_utils import construct_restore_args

#  jax
#  trainer


# path = "/sphinx/u/houjun/checkpoints/fork/jax/pretrain/best"

# restore_args = 

# # Force a direct PyTree restore
# checkpointer = ocp.StandardCheckpointer()
# restored_state = checkpointer.restore(
#     os.path.join(path, "checkpoint"),
#     trainer.state
# )
# model = vs(jax.device_get(trainer.state))
# model



# print(jax.device_get(trainer.state).opt_state[1][0].mu["blocks_0"]["attn"]["c_proj"]["kernel"].value.mean())
# print()
# ax = 
# ax.params

# print(jax.tree_util.tree_reduce(lambda carry, xs:carry+xs, jax.tree_util.tree_map(lambda x:x.mean(), ax.params)))


# print(jax.device_get(trainer.state).opt_state[1][0].mu["blocks_0"]["attn"]["c_proj"]["kernel"].value.shape)

# print(jax.device_get(restored_state).opt_state[1][0].mu["blocks_0"]["attn"]["c_proj"]["kernel"].value.mean())
# print(jax.device_get(restored_state).opt_state[1][0].mu["blocks_0"]["attn"]["c_proj"]["kernel"].value)
# print(jax.device_get(restored_state).opt_state[1][0].mu["blocks_0"]["attn"]["c_proj"]["kernel"].value.shape)

# trainer.state.opt_state[1][0].mu["blocks_0"]["attn"]["c_proj"]["kernel"].value.sharding

# def to_abstract_local(x):
#     # Convert any leaf array into an abstract leaf with LOCAL sharding.
#     if isinstance(x, jax.Array):
#         return jax.ShapeDtypeStruct(x.shape, x.dtype)
#     return x

# abstract_state = jax.tree_util.tree_map(to_abstract_local, trainer.state)

# import jax
# x,y,_ = trainer.batch()
# x,y = jax.device_put((x,y))
# x.shape
# trainer.state.apply_fn({"params": trainer.state.params},
#                        x[:120, :32], y[:120, :32], padding_mask=None, deterministic=True)[1]


# with ocp.CheckpointManager(...) as mngr:
#   mngr.restore(
#       mngr.latest_step(), 
#       args=ocp.args.Composite(
#           state=ocp.args.StandardRestore(abstract_state),
#       )
#   )


# # restored_state
# # trainer.state = restored
# # x,y,_ = trainer.batch()
# # x,y = jax.device_put((x[:2],y[:2]))
# # restored

# # print(ocp.__version__)
# # x
# # y



# # type(restored)
# # restored.params["blocks_0"]["attn"]["c_attn"]["kernel"]


# # l
# # # print(enc.decode_batch(x.tolist())[0])

# # l

# # def _autoregress(state, input, input_mask, num_tokens, temperature):
# #     seq = jnp.arange(num_tokens)

# #     inp_buf = jnp.zeros((len(input), input.shape[1] + num_tokens))
# #     mask_buf = jnp.zeros((len(input), input_mask.shape[1] + num_tokens))

# #     inp_buf = inp_buf.at[:, :input.shape[1]].set(input)
# #     mask_buf = mask_buf.at[:, :input_mask.shape[1]].set(input_mask)
# #     inp_buf, mask_buf = inp_buf.astype(jnp.int32), mask_buf.astype(jnp.bool_)

# #     def reduce(carry, xb):
# #         inputs, masks = carry
# #         offset = xb + input.shape[1]

# #         outputs, loss_i = state.apply_fn(
# #             {'params': state.params},
# #             inputs,
# #             padding_mask=masks,
# #             deterministic=True
# #         )

# #         next_token = outputs[:, offset-1, :] / temperature
# #         next_token = jax.nn.softmax(next_token, axis=-1)

# #         next_mask = jnp.ones_like(next_token, dtype=jnp.bool_)
# #         new_inputs = inputs.at[:, offset].set(next_token)
# #         new_masks = masks.at[:, offset].set(next_mask)

# #         return (new_inputs, new_masks), None

# #     (final_inputs, final_masks), _ = jax.lax.scan(reduce, (inp_buf, mask_buf), seq)
# #     return final_inputs





# # !git fetch && git checkout e062d8d00a549f781cbcd175d998e489bccbe0b7

# # self.pad(enc.encode_batch(["I'm a big ol' chicken, but", "3+12"])[0]).shape
# # res.shape
# # prompts = enc.encode_batch(["I'm a big ol' chicken, but", "3+12"])

# # prompts=

# # import jax.

# # x, mask = pad(prompts)

# # # from tiktoken import get_encoding

# # args = configure(
# #     "test",
# #     flops_promised=275e12,
# #     report_interval=1
# # )
# # #     validation_interval=10,
# # #     data_file="/juice2/scr2/houjun/fork-xla/experiments/data/pretrain.toml",
# # #     total_steps=8500,
# # #     per_device_batch_size=32, #for h200s
# # #     shard_into=1, # h100
# # #     plan=["regular", "regular", "fork", "regular", "regular", "regular", "fork", "regular", "regular"],
# # # #     # per_device_batch_size=20, for a100s
# # # #     per_device_batch_size=32, # for tpu-v4
# # # #     # batch_size=64, # so we don't insanely accumulate
# # # #     # shard_into=4,
# # # #     validation_steps=1024
# # # )

# # # # !git fetch && git checkout a198ce933867b686644b56103ecdfc59daca1c43
# # # # f812346b1fb05c7704e64b38bc8ce4893dfe45c6
# # # # 3263cec199374c73bed21ff7dd670a066ae1477b

# # # # 1f73ce74cd3c29958c85fcdfd758e0fa577af029

# # # # feec32b5d35c67261af7da168b25c8a051d969ed
# # # # b540e313c012a5b9251f903424abc0584ef2c26a
# # # # 31d18d0529e87e1547addd561ea388283047308a
# # # # d6a7b71b34f5b468fb147676e63e60688d4be140
# # # # 1ac1b0df8b1224a74fb942892f2448d1c3a16ea3
# # # # 60191eedf68f4fa60fecb6195a7eda8b6058554b
# # # #0d24782fed313b9ff9ac7b11f566263495b1504e

# # #  # a060bb2ff75ed8069d1bc1f0e0b89952c235e090
# # # # ecb2fb01ee4cd15aa2f081985e6ff83310089f37

# # # # da77456784a9dcbfdf1ce40676317e6eb3c37a06
# # # # path = "/sphinx/u/houjun/checkpoints/fork/jax/pretrain/final_pretrain_1_9b_baseline/checkpoint/184320"

# # # trainer = Pretrainer(args)
# # # trainer.train()

# # # x.shape
# # # # padding_mask = p[:8,-128:]
# # # # padding_mask

# # # from types import SimpleNamespace
# # # self = SimpleNamespace()
# # # x,y,p = trainer.batch()
# # # padding_mask = p[:8, :256]

# # # qkv = jax.random.normal(jax.random.PRNGKey(8), (8,512,512*3))
# # # cumulative_scores = jnp.zeros((8,512))
# # # token_index = jnp.arange(256).repeat(2)[None].repeat(8, axis=0)
# # # self.n_head = 16
# # # self.config = args

# # # token_index.shape

# # # # # x.shape
# # # # # y.shape
# # # # # trainer.state.params
# # # # # !git log
# # # # def chicken(params):
# # # #     return trainer.state.apply_fn({"params": params}, x[:2,:],y[:2,:], p[:2,:])[1]

# # # # self.state.apply_fn({"params": self.state.params}, x[:2,:],y[:2,:], p[:2,:])
# # # # print(self.state.apply_fn({"params": self.state.params}, x[:1,:],y[:1,:], p[:1,:])[0].sum())
# # # # logits = self.state.apply_fn({"params": self.state.params}, x[:1,:],y[:1,:], p[:1,:])[0]
# # # # targets = y[:1,:]
# # # # # self.state.apply_fn({"params": self.state.params}, x[0][p[0]][None, :],y[0][y[0] != -1][None, :])[1]

# # # # x[0][p[0]].shape
# # # # y[0][y[0] != -1][1:].shape
# # # # y[0]



# # # # x[:1,:]

# # # # import jax
# # # # jax.value_and_grad(chicken)(trainer.state.params)

# # # # params = trainer.state.params

# # # # from model import key_padding_bias, causal_bias
# # # # key_padding_bias(p[:2,:]).shape

# # # # mask = causal_bias(trainer.args.max_block_size)
# # # # mask
# # # # mask = mask[:,:,:512,:512]


# # # # p[:2,:]
# # # # casual_pad_bias(512, p[:2,:])[0]

# # # # key_padding_bias(p[:2,:]).shape
# # # # mask.shape
# # # # (mask + key_padding_bias(p[:2,:])).shape

# # # # .shape

# # # # trainer.state.apply_fn({"params": params}, x[:2,:],y[:2,:],padding_mask=p[:2,:])




# # # # # !cat /juice2/scr2/houjun/fork-xla/experiments/data/midtrain.toml

# # # # # !git status
# # # # # 1+1

# # # # # self = SimpleNamespace()
# # # # # path = "/sphinx/u/houjun/dataset/smoltalk"
# # # # # with open(os.path.join(path, "config.json"), "r") as df:
# # # # #     data = json.load(df)
# # # # # args = Namespace(**data.get("config", {}))
# # # # # args.shard_into = 1
# # # # # args.wandb = False
# # # # # args.data_file="/juice2/scr2/houjun/fork-xla/experiments/data/pretrain.toml"
# # # # # args.out_dir="/sphinx/u/houjun/checkpoints/fork/jax/midtrain/final_pretrain_1_9b_baseline/checkpoint/184320"
# # # # # self = Trainer(args)
# # # # # self.load("/sphinx/u/houjun/checkpoints/fork/jax/pretrain/final_pretrain_1_9b_baseline/checkpoint/18432")

# # # # #     plan = [
# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",

# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",

# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",

# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",

# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",

# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",

# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",

# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",

# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",
# # # # #         "regular",
# # # # #     ],
# # # # #     out_dir="/sphinx/u/houjun/checkpoints/fork",
# # # # #     # per_device_batch_size=48, for h200s
# # # # #     # per_device_batch_size=20, for a100s
# # # # #     per_device_batch_size=32, # for tpu-v4
# # # # #     # batch_size=64, # so we don't insanely accumulate
# # # # #     shard_into=1, # h100
# # # # #     # shard_into=4,
# # # # #     validation_steps=1024
# # # # # )

# # # # # # from types import SimpleNamespace
# # # # # # args.block_size = 64
# # # # # # args.n_embd = 512
# # # # # # self = SimpleNamespace(config=args)
# # # # # # args.plan=["regular", "regular", "fork", "regular", "regular", "fork", "regular", "regular"]

# # # # # # # args.averaging_method = "rightmost"

# # # # # # model = Thoughtbubbles(args)
# # # # # # variables = model.init(
# # # # # #     jax.random.PRNGKey(8),
# # # # # #     jnp.ones((3, 3)).astype(jnp.int16)
# # # # # # )

# # # # # # # model.apply(variables, jnp.ones((1,3)).astype(jnp.int16), deterministic=True)[0].argmax(axis=-1)

# # # # # # # import jax

# # # # # # @jax.jit
# # # # # # def test(values):
# # # # # #     a,b = model.apply(values, jnp.ones((8,3)).astype(jnp.int16), deterministic=True)
# # # # # #     return a.sum()

# # # # # # jax.value_and_grad(test)(variables)
# # # # # # test()

# # # # # # # logits, loss = 
# # # # # # test(jnp.ones((1,3)).astype(jnp.int16))
# # # # # # test(.astype(jnp.int16))
# # # # # # test(jnp.ones((3,3)).astype(jnp.int16))
# # # # # # # loss


# # # # # # def f_single(x1):  # x1: (T,)
# # # # # #     logits, _ = model.apply(variables, x1[None, :], deterministic=True)
# # # # # #     return logits[0]

# # # # # # def f_batch(xB):   # xB: (B,T)
# # # # # #     logits, _ = model.apply(variables, xB, deterministic=True)
# # # # # #     return logits

# # # # # # xB = jnp.ones((3, 3), jnp.int16)

# # # # # # y_v = jax.vmap(f_single)(xB).astype(jnp.float32)
# # # # # # y_b = f_batch(xB).astype(jnp.float32)

# # # # # # print("maxabs all:", jnp.max(jnp.abs(y_v - y_b)))
# # # # # # print("maxabs row0:", jnp.max(jnp.abs(y_v[0] - y_b[0])))

# # # # # # rel = jnp.max(jnp.abs(y_v - y_b)) / (jnp.max(jnp.abs(y_v)) + 1e-12)
# # # # # # print(rel)



# # # # # # !ls
# # # # # # !pushd ./output && pwd -P && popd
# # # # # # !rm -rf output

# # # # # # # from flywheel import MemmapDataset

# # # # # # # data = MemmapDataset(args, "/home/houjun/data-local/datasets/pes2o/")
# # # # # # # x,y = data.get_batch(1024)

# # # # # # # import tiktoken
# # # # # # # enc = tiktoken.get_encoding("gpt2")

# # # # # # # print(enc.decode_batch(x)[14])

# # # # # # # args.validation_steps
# # # # # # # spec = parse_dataset_spec("/juice2/scr2/houjun/fork/experiments/data/pretrain.toml", args)
# # # # # # # x,y =spec.get_batch(3)
# # # # # # # x
# # # # # # # y

# # # # # # # # !cat /etc/hostname
# # # # # # # args.data_file = "./experiments/data/pretrain.toml"
# # # # # # # print(jax.devices())

# # # # # # trainer = Trainer(args)
# # # # # # import time
# # # # # # a = time.time()
# # # # # # x,y = trainer.batch()
# # # # # # b = time.time()
# # # # # # print(b-a)
# # # # # # trainer.async_dl_cache.get("train")
# # # # # # # # trainer.train()
# # # # # # # strategy = trainer.data_strategy


# # # # # # # res = AsyncStrategy(strategy, {
# # # # # # #     "batch_size": 64,
# # # # # # #     "split": "train",
# # # # # # #     "deterministic_key": None
# # # # # # # })

# # # # # # # time.sleep()

# # # # # # # for _ in range(10):
# # # # # # #     a = time.time()
# # # # # # #     res.get_batch()
# # # # # # #     b = time.time()
# # # # # # #     print(b-a)

# # # # # # # print("----")

# # # # # # # for _ in range(10):
# # # # # # #     a = time.time()
# # # # # # #     strategy.get_batch(64)
# # # # # # #     b = time.time()
# # # # # # #     print(b-a)




# # # # # # # import time
# # # # # # # # from types import SimpleNamespace
# # # # # # # # self = SimpleNamespace()
# # # # # # # # kwargs = {
# # # # # # # #     "batch_size": 18,
# # # # # # # #     "split": "train",
# # # # # # # #     "deterministic_key": 8
# # # # # # # # }


# # # # # # # # Async

# # # # # # # # res = trainer.make_valid_step()(trainer.state)
# # # # # # # # trainer.args.estimate_mfu
# # # # # # # # self = trainer
# # # # # # # # self = trainer

# # # # # # # # valid_step = self.make_valid_step()
# # # # # # # # # valid_step
# # # # # # # # result = valid_step(self.state)
# # # # # # # # print(result)
# # # # # # # # print(result)
# # # # # # # # print(result)


# # # # # # # # import time
# # # # # # # # a = time.time()
# # # # # # # # x,y = trainer.batch()
# # # # # # # # b = time.time()

# # # # # # # # b-a

# # # # # # # # x.shape

# # # # # # # # trainer.train()

# # # # # # # # self = trainer

# # # # # # # # from pathlib import Path
# # # # # # # # Path(trainer.recovery_dir).exists()
# # # # # # # # trainer.recovery_dir
# # # # # # # # !ls '/home/houjun/bubbles/checkpoints/test'
# # # # # # # # trainer.train()

# # # # # # # # x,y= trainer.batch()
# # # # # # # # print(x)
# # # # # # # # type(x)

# # # # # # # # print(jax.devices())

# # # # # # # # print(trainer.model)
# # # # # # # # # !git fetch && git checkout 038532e14a345b14fc2b900da09ca7d3a91be178

# # # # # # # # # 26435936ef15a93b74e43dbfb5f2339fdad2c5a8

# # # # # # # # # f944d935b765bfe81d876391fc6dc3f8e269f136

# # # # # # # # # import os
# # # # # # # # # import numpy as np

# # # # # # # # # data = np.memmap(
# # # # # # # # #     os.path.join("/sphinx/u/houjun/dataset/fineweb", "train.bin"), dtype=np.uint16, mode="r"
# # # # # # # # # ).shape)
# # # # # # # # # data[70_000_000_000]
# # # # # # # # # print(data.shape)
# # # # # # # # # 131 B
# # # # # # # # # data[71072000000]

# # # # # # # # # tmp = np.memmap(
# # # # # # # # #     os.path.join("/sphinx/u/houjun/dataset/pes2o_large", "train.bin"), dtype=np.uint16, mode="r"
# # # # # # # # # )

# # # # # # # # # tmp[]
# # # # # # # # # batch = tmp[500000000:500000000+72].reshape(8,9)

# # # # # # # # # batch_size = 8
# # # # # # # # # cut_batch = batch[(~(batch == 0).all(axis=-1))]
# # # # # # # # # np.concat([cut_batch, cut_batch], axis=0)
# # # # # # # # # if cut_batch.shape[0] < batch_size:
# # # # # # # # #     new_batch = 
    

# # # # # # # # # mask = np.zeros_like(data, dtype=bool)
# # # # # # # # # mask[len(data) - np.argmax(data[::-1] != 0):] = True

# # # # # # # # # mask

# # # # # # # # # (data == 0)



# # # # # # # # # w d

# # # # # # # # # trainer.train()
# # # # # # # # # # x,y = trainer.batch()

# # # # # # # # # # 
# # # # # # # # # # enc.decode_batch(x.tolist())
# # # # # # # # # # 1+1
# # # # # # # # # # x
# # # # # # # # # # y
# # # # # # # # # # !ls 
# # # # # # # # # # !realpath ~/nlp/fork/experiments/data/pretrain.toml
# # # # # # # # # # self = trainer._vomit()

# # # # # # # # # # # !cat /etc/hostname

# # # # # # # # # for i in range(10):
# # # # # # # # #     break
# # # # # # # # # else:
# # # # # # # # #     print("bchicken")



# # # # # # # # # # # idx = torch.randint(128,(9, self.config.block_size)).cuda()
# # # # # # # # # # # !git fetch && git checkout 038532e14a345b14fc2b900da09ca7d3a91be178

# # # # # # # d924d2f8f297626c08837a099eb83a56b42dcee0

# # # # # # # efcca7d57f2a17df00347b3f6d3dc8d1b0e9712e

# # # # # # # # # #  # fd4d5d85b447d7abc56f8e6ae5d1bba767779efa


# # # # # # # # # # # from flywheel import MemmapDataset, Sampling, Strategy
# # # # # # # # # # # sampling = Strategy(args, [
# # # # # # # # # # #     Sampling(
# # # # # # # # # # #         MemmapDataset(args, "/sphinx/u/houjun/dataset/pes2o_new"),


# # # # # # # # # # #         0.5
# # # # # # # # # # #     ),
# # # # # # # # # # #     Sampling(
# # # # # # # # # # #         MemmapDataset(args, "/sphinx/u/houjun/dataset/openwebtext"),
# # # # # # # # # # #         0.5
# # # # # # # # # # #     )
# # # # # # # # # # # ])


# # # # # # # # # # import tiktoken
# # # # # # # # # # enc = tiktoken.get_encoding("gpt2")

# # # # # # # # # # # enc.decode_batch(sampling.get_batch(2, deterministic_key=4)[0].
