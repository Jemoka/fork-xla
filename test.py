from trainer import Pretrainer
import jax

jax.distributed.initialize()

trainer = Pretrainer.from_pretrained("/home/houjun/checkpoints/final_pretrain_1_9b_regular/best")

print(jax.device_get(trainer.state).opt_state[1][0].mu["blocks_0"]["attn"]["c_proj"]["kernel"].value.mean())
print(jax.device_get(trainer.state).opt_state[1][0].mu["blocks_0"]["attn"]["c_proj"]["kernel"].value)
