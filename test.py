from trainer import Pretrainer
import jax

jax.distributed.initialize()

trainer = Pretrainer.from_pretrained("/home/houjun/checkpoints/final_pretrain_1_9b_regular/best")
valid_step = trainer.make_valid_step()
print(valid_step(trainer.state)[1])
#  cc
# print(trainer.state.apply_fn({"params": trainer.state.params},
#                              x, y, padding_mask=None, deterministic=True)[1])



