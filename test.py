from trainer import Pretrainer
import jax

jax.distributed.initialize()

trainer = Pretrainer.from_pretrained("/home/houjun/checkpoints/final_pretrain_1_9b_regular/checkpoint/61440")

print(jax.tree_util.tree_reduce(lambda carry, xs:carry+xs, jax.tree_util.tree_map(lambda x:x.mean(), trainer.state.params)))
