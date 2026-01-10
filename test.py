from trainer import Pretrainer
import jax

trainer = Pretrainer.from_pretrained("/home/houjun/checkpoints/final_pretrain_1_9b_regular/best")

x,y,_ = trainer.batch()
x,y = jax.device_put((x[:2],y[:2]))

print(trainer.apply_fn({"params": trainer.state.params},
                       x, y, padding_mask=None, deterministic=True)[1])



