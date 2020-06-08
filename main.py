from environment import environment
import os

config = {
    'discount': 0.94,
    'exploration_rate':0.5,
    'decay_factor':0.99,
    'learning_rate':0.0001,
    'episode':10000,
    'debug' : 0,
}
os.environ['WANDB_MODE'] = 'dryrun'

e = environment(config=config)

e.start()
e.agent.save()
e.agent.printQTable()
print(e.metrics)
 