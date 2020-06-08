from environment import environment
import os

config = {
    'discount': 0.94,
    'exploration_rate':0.9,
    'decay_factor':0.9999,
    'learning_rate':0.0001,
    'episode':1000000,
    'debug' : 0,
}
os.environ['WANDB_MODE'] = 'dryrun'

e = environment(config=config)

e.start()
e.agent.save()
e.agent.printAllQTable()
print(e.metrics)
 