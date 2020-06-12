from environment import environment
import os

config = {
    'discount': 0.28,
    'exploration_rate':0.86,
    'decay_factor':0.9999,
    'learning_rate':0.001,
    'episode':100000,
    'debug' : 0,
}
os.environ['WANDB_MODE'] = 'dryrun'

e = environment(config=config)

e.start()
e.agent.save()
e.agent.printAllQTable()
print(e.metrics)
 