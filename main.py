from environment import environment

config = {
    'discount': 0.95,
    'exploration_rate':0.9,
    'decay_factor':0.9999,
    'learning_rate':0.01,
    'episode':100000,
    'debug' : 0,
}

e = environment(config=config)

e.start()
e.agent.save()
e.agent.printQTable()