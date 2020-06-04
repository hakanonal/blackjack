from environment import environment
import wandb

def train():
    e = environment()
    e.start()
    e.agent.save()
    e.agent.printQTable()    


wandb.agent('hakanonal/blackjack/q4rd4pu3',function=train)    