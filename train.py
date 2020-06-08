from environment import environment
import wandb

def train():
    e = environment()
    e.start()
    e.agent.save()
    e.agent.printAllQTable()  
    print(e.metrics)  


wandb.agent('hakanonal/blackjack/5frlvae6',function=train)
