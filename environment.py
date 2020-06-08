import random
from agent import agent
import wandb

class environment:

    def __init__(self, config=None):

        if(config is None):
            wandb.init(project="blackjack")
            self.config = wandb.config
        else:
            wandb.init(project="blackjack",config=config)
            self.config = config
        self.agent = agent(
            discount=self.config['discount'],
            exploration_rate=self.config['exploration_rate'],
            decay_factor=self.config['decay_factor'],
            learning_rate=self.config['learning_rate']
            )
        self.initGame()
        self.metrics = {
            'tot_win' : 0,
            'tot_draw' : 0,
            'tot_lose': 0,
            'exploration_rate' : self.agent.exploration_rate,
        }

    def initGame(self):
        self.state = {'player_sum':0,'dealer_sum':0, 'usable_ace': False} 
        self.state,_ = self.play(1)
        self.state,_ = self.play(1)
        self.dealer_usable_ace = False
        self.state,_ = self.playDealer()
        self.actions_played = []
        

    def start(self):
        for episode in range(1,self.config['episode']+1):
            self.initGame()
            #player turn
            while True:
                action_to_play = self.agent.get_next_action(self.state)
                new_state, ended = self.play(action_to_play)
                self.actions_played.append((self.state,new_state,action_to_play))
                self.debug1(episode,self.state,new_state,action_to_play)
                self.state = new_state
                if ended:
                    break

            #dealer turn
            while True:
                new_state, ended = self.playDealer()
                self.debug1(episode,self.state,new_state,-1)
                self.state = new_state
                if ended:
                    break

            #q-tatble update backpropogation
            reward = self.findWinner()
            if reward == 1:
                self.metrics['tot_win'] += 1
                self.metrics['avg_win'] = self.metrics['tot_win'] / episode
            if reward == 0:
                self.metrics['tot_draw'] += 1
                self.metrics['avg_draw'] = self.metrics['tot_draw'] / episode
            if reward == -1:
                self.metrics['tot_lose'] += 1
                self.metrics['avg_lose'] = self.metrics['tot_lose'] / episode
            for old_state,new_state,action in reversed(self.actions_played):
                new_reward = self.agent.update(old_state,new_state,action,reward)
                self.debug2(episode,old_state,new_state,action,reward,new_reward)
                reward = new_reward

            self.metrics['exploration_rate'] = self.agent.exploration_rate
            wandb.log(self.metrics,step=episode)

    def hit(self):
        return random.randint(1,10)

    def play(self,action):
        new_state = self.state.copy()
        ended = False
        if action:
            new_card = self.hit()
            new_state['player_sum'] += new_card
            if new_card == 1 and new_state['player_sum'] <= 11:
                new_state['player_sum'] += 10
                new_state['usable_ace'] = True
        else:
            ended = True
        if new_state['player_sum'] > 21:
            if new_state['usable_ace']:
                new_state['player_sum'] -= 10
                new_state['usable_ace'] = False
            else:
                ended = True
        return new_state, ended
        
    def playDealer(self):
        new_state = self.state.copy()
        ended = False
        new_card = self.hit()
        new_state['dealer_sum'] += new_card
        if new_card == 1 and new_state['dealer_sum'] <= 11:
            new_state['dealer_sum'] += 10
            self.dealer_usable_ace = True
        if new_state['dealer_sum'] > 21 and self.dealer_usable_ace:
            new_state['dealer_sum'] -= 10
            self.dealer_usable_ace = False
        if new_state['dealer_sum'] >= 17:
            ended = True
        return new_state, ended
        
    def findWinner(self):
        # player 1 | draw 0 | dealer -1
        winner = 0
        if self.state['player_sum'] > 21:
            if self.state['dealer_sum'] > 21:
                winner = -1
            else:
                winner = -1
        else:
            if self.state['dealer_sum'] > 21:
                winner = 1
            else:
                if self.state['player_sum'] < self.state['dealer_sum']:
                    winner = -1
                elif self.state['player_sum'] > self.state['dealer_sum']:
                    winner = 1
                else:
                    winner = 0
        return winner
    
    def debug1(self,episode,old_state,new_state,action):
        if(self.config['debug']):
            print("%d = %s -> %s -> %s"%(episode,old_state,action,new_state))
            input("continue?")

    def debug2(self,episode,old_state,new_state,action,old_reward,new_reward):
        if(self.config['debug']):
            print("%d = %s -> %s -> %s | %s->%s"%(episode,old_state,action,new_state,old_reward,new_reward))
            input("continue?")
