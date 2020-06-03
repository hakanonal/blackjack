import numpy as np
import random
import pickle
import wandb

class agent:

    def __init__(self,discount,exploration_rate,decay_factor, learning_rate):
        self.discount = discount # How much we appreciate future reward over current
        self.exploration_rate = exploration_rate # Initial exploration rate
        self.decay_factor = decay_factor
        self.learning_rate = learning_rate
        self.q_table = {}
        
    def get_next_action(self, state):
        if random.random() < self.exploration_rate: # Explore (gamble) or exploit (greedy)
            return self.random_action()
        else:
            return self.greedy_action(state)

    def greedy_action(self, state):
        return np.argmax(self.getQ(state))
    def random_action(self):
        return random.random() > 0.5

    def getQ(self,state):
        player_sum = state['player_sum']
        dealer_sum = state['dealer_sum']
        if (player_sum,dealer_sum) not in self.q_table:
            self.q_table[(player_sum,dealer_sum)] = [0,0]
        return self.q_table[(player_sum,dealer_sum)]

    def train(self, old_state, new_state, action, reward):
        
        old_state_prediction = self.getQ(old_state)[action]
        new_state_prediction = self.getQ(new_state)

        old_state_prediction = ((1-self.learning_rate) * old_state_prediction) + (self.learning_rate * (reward + self.discount * np.amax(new_state_prediction)))

        self.q_table[(old_state['player_sum'],old_state['dealer_sum'])][action] = old_state_prediction
        return old_state_prediction

    def update(self, old_state, new_state, action, reward):        
        reward = self.train(old_state, new_state, action, reward)
        self.exploration_rate *= self.decay_factor
        return reward

    def save(self, file="policy"):
        fw = open(file, 'wb')
        pickle.dump(self.q_table, fw)
        fw.close()
        wandb.save(file)

    def load(self, file="policy"):
        fr = open(file, 'rb')
        self.q_table = pickle.load(fr)
        fr.close()        

    def toNumPy(self):
        d = np.zeros((22,11,2))
        for state,alist in self.q_table.items():
            if(state[0] > 21):
                continue
            if(state[1] > 10):
                continue
            d[state[0]][state[1]][0] = alist[0]
            d[state[0]][state[1]][1] = alist[1]
        return d

    def printQTable(self):
        _sign = lambda x: x and (1, -1)[x<0]
        d = self.toNumPy()
        print("---------------------------------------------------------------------------------------------------------------------------")
        print("  |     1     |     2     |     3     |     4     |     5     |     6     |     7     |     8     |     9     |    10     |")
        for i in range(1,22):
            print("%02d|"%i, end="")
            for j in range(1,11):
                if _sign(d[i][j][0]) == 1:
                    print("\x1b[1;32;40m%05.2f\x1b[0m"%d[i][j][0],end="|")
                if _sign(d[i][j][0]) == -1:
                    print("\x1b[1;31;40m%05.2f\x1b[0m"%abs(d[i][j][0]),end="|")
                if _sign(d[i][j][1]) == 1:
                    print("\x1b[6;32;47m%05.2f\x1b[0m"%d[i][j][1],end="|")
                if _sign(d[i][j][1]) == -1:
                    print("\x1b[6;31;47m%05.2f\x1b[0m"%abs(d[i][j][1]),end="|")
                if(d[i][j][0] == 0):
                    print("\x1b[1;31;40m%05.2f\x1b[0m"%abs(d[i][j][0]),end="|")
                if(d[i][j][1] == 0):
                    print("\x1b[6;31;47m%05.2f\x1b[0m"%abs(d[i][j][1]),end="|")
            print("")
        print("---------------------------------------------------------------------------------------------------------------------------")

