# AI Agent For Game BlackJack

This repository's goal is to train a blackjack agent. 

## Background and Scope

This idea has been started with a conversation that I have made with my nephew. He asked me if we could build an AI agent that plays the best moves for blackjack. After some arguing on the topic, we have agreed that it would be overkill to train a deep net. Hence there were already best posible moves avaible. However this table only keeps the best action to play in a total matrix of player and dealer. So we have agreed to at least to construct a q-value table via reinforcement learning. We are expecting to construct a q-value table that will consists of the best move to to make (Hit,Double,Stand) in percentages. Our first intension was to keep it simple so we would not implement the full rules of the game. We did consider Ace as 1 not 11. I've learned that this is called hard totals. And also we will not add the double action to action space. The avaible best moves are as follows:

![Best Action Table](best_moves_in_hard_totals.jpeg)

So our plan is to put the percentages of each action for each state. (Q-Values). If you have more to do please do not hesitate to chip-in.

## Metodology

As a habbit I keep a journal for myself. I keep it as jupyter notebook [here](experiment.ipynb). You can read and see my development process.

## Conclusion

This repository is mostly completetd via this project board [here](https://github.com/hakanonal/blackjack/projects/1). It is curentlly on training process. You can watch the training process via [this](https://app.wandb.ai/hakanonal/blackjack) dashboard. 

In regular basis I commit the best policy to the repository. You can use [this](https://github.com/hakanonal/blackjack/blob/master/qtable.ipynb) notebook to check the q-values in the q-table.

Addtionally we have decided to expand the scope and include the full rules of the blackjack (hard/soft totals and include double action). This addittional scope is curentlly under development.

If you have any contrubutions please do not hesitade to open a [new issue](https://github.com/hakanonal/blackjack/issues/new).