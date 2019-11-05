from flappy_agent import FlappyAgent

from ple.games.flappybird import FlappyBird
from ple import PLE
import random

import pickle
import pandas as pd
import numpy as np
import scipy.stats as st
import sys
import math
import matplotlib.pyplot as plt

class BestAgent(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)

        self.bird_pipe_difference_bins = np.linspace(13, 76, 3)

    
    def state_tf(self, state):

        bird_y = state["player_y"]
        bird_vel = state["player_vel"]
        pipe_top_y = state["next_pipe_top_y"]

        bird_pipe_difference = bird_y - pipe_top_y

        bird_pipe_difference_bin = np.digitize([bird_pipe_difference], self.bird_pipe_difference_bins)[0]

        if bird_vel <= -8:
            bird_vel = 0
        elif bird_vel > 0:
            bird_vel = 2
        else:
            bird_vel = 1

        return (bird_pipe_difference_bin, bird_vel)

    
    def observe(self, s1, a, r, s2, end):

        s1 = self.state_tf(s1)
        s2 = self.state_tf(s2)
        
        # count for graphs
        self.update_counts((s1, a))

        self.learn_from_observation(s1, a, r, s2)

        # Count episodes
        if end:
            self.num_of_episodes += 1


    def learn_from_observation(self, s1, a, r, s2):        
        
        # Get state values
        Qs1a = self.Q.get((s1, a))
        if Qs1a is None:
            Qs1a = self.get_initial_return_value(s1, a)

        # Calculate return
        G = r + self.gamma * self.get_max_a(s2) 

        # Calculate learning rate
        lr = 1.0 / self.s_a_counts[(s1,a)]

        # Update Q table
        self.Q[(s1, a)] = Qs1a + lr * (G - Qs1a) # update rule

    
    def get_initial_return_value(self, state, action):
        if state[0] == 0 or state[0] == 3:
            return 4
        else:
            return 5

agent = BestAgent6('best_agent')

try:    # If agent already exists, load it's snapshot and use it.
    with open('best_agent/agent.pkl', 'rb') as f:
        agent = pickle.load(f)
        print('Running snapshot {}'.format(agent.num_of_episodes))
except:
    pass

# Otherwise start a new one.
agent.run(sys.argv[1]) #Either 'train' or 'play' should be passed to argv