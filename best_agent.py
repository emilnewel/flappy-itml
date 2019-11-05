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




class BestAgent6(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)

        self.player_pipe_difference_bins = np.linspace(13, 76, 3)

    
    def map_state(self, state):

        player_y = state["player_y"]
        player_vel = state["player_vel"]
        pipe_top_y = state["next_pipe_top_y"]

        player_pipe_difference = player_y - pipe_top_y

        player_pipe_difference_bin = np.digitize([player_pipe_difference], self.player_pipe_difference_bins)[0]

        if player_vel <= -8:
            player_vel = 0
        elif player_vel > 0:
            player_vel = 2
        else:
            player_vel = 1

        return (player_pipe_difference_bin, player_vel)

    
    def observe(self, s1, a, r, s2, end):

        s1 = self.map_state(s1)
        s2 = self.map_state(s2)
        
        # count for graphs
        self.update_counts((s1, a))

        self.learn_from_observation(s1, a, r, s2)

        # Count episodes
        if end:
            self.episode_count += 1


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


    
name = "best_agent_6"

try:
    name += "_{}".format(sys.argv[2])
except:
    pass

agent = BestAgent6(name)

try:
    with open("{}/agent.pkl".format(name), "rb") as f:
        agent = pickle.load(f)
        print("Running snapshot {}".format(agent.episode_count))
except:
    if sys.argv[1] == "plot":
        print("No data available to plot")
        quit()
    print("Starting new {} agent".format(name))

agent.run(sys.argv[1]) # Use 'train', 'play', 'score' or 'plot'