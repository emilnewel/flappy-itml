from flappy_agent import FlappyAgent

from ple.games.flappybird import FlappyBird
from ple import PLE
import random

import pickle
import pandas as pd
import numpy as np
import scipy.stats as st
import sys


class QLearning(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)

        self.player_y_bins = np.linspace(0, 512, 15)
        self.player_vel_bins = np.linspace(-8, 10, 15)
        self.next_pipe_dist_bins = np.linspace(0, 288, 15)
        self.next_pipe_top_bins = np.linspace(0, 512, 15)
  
    
    def map_state(self, state):

        player_y = np.digitize([state['player_y']], self.player_y_bins)[0]
        player_vel = np.digitize([state['player_vel']], self.player_vel_bins)[0]
        next_pipe_dist_to_player = np.digitize([state['next_pipe_dist_to_player']], self.next_pipe_dist_bins)[0]
        next_pipe_top = np.digitize([state['next_pipe_top_y']], self.next_pipe_top_bins)[0]

        return (player_y, player_vel, next_pipe_dist_to_player, next_pipe_top)

    
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
            Qs1a = 0
        # Calculate return
        G = r + self.gamma * self.get_max_a(s2) 

        # Update Q table
        self.Q[(s1, a)] = Qs1a + self.lr * (G - Qs1a) # update rule
    
agent = QLearning('q_learning')

try:
    with open('{}/agent.pkl'.format('q_learning'), 'rb') as f:
        agent = pickle.load(f)
        print('Running snapshot {}'.format(agent.episode_count))
except:
    if sys.argv[1] == 'plot':
        print('No data available to plot')
        quit()
    print('Starting new {} agent'.format('q_learning'))

agent.run(sys.argv[1]) # Use 'train', 'play', 'score' or 'plot'