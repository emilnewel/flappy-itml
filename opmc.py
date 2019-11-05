from flappy_agent import FlappyAgent    # Flappy agent
import pickle                           # Used to load agent snapshots
import numpy as np                      # For working with containers
import sys                              # Used to read argv


class MonteCarlo(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)
        
        self.episode = []
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
        self.episode.append((s1, a, r))
        
        # count for graphs
        self.update_counts((s1, a))

        # 2. if reached the end of episode, learn from it, then reset
        if end:
            self.learn_from_episode(self.episode)
            self.episode_count += 1
            self.episode = []


    def learn_from_episode(self, episode):        
        G = 0
        for s, a, r in reversed(episode):
            G = r + self.gamma * G
            if (s, a) in self.Q:
                self.Q[(s, a)] = self.Q[(s, a)] + self.learning_rate * (G - self.Q[(s, a)]) # update rule
            else:
                self.Q[(s, a)] = G # initialize to the first example
    
agent = MonteCarlo('opmc')

try:    # If agent already exists, load it's snapshot and use it.
    with open('opmc/agent.pkl', 'rb') as f:
        agent = pickle.load(f)
        print('Running snapshot {}'.format(agent.episode_count))
except:
    pass

# Otherwise start a new one.
agent.run(sys.argv[1]) #Either 'train' or 'play' should be passed to argv