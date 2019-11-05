from flappy_agent import FlappyAgent    # Flappy agent
import pickle                           # Used to load agent snapshots
import numpy as np                      # For working with containers
import sys                              # Used to read argv

class QLearning(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)
        self.player_y_bins = np.linspace(0, 512, 15)
        self.player_vel_bins = np.linspace(-8, 10, 15)
        self.next_pipe_dist_bins = np.linspace(0, 288, 15)
        self.next_pipe_top_bins = np.linspace(0, 512, 15)
    
    def state_tf(self, state):
        player_y = np.digitize([state['player_y']], self.player_y_bins)[0]
        player_vel = np.digitize([state['player_vel']], self.player_vel_bins)[0]
        next_pipe_dist_to_player = np.digitize([state['next_pipe_dist_to_player']], self.next_pipe_dist_bins)[0]
        next_pipe_top = np.digitize([state['next_pipe_top_y']], self.next_pipe_top_bins)[0]

        return (player_y, player_vel, next_pipe_dist_to_player, next_pipe_top)

    
    def observe(self, s1, a, r, s2, end):
        s1 = self.state_tf(s1)
        s2 = self.state_tf(s2)
        
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
        self.Q[(s1, a)] = Qs1a + self.learning_rate * (G - Qs1a)
    
agent = QLearning('q_learning')

try:    # If agent already exists, load it's snapshot and use it.
    with open('q_learning/agent.pkl', 'rb') as f:
        agent = pickle.load(f)
        print('Running snapshot {}'.format(agent.episode_count))
except:
    pass

# Otherwise start a new one.
agent.run(sys.argv[1]) #Either 'train' or 'play' should be passed to argv