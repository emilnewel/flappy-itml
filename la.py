from flappy_agent import FlappyAgent    # Flappy agent
import pickle                           # Used to load agent snapshots
import numpy as np                      # For working with containers
import sys                              # Used to read argv

class LinearApproximation(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)
        self.learning_rate = 0.00001
        self.episode = []
        self.weights = [np.array([0]*4, dtype=np.float64), np.array([0]*4, dtype=np.float64)]    

    def map_state(self, state):
        player_y = state['player_y']
        player_vel = state['player_vel']
        next_pipe_dist = state['next_pipe_dist_to_player']
        next_pipe_top = state['next_pipe_top_y']

        return (player_y, player_vel, next_pipe_dist, next_pipe_top)
    
    def observe(self, s1, a, r, s2, end):
        s1 = self.map_state(s1)
        self.episode.append((s1, a, r))
        self.update_counts((s1, a))

        if end:
            self.learn_from_episode(self.episode)
            self.num_of_episodes += 1
            self.episode = []
    
    def learn_from_episode(self, episode):
        G = 0
        for s,a,r in reversed(episode):
            G = r + self.gamma * G
            features = np.array(s, dtype=np.float64)
            self.weights[a] = self.weights[a] - self.learning_rate * (np.dot(self.weights[a], features) - G) * features


agent = LinearApproximation('la')

try:    # If agent already exists, load it's snapshot and use it.
    with open('la/agent.pkl', 'rb') as f:
        agent = pickle.load(f)
        print('Running snapshot {}'.format(agent.episode_count))
except:
    print('No agent snapshot exists, starting new one.')

# Otherwise start a new one.
agent.run(sys.argv[1]) #Either 'train' or 'play' should be passed to argv