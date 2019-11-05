from flappy_agent import FlappyAgent    # Flappy agent
import pickle                           # Used to load agent snapshots
import numpy as np                      # For working with containers
import sys                              # Used to read argv


class MonteCarlo(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)
        
        self.episode = []
        self.bird_y_bins = np.linspace(0, 512, 15)
        self.bird_vel_bins = np.linspace(-8, 10, 15)
        self.next_pipe_dist_bins = np.linspace(0, 288, 15)
        self.next_pipe_top_bins = np.linspace(0, 512, 15)
        
    #Tranform the state into what we're supposed to use
    def state_tf(self, state):
        bird_y = np.digitize([state['player_y']], self.bird_y_bins)[0]
        next_pipe_top = np.digitize([state['next_pipe_top_y']], self.next_pipe_top_bins)[0]
        next_pipe_dist_to_player = np.digitize([state['next_pipe_dist_to_player']], self.next_pipe_dist_bins)[0]
        bird_vel = np.digitize([state['player_vel']], self.bird_vel_bins)[0]
        
        return (bird_y, next_pipe_top, next_pipe_dist_to_player, bird_vel)


    def observe(self, s1, a, r, s2, end):

        s1 = self.state_tf(s1)
        self.episode.append((s1, a, r))
        
        # count for graphs
        self.update_counts((s1, a))

        # If end of episode, learn from the episode, and clear the list of states.
        if end:
            self.learn_from_episode(self.episode)
            self.num_of_episodes += 1
            self.episode = []


    def learn_from_episode(self, episode):        
        tot_reward = 0
        for s, a, r in reversed(episode):
            tot_reward = r + self.gamma * tot_reward
            if (s, a) in self.Q:
                self.Q[(s, a)] = self.Q[(s, a)] + self.learning_rate * (tot_reward - self.Q[(s, a)]) # update rule
            else:
                self.Q[(s, a)] = tot_reward # initialize to the first example
    
agent = MonteCarlo('opmc')

try:    # If agent already exists, load it's snapshot and use it.
    with open('opmc/agent.pkl', 'rb') as f:
        agent = pickle.load(f)
        print('Running snapshot {}'.format(agent.num_of_episodes))
except:
    pass

# Otherwise start a new one.
agent.run(sys.argv[1]) #Either 'train' or 'play' should be passed to argv