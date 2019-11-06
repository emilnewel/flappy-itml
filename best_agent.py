from flappy_agent import FlappyAgent
import pickle
import numpy as np
import sys

class BestAgent(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)

        self.pipe_next_pipe_difference_bins = np.linspace(-158, 158, 5)
        self.distance_to_pipe_bins = np.linspace(0, 65, 4)
        self.bird_pipe_difference_bins = np.linspace(0, 100, 11)

    
    def state_tf(self, state):

        bird_y = state["player_y"]
        bird_vel = state["player_vel"]
        pipe_top_y = state["next_pipe_top_y"]
        next_pipe_top_y = state["next_next_pipe_top_y"]
        distance_to_pipe = state["next_pipe_dist_to_player"]

        bird_pipe_difference = bird_y - pipe_top_y
        pipe_next_pipe_difference = pipe_top_y - next_pipe_top_y

        bird_pipe_difference_bin = np.digitize([bird_pipe_difference], self.bird_pipe_difference_bins)[0]
        pipe_next_pipe_difference_bin = np.digitize([pipe_next_pipe_difference], self.pipe_next_pipe_difference_bins)[0]
        distance_to_pipe_bin = np.digitize([distance_to_pipe], self.distance_to_pipe_bins)[0]

        if bird_vel <= -8:
            bird_vel = -8

        return (bird_pipe_difference_bin, bird_vel, pipe_next_pipe_difference_bin, distance_to_pipe_bin)



    
    def observe(self, s1, a, r, s2, end):

        s1 = self.state_tf(s1)
        s2 = self.state_tf(s2)
        
        # 
        self.update_counts((s1, a))
        self.learn_from_observation(s1, a, r, s2)

        # Count episodes
        if end:
            self.num_of_episodes += 1


    def learn_from_observation(self, s1, a, r, s2):        
        
        # Get state values
        Qs1a = self.Q.get((s1, a))
        if Qs1a is None:
            Qs1a = 0

        # Calculate return
        tot_reward = r + self.gamma * self.get_max_a(s2) 

        # Update Q table
        self.Q[(s1, a)] = Qs1a + self.learning_rate * (tot_reward - Qs1a) # update rule

agent = BestAgent('best_agent')

try:    # If agent already exists, load it's snapshot and use it.
    with open('best_agent/agent.pkl', 'rb') as f:
        agent = pickle.load(f)
        print('Running snapshot {}'.format(agent.num_of_episodes))
except:
    print('No agent snapshot exists, starting new one.')

# Otherwise start a new one.
agent.run(sys.argv[1]) #Either 'train' or 'play' should be passed to argv