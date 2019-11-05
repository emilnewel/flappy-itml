from ple.games.flappybird import FlappyBird
from ple import PLE
import random

import pickle
import pandas as pd
import numpy as np
import scipy.stats as st
import os.path
import os



class FlappyAgent:
    def __init__(self, name):
        self.name = name
        self.Q = {}
        self.gamma = 1 # discount
        self.learning_rate = 0.1
        self.epsilon = 0.1
        
        # For graphs
        self.s_a_counts = {}
        self.episode_count = 0
        self.frame_count = 0
    

    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {'positive': 1.0, 'tick': 0.0, 'loss': -5.0}
    

    def state_tf(self, state):
        pass


    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
        """
        pass


    def get_argmax_a(self, state):
        G0 = self.Q.get((state, 0))
        G1 = self.Q.get((state, 1))

        if G0 is None:
            G0 = 0
        if G1 is None:
            G1 = 0

        if G0 == G1:
            return random.randint(0, 1)
        elif G0 > G1:
            return 0
        else:
            return 1


    def update_counts(self, s_a):
        if s_a in self.s_a_counts:
            self.s_a_counts[s_a] += 1
        else:
            self.s_a_counts[s_a] = 1
        self.frame_count += 1
       
    def get_initial_return_value(self, state, action):
        return 0
    
    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """

        state = self.state_tf(state)

        greedy = np.random.choice([False, True], p=[self.epsilon, 1-self.epsilon])

        action = 0
        if greedy:
            action = self.get_argmax_a(state)
        else:
            action = random.randint(0, 1)

        
        return action


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """

        state = self.state_tf(state)
        action = self.get_argmax_a(state)
        return action

    def get_max_a(self, state):
        G0 = self.Q.get((state, 0))
        G1 = self.Q.get((state, 1))

        if G0 is None:
            G0 = self.get_initial_return_value(state, 0)
        if G1 is None:
            G1 = self.get_initial_return_value(state, 1)

        if G0 > G1:
            return G0
        else:
            return G1

    def make_df(self):
        d = {}
        i = 0
        for sa, G in self.Q.items():
            
            state = sa[0]
            action = sa[1]
            d[i] = self.state_to_dict(state, action, G)
            i += 1

        df = pd.DataFrame.from_dict(d, 'index')
        return df.pivot_table(index='y_difference', columns='next_pipe_dist_to_player', values=['action', 'return', 'count_seen'])


    def state_to_dict(self, state, action, G):
        return {'next_pipe_dist_to_player':state[2],
                'y_difference':state[0] - state[3],
                'action':self.get_argmax_a(state),
                'return':G,
                'count_seen':self.s_a_counts[(state, action)]}

    def run(self, arg):
        if arg == 'train':
            self.train()
        elif arg == 'play':
            self.play()
        else:
            print('Invalid argument, use "train" or "play"')


    def train(self):
        """ Runs nb_episodes episodes of the game with agent picking the moves.
            An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
        """

        if not os.path.exists(self.name):
            os.mkdir(self.name)

        reward_values = self.reward_values()        
        env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True , rng=None, reward_values=reward_values)
        env.init()

        score = 0
        while self.frame_count <= 1000000:
            # pick an action
            state1 = env.game.getGameState()
            action = self.training_policy(state1)

            # step the environment
            reward = env.act(env.getActionSet()[action])
            # print('reward=%d' % reward)

            state2 = env.game.getGameState()

            end = env.game_over() or score >= 100 # Stop after reaching 100 pipes
            self.observe(state1, action, reward, state2, end)

            # reset the environment if the game is over
            if end:
                env.reset_game()
                score = 0

            if self.frame_count % 25000 == 0:
                print('++++++++++++++++++++++++++')
                
                print('episodes done: {}'.format(self.episode_count))
                print('frames done: {}'.format(self.frame_count))

                self.score()

                with open('{}/agent.pkl'.format(self.name), 'wb') as f:
                    pickle.dump((self), f, pickle.HIGHEST_PROTOCOL)

                print('++++++++++++++++++++++++++')

    def play(self):
        print('Playing {} agent after training for {} episodes or {} frames'.format(self.name, self.episode_count, self.frame_count))
        reward_values = {'positive': 1.0, 'negative': 0.0, 'tick': 0.0, 'loss': 0.0, 'win': 0.0}

        env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None, reward_values=reward_values)
        env.init()
        
        score = 0
        last_print = 0

        nb_episodes = 50
        while nb_episodes > 0:
            # pick an action
            state = env.game.getGameState()
            action = self.policy(state)                                          

            # step the environment
            reward = env.act(env.getActionSet()[action])

            score += reward

            # reset the environment if the game is over
            if env.game_over():            
                print('Score: {}'.format(score))
                env.reset_game()
                nb_episodes -= 1
                score = 0

    def score(self, training=True, nb_episodes=10):
        reward_values = {'positive': 1.0, 'negative': 0.0, 'tick': 0.0, 'loss': 0.0, 'win': 0.0}

        env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None, reward_values=reward_values)
        env.init()

        total_episodes = nb_episodes
        score = 0
        scores = []
        while nb_episodes > 0:
            # pick an action
            state = env.game.getGameState()
            action = self.policy(state)

            # step the environment
            reward = env.act(env.getActionSet()[action])

            score += reward

            # reset the environment if the game is over
            if env.game_over() or score >= 100:
                scores.append(score)
                env.reset_game()
                nb_episodes -= 1
                score = 0
                # print(nb_episodes)
        
        avg_score = sum(scores) / float(len(scores))
        confidence_interval = st.t.interval(0.95, len(scores)-1, loc=np.mean(scores), scale=st.sem(scores))  
        if np.isnan(confidence_interval[0]):
            confidence_interval = (avg_score, avg_score)
        
        print('Games played: {}'.format(total_episodes))
        print('Average score: {}'.format(avg_score))
        print('95 confidence interval: {}'.format(confidence_interval))

        if training:
            score_file = '{}/scores.csv'.format(self.name)
            # If file doesn't exist, add the header
            if not os.path.isfile(score_file):
                with open(score_file, 'a') as f:
                    f.write('avg_score,episode_count,frame_count,interval_lower,interval_upper,min,max\n')

            # Append scores to the file
            with open(score_file, 'a') as f:
                f.write('{},{},{},{},{},{},{}\n'.format(avg_score, self.episode_count, self.frame_count, confidence_interval[0], confidence_interval[1], min(scores), max(scores)))

            count = 0
            for score in scores:
                if score >= 50:
                    count += 1
            if count >= len(scores) * 0.9:
                print('\n REACHED 50 PIPES IN {} FRAMES \n'.format(self.frame_count))
                with open('pass_50.csv', 'a') as f:
                    f.write('{},{}\n'.format(self.name, self.frame_count))
        else:
            with open('scores.txt', 'a') as f:
                for score in scores:
                    f.write('{},{}\n'.format(self.name, score))