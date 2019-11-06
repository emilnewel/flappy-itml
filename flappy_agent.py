from ple.games.flappybird import FlappyBird
from ple import PLE
import random

import pickle
import pandas as pd
import numpy as np
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
        self.state_action_counter = {}
        self.num_of_episodes = 0
        self.num_of_frames = 0
    

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
        G1 = self.Q.get((state, 0))
        G2 = self.Q.get((state, 1))

        if G1 is None:
            G1 = 0
        if G2 is None:
            G2 = 0

        if G1 == G2:
            return random.randint(0, 1)
        elif G1 > G2:
            return 0
        else:
            return 1

    def get_max_a(self, state):
        G1 = self.Q.get((state, 0))
        G2 = self.Q.get((state, 1))

        if G1 is None:
            G1 = 0
        if G2 is None:
            G2 = 0

        if G1 > G2:
            return G1
        else:
            return G2

    def update_counts(self, s_a):
        if s_a in self.state_action_counter:
            self.state_action_counter[s_a] += 1
        else:
            self.state_action_counter[s_a] = 1
        self.num_of_frames += 1
    
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
        #Check if the agent folder exists
        #If not, create it.
        if not os.path.exists(self.name):
            print(self.name)
            os.mkdir(self.name)

        reward_values = self.reward_values()        
        env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True , rng=None, reward_values=reward_values)
        env.init()

        score = 0
        while self.num_of_frames <= 1000000:
            # pick an action
            state1 = env.game.getGameState()
            action = self.training_policy(state1)

            reward = env.act(env.getActionSet()[action])

            state2 = env.game.getGameState()

            end = env.game_over() or score >= 100 # Stop after reaching 100 pipes
            self.observe(state1, action, reward, state2, end)

            # reset the environment if the game is over
            if end:
                env.reset_game()
                score = 0

            if self.num_of_frames % 25000 == 0:
                print('++++++++++++++++++++++++++')
                
                print('Episodes finished: {}'.format(self.num_of_episodes))
                print('Number of frames: {}'.format(self.num_of_frames))

                self.score()

                with open('{}/agent.pkl'.format(self.name), 'wb') as f:
                    pickle.dump((self), f, pickle.HIGHEST_PROTOCOL)

                print('++++++++++++++++++++++++++\n')

    def play(self):
        print('Playing {} agent after training for {} episodes or {} frames'.format(self.name, self.num_of_episodes, self.num_of_frames))
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
        
        avg_score = sum(scores) / float(len(scores))
        print('Games played: {}'.format(total_episodes))
        print('Average score: {}'.format(avg_score))

        if training:
            score_file = '{}/scores.csv'.format(self.name)
            # If file doesn't exist, add the header
            if not os.path.isfile(score_file):
                with open(score_file, 'a') as f:
                    f.write('avg_score,episode_count,num_of_frames,min,max\n')

            # Append scores to the file
            with open(score_file, 'a') as f:
                f.write('{},{},{},{},{}\n'.format(avg_score, self.num_of_episodes, self.num_of_frames, min(scores), max(scores)))
                
        else:
            with open('scores.txt', 'a') as f:
                for score in scores:
                    f.write('{},{}\n'.format(self.name, score))