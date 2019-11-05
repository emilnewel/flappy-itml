from ple.games.flappybird import FlappyBird
from ple import PLE
import random

class IfElseAgent:

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        print("state: %s" % state)

        bird_position = state['player_y']
        bird_vel = state['player_vel']
        next_pipe_distance = state['next_pipe_dist_to_player']
        next_pipe_center = state['next_pipe_bottom_y'] - 50
        next_next_pipe_distance = state['next_next_pipe_dist_to_player']
        next_next_pipe_center =state['next_next_pipe_bottom_y'] - 50
        action = 0
        if next_pipe_distance < 10:
            next_pipe_distance = next_next_pipe_distance
            next_pipe_center = next_next_pipe_center

        if bird_position < next_pipe_center:
            action = 1
        if next_pipe_distance < 100:
            if bird_vel < 0:
                action = 0
        else:
            if bird_vel < 0:
                action = 1

        return action

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    # reward_values = agent.reward_values
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=True, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        action = agent.policy(env.game.getGameState())

        # step the environment
        reward = env.act(env.getActionSet()[action])
        print("reward=%d" % reward)

        # TODO: for training let the agent observe the current state transition

        score += reward
        
        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0

agent = IfElseAgent()
run_game(1, agent)
