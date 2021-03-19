import os
import pickle
import random

import numpy as np

from agent_funcs import *

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not self.train:
        if not os.path.isfile("bombstate_action.pt") or not os.path.isfile("seen_bombstates.pt"):
            raise Exception("No Files found for bombstates")
        else:

            self.logger.info("Loading model from saved state.")
            with open("seen_bombstates.pt", "rb") as file:
                self.seen_bombstates = pickle.load(file)
            
            with open("bombstate_action.pt", "rb") as file:
                self.bombstate_action = pickle.load(file)

    #setting up coordinates for rotating the board
    setup_coords(self)

    self.seen_bombstates = []
    self.bombstate_action = {}
    self.last_bombstate = None
    
    try:
        with open("seen_bombstates.pt", "rb") as file:
            self.seen_bombstates = pickle.load(file)
        
        with open("bombstate_action.pt", "rb") as file:
            self.bombstate_action = pickle.load(file)
    except:
        self.seen_bombstates = []
        self.bombstate_action = {}

        
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    
    
    new_game_state, state = game_state_transformer(self, game_state)

    bombstate = threat_transformer(self, game_state)[0]

    if bombstate == None:
        # if theres no bomb bomb
        return "BOMB"

    self.logger.debug(f"{bombstate}")

    if bombstate not in self.seen_bombstates:
        self.seen_bombstates.append(bombstate)

        for action in ACTIONS:
            self.bombstate_action[bombstate + str(action)] = 1
        
        self.last_bombstate = bombstate

        self.logger.info("Choosing action purely at random.")
        return np.random.choice(ACTIONS)
    else:
        a = ["WAIT"]
        val = self.bombstate_action[bombstate + "WAIT"]
        for action in ACTIONS:
            if self.bombstate_action[bombstate + action] > val:
                a = [action]
            elif self.bombstate_action[bombstate + action] == val:
                a.append(action)
            
        if len(a) == 1:
            return a[0]
        else:
            if a[0] != "WAIT":
                return np.random.choice(a)
            else:
                del a[0]
                if len(a) == 1:
                    return a[0]
                else:
                    return np.random.choice(a)

        
    
    


    for action in ACTIONS:

        self.bombstate_action[bombstate + str(action)]

        
    ownpos = new_game_state["self"][3]
    coin, min = get_nearest_coin_position(ownpos, new_game_state["coins"])


    ## EXPLORATION
    random_prob = 1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(["LEFT", "RIGHT", "UP", "DOWN"])

    # self.logger.debug(f"{a}")
    # if a in ["LEFT", "RIGHT", "UP", "DOWN"]:
    #                 if state == "ru":
    #                     a = self.order_ru[a]
    #                 if state == "rd":
    #                     a = self.order_rd[a]
    #                 if state == "ld":
    #                     a = self.order_ld[a]
    # self.logger.debug(f"{a}")
    # return a


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)