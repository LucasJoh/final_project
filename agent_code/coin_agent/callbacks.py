import os
import pickle
import random

import numpy as np


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

    self.last_a = None

     # coordinate setup
    coords = [[(i,j) for j in range(1, 16)] for i in range(1,16)]
    self.ld = list(zip(*coords))[::-1]
    self.rd = list(zip(*self.ld))[::-1]
    self.ru = list(zip(*self.rd))[::-1]

    self.order_ru = {"LEFT": "DOWN", "RIGHT": "UP", "UP": "LEFT", "DOWN": "RIGHT"}
    self.order_rd = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
    self.order_ld = {"LEFT": "UP", "UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT"}

    
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    # try:
    #     self.coin_states_actions = pickle.load(open("coin_states_actions.pt", "rb"))
    # except:
    #     raise Exception("coin_states_actions could not be loaded")
    #     self.coin_states_actions = None
    #     self.logger.debug(f"coin_states_actions could not be loaded")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.logger.debug(f"{game_state['self'][3]}")

    new_game_state, state= game_state_transformer(self, game_state)

    self.logger.debug(f"{new_game_state['self'][3]}")
    ownpos = new_game_state["self"][3]
    coin, min = get_nearest_coin_position(ownpos, game_state["coins"])
    state_id = str(ownpos[0]) + str(ownpos[1]) + str(coin[0]) + str(coin[1])

    val = -10000000
    ACTIONS = []

    for action in ["LEFT", "RIGHT", "UP", "DOWN"]:
        if self.coin_states_actions[state_id + action] == val:
            ACTIONS.append(action)

        elif self.coin_states_actions[state_id + action] > val:
            val = self.coin_states_actions[state_id + action]
            ACTIONS = [action]
    

    if len(ACTIONS) > 1:
        a = np.random.choice(ACTIONS)
    else:
        a = ACTIONS[0]

    if val == -10000000:
        a = "WAIT"
    # # todo Exploration vs exploitation
    # random_prob = .1
    # if self.train and random.random() < random_prob:
    #     self.logger.debug("Choosing action purely at random.")
    #     # 80%: walk in any direction. 10% wait. 10% bomb.
    #     return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0])

    # self.logger.debug("Querying model for action.")
    # return np.random.choice(ACTIONS, p=self.model)


    if a in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    if state == "ru":
                        a = self.order_ru[a]
                    if state == "rd":
                        a = self.order_rd[a]
                    if state == "ld":
                        a = self.order_ld[a]
    self.last_a = a
    return a

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

#turn board
def game_state_transformer(self, game_state):
    new_game_state = game_state
    own_pos = game_state['self'][3]
    state = ""

    for i in range(len(new_game_state['bombs'])):
        new_game_state['bombs'][i] = list(new_game_state['bombs'][i])
 
    if own_pos[1] > 8:

        if own_pos[0] > 8:

            new_game_state['field'] = np.array(list(zip(*new_game_state['field']))[::-1])
            new_game_state['field'] = np.array(list(zip(*new_game_state['field']))[::-1])

            new_game_state['explosion_map'] = np.array(list(zip(*new_game_state['explosion_map']))[::-1])
            new_game_state['explosion_map'] = np.array(list(zip(*new_game_state['explosion_map']))[::-1])

            own_pos = self.rd[own_pos[0] - 1][own_pos[1] - 1]

            for i in range(len(new_game_state['coins'])):
                new_game_state['coins'][i] = self.rd[new_game_state['coins'][i][0] - 1][new_game_state['coins'][i][1] - 1]

            for i in range(len(new_game_state['bombs'])):
                new_game_state['bombs'][i][0] = self.rd[new_game_state['bombs'][i][0][0] - 1][new_game_state['bombs'][i][0][1] - 1]
            
            state = "rd"
        else:

            new_game_state['field'] = np.array(list(zip(*new_game_state['field']))[::-1])
            new_game_state['field'] = np.array(list(zip(*new_game_state['field']))[::-1])
            new_game_state['field'] = np.array(list(zip(*new_game_state['field']))[::-1])
            

            new_game_state['explosion_map'] = np.array(list(zip(*new_game_state['explosion_map']))[::-1])
            new_game_state['explosion_map'] = np.array(list(zip(*new_game_state['explosion_map']))[::-1])
            new_game_state['explosion_map'] = np.array(list(zip(*new_game_state['explosion_map']))[::-1])
            
            own_pos = self.ru[own_pos[0] - 1][own_pos[1] - 1]

            for i in range(len(new_game_state['coins'])):
                new_game_state['coins'][i] = self.ru[new_game_state['coins'][i][0] - 1][new_game_state['coins'][i][1] - 1]

            for i in range(len(new_game_state['bombs'])):
                new_game_state['bombs'][i][0] = self.ru[new_game_state['bombs'][i][0][0] - 1][new_game_state['bombs'][i][0][1] - 1]

            state = "ru"
            
            
    elif own_pos[0] > 8:

        new_game_state['field'] = np.array(list(zip(*new_game_state['field']))[::-1])
            
        new_game_state['explosion_map'] = np.array(list(zip(*new_game_state['explosion_map']))[::-1])

        own_pos = self.ld[own_pos[0] - 1][own_pos[1] - 1]

        for i in range(len(new_game_state['coins'])):
            new_game_state['coins'][i] = self.ld[new_game_state['coins'][i][0] - 1][new_game_state['coins'][i][1] - 1]

        for i in range(len(new_game_state['bombs'])):
            new_game_state['bombs'][i][0] = self.ld[new_game_state['bombs'][i][0][0] - 1][new_game_state['bombs'][i][0][1] - 1]

        state = "ld"

    new_game_state['self'] = list(new_game_state['self'])
    new_game_state['self'][3] = own_pos

    return new_game_state, state

    
    

# gets position of nearest coin
def get_nearest_coin_position(own_pos, coin_pos):
    min = 1000
    coin = (0,0)

    for c in coin_pos:
        dist = np.linalg.norm(np.array(c) - np.array(own_pos))

        if dist < min:
            min = dist
            coin = c
    if coin == (0,0):
        return None
    else:
        return coin, min

    