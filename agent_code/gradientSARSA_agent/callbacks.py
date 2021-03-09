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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


#initial guess
def q_hat(S,A,w):
    P=S[0]
    P_vec = np.full_like(S,P)
    diff = np.linalg.norm(S-P_vec,axis=1)

    return w@(A*diff**(-1))


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    #initial guess
    def q_hat(S,A,w):
        P=S[0]
        P_vec = np.full_like(S,P)
        diff = np.linalg.norm(S-P_vec,axis=1)

        return w@(A*diff)

    S= state_to_features(game_state)
    action = range(len(ACTIONS))
    if len(self.model)!=len(S):
        
        w=np.concatenate((np.array(self.model),np.full(np.abs(len(S)-len(self.model)),0.0)))
    else:
        w=self.model
    
    assert len(S)==len(w)
    greedy_ind = np.argmax(np.array([q_hat(S,a,w) for a in action]))
    greedy=ACTIONS[greedy_ind]
    
          
    # todo Exploration vs exploitation
    epsilon = 0.1
    
    if self.train:
        #self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        return np.random.choice([greedy,np.random.choice(ACTIONS,1)],1,p=[1-epsilon,epsilon])

    self.logger.debug("Querying model for action")
    
    return greedy


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

    #Work with polynomial Basis (as seen in 9.5.1, p.210f.) (LS)

    # For example, you could construct several channels of equal shape, ...
    channels = []
    #for collecting coins we consider our own position and the position of coins
    
    states = np.array([game_state['self'][3]]+game_state['coins'])
    """
    for i in range(len(states)):
        for k in range(3):
            for j in range(i):
                for l in range(3):
                    channels.append((states[i]**k)*(states[j]**l))
    """
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector
    #return stacked_channels.reshape(-1)
    
    return states
