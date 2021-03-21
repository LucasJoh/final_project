import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features, q_hat

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
GETTING_CLOSER = "GETTING_CLOSER"
LOOSING_COIN = "LOOSING_COIN"
WELL_PLACED_BOMB = "WELL_PLACED_BOMB"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    

def grad_q_hat(S,A,w):
    """
    Just a short implementation of gradient of q_hat with respect to w.
    This of course only works for q_hat as defined in callbacks.py

    :param S: A dictionary with current game state
    :param A: A string with an action out of list ACTIONS
    :param w: An array with current weights
    :return: An array with the values of the gradient
    """
    grad=np.zeros_like(w)
    for i in range(len(w)):#use that q is just linear wrt w
        w_temp=np.zeros_like(w)
        w_temp[i]=1
        grad[i]= q_hat(S,A,w_temp)
    return grad


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state == None: #avoid bug
        pass
    else:
        #shorter nomenclature
        S=old_game_state
        S_prime = new_game_state
        A=self_action
        R=reward_from_events(self,events)

        old_features = state_to_features(S)
        new_features = state_to_features(S_prime)
        
        #trigger event when agent gets closer to coin or looses it
        if new_features[0]>old_features[0]:
            events.append(GETTING_CLOSER)
            
        if (new_features[0]<old_features[0]):
            events.append(LOOSING_COIN)

        #trigger event that gives good reward for a well-placed bomb in the sense of reached crates
        if old_features[3]>=(2/13) and self_action=="BOMB":
            events.append(WELL_PLACED_BOMB)
            
        #hyperparameter for training algorithm  
        #TODO Find good hyperparameter (Sutton 9.6) @Simon 
        gamma=0.7
        alpha=0.2
        
        w=self.model
        
        ###first crucial distinction: which algorithm to choose. MAin difference is how to choose A'
        #Right now it is Q-learning, but SARSA or expected-SARSA are also possible (lecture 3, p.2; Sutton 6 and 10)
        #TODO: Try different algorithms @Simon

        tester = np.array([(q_hat(S,a,w)) for a in ACTIONS])
        if np.all(tester==tester[0]):###if all entries are equal the first entry is chosen by argmax
            greedy = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
        else:
            greedy_ind = np.argmax(tester)
            greedy=ACTIONS[greedy_ind]
        
        ###SARSA
        #epsilon=0.05
        #A_prime=np.random.choice((greedy,np.random.choice(ACTIONS,1)),1,[1-epsilon,epsilon])

        ###Q-learning
        A_prime = greedy 
        
        # You can find the algorithm for SARSA in Sutton, p.244
        # But it is not hard to find that algorithm:
        # We want to find a better w, because q_hat=Xw (with X beeing the feature vector)
        # If we define R+gamma*g_hat(S',A',w):=Y (as seen in lecture 3, p.2) we want to minimize the squared Error (Y-Xw)^2=(Y-q_hat)^2
        # We want to find a w that the squared error is minimal:
        # We take gradient of (Y-q_hat)^2 with respect to w and get a direction in which we have to correct our w (like in every Newton-method)
        
        
        ####Iteration
        w=w+alpha*(R+gamma*q_hat(S_prime,A_prime,w)-q_hat(S,A,w))*grad_q_hat(S,A,w)

        self.model=w

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    S=last_game_state   
    A=last_action

    #hyperparameter   
    gamma=0.7
    alpha=0.2

    w=self.model
    
    ####Iteration
    w=w+alpha*(reward_from_events(self,events)-gamma*q_hat(S,A,w)-q_hat(S,A,w))*grad_q_hat(S,A,w)

    self.model=w

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        #e.KILLED_OPPONENT: 5,
        e.MOVED_DOWN: -.1,
        e.MOVED_LEFT: -.1,
        e.MOVED_RIGHT: -.1,
        e.MOVED_UP: -.1,
        e.WAITED: -.1,
        e.INVALID_ACTION: -1,
        e.KILLED_SELF: -1,
        e.GOT_KILLED: -0.5,
        e.CRATE_DESTROYED: 0.2,
        e.COIN_FOUND: 5,
        e.BOMB_DROPPED: 0.5,
        e.BOMB_EXPLODED: 0.0,
        GETTING_CLOSER: 3,
        LOOSING_COIN: -0.2,
        WELL_PLACED_BOMB: 1,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
