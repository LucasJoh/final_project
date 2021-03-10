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
THREATEN_BY_ONE = "THREATEN_BY_ONE"
GETTING_CLOSER = "GETTING_CLOSER"

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

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    if old_game_state == None:
        pass
    else:
        S=state_to_features(old_game_state)
        
        S_prime = state_to_features(new_game_state)
        #A=np.where(self_action==ACTIONS)[0]
        A=self_action
        
        threat=0
        #new events
        if (S_prime[0] in range(S_prime[4]-3,S_prime[4]+3)) or (S_prime[1] in range(S_prime[5]-3,S_prime[5]+3)):
            threat+=1
        
        if threat==1:
            events.append(THREATEN_BY_ONE)

        old_p = S[0:2]
        new_p =S_prime[:2]
        old_c = S[2:4]
        new_c = S_prime[2:4]
        if np.linalg.norm(old_p-old_c)>np.linalg.norm(new_p-new_c):
            events.append(GETTING_CLOSER)

        gamma=0.9
        alpha=0.3
        
        """
        if (self.model).all() ==None:
            w=np.full(0.0,len(S))
        elif len(self.model)!=len(S):
            
            w=np.concatenate((np.array(self.model),np.full(np.abs(len(S)-len(self.model)),0.0)))
        else:
        """
        w=self.model
        
        action=range(1,len(ACTIONS)+1)
        epsilon=0.05
        greedy_ind = np.argmax(np.array([q_hat(S,a,w) for a in ACTIONS]))
        greedy=ACTIONS[greedy_ind]
        A_prime=np.random.choice((greedy,np.random.choice(ACTIONS,1)),1,[1-epsilon,epsilon])
        
        grad=np.zeros_like(w)
        for i in range(len(w)):#use that q is just linear wrt w
            w_temp=w 
            w_temp[i]=1
            grad[i]= q_hat(S,A,w_temp)
        #use gradient SARSA (p.244)
        w=w+alpha*(reward_from_events(self,events)-gamma*q_hat(S_prime,A_prime,w)-q_hat(S,A,w))*grad

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

    S=state_to_features(last_game_state)
        
        
    #A=np.where(last_action==ACTIONS)[0]+1
    A=last_action
        
    gamma=0.9
    alpha=0.3

    """    
    if (self.model).all() ==None:
        w=np.full(0.0,len(S))
    elif len(self.model)!=len(S):
            
        w=np.concatenate((np.array(self.model),np.full(np.abs(len(S)-len(self.model)),0.0)))
    else:
    """
    w=self.model
        
    action=range(1,len(ACTIONS)+1)
    epsilon=0.05
    greedy_ind = np.argmax(np.array([q_hat(S,a,w) for a in ACTIONS]))
    
    grad=np.zeros_like(w)
    for i in range(len(w)):#use that q is just linear wrt w
        w_temp=w 
        w_temp[i]=1
        grad[i]= q_hat(S,A,w_temp)

    w=w+alpha*(reward_from_events(self,events)-gamma*q_hat(S,A,w)-q_hat(S,A,w))*grad

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
        e.COIN_COLLECTED: 100,
        #e.KILLED_OPPONENT: 5,
        e.MOVED_DOWN: -.1,
        e.MOVED_LEFT: -.1,
        e.MOVED_RIGHT: -.1,
        e.MOVED_UP: -.1,
        e.WAITED: -.1,
        e.INVALID_ACTION: -1,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -5,
        THREATEN_BY_ONE: -0.1,
        GETTING_CLOSER: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
