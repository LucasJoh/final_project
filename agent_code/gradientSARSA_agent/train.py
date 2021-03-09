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
        self.model = np.ones(5)
    else:
        S=state_to_features(old_game_state)
        
        S_prime = new_game_state
        A=np.where(self_action==ACTIONS)[0]
        
        gamma=1
        alpha=0.1
        
        if (self.model).all() ==None:
            w=np.full(0.0,len(S))
        elif len(self.model)!=len(S):
            
            w=np.concatenate((np.array(self.model),np.full(np.abs(len(S)-len(self.model)),0.0)))
        else:
            w=self.model
        
        action=range(len(ACTIONS))
        epsilon=0.05
        greedy_ind = np.argmax(np.array([q_hat(S,a,w) for a in action]))
        A_prime=np.random.choice((greedy_ind,np.random.choice(action,1)),1,[1-epsilon,epsilon])
        greedy=ACTIONS[greedy_ind]
        y=np.full_like(w,q_hat(S,A,w))
        grad = np.concatenate((np.diff(y)/np.diff(w),np.array([-1])))
        #use gradient SARSA (p.244)
        w=w+alpha*(reward_from_events(self,events)-gamma*q_hat(state_to_features(S_prime),A_prime,w)-q_hat(S,A,w))*grad

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
        
        
    A=np.where(last_action==ACTIONS)[0]
        
    gamma=1
    alpha=0.1
        
    if (self.model).all() ==None:
        w=np.full(0.0,len(S))
    elif len(self.model)!=len(S):
            
        w=np.concatenate((np.array(self.model),np.full(np.abs(len(S)-len(self.model)),0.0)))
    else:
        w=self.model
        
    action=range(len(ACTIONS))
    epsilon=0.05
    greedy_ind = np.argmax(np.array([q_hat(S,a,w) for a in action]))
    y=np.full_like(w,q_hat(S,A,w))
    grad = np.concatenate((np.diff(y)/np.diff(w),np.array([0])))
        #use gradient SARSA (p.244)
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
        e.COIN_COLLECTED: 1,
        #e.KILLED_OPPONENT: 5,
        e.MOVED_DOWN: -.1,
        e.MOVED_LEFT: -.1,
        e.MOVED_RIGHT: -.1,
        e.MOVED_UP: -.1,
        e.WAITED: -.1,
        e.INVALID_ACTION: -.1,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
