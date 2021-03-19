import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features

from agent_funcs import *


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    try:
        with open("seen_bombstates.pt", "rb") as file:
            self.seen_bombstates = pickle.load(file)
        
        with open("bombstate_action.pt", "rb") as file:
            self.bombstate_action = pickle.load(file)
    except:
        self.seen_bombstates = []
        self.bombstate_action = {}
    self.last_three_bombstates = [None, None, None]
    self.bombstate_action_log = []


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

    
            
    bombstate = self.last_bombstate
    rewards = 0

    if bombstate != None and self_action != None:
        
        bombstate_action = bombstate + self_action

        self.last_three_bombstates.pop()

        self.last_three_bombstates.append(bombstate_action)

        self.bombstate_action_log.append(bombstate_action)

        last_bombstate, last_threats = threat_transformer(self, old_game_state)
        new_bombstate, new_threats = threat_transformer(self, new_game_state)

        if e.INVALID_ACTION in events:
            rewards = -10
            self.bombstate_action[bombstate_action] += rewards
            return None
        

        if last_threats != 0:

            old_pos = old_game_state['self'][3]
            new_pos = new_game_state['self'][3]

            old_bomb, old_dist = get_nearest_bomb_position(old_pos, old_game_state['bombs'])
            self.logger.debug(f"{old_bomb}")
            new_dist = abs(old_bomb[0] - new_pos[0]) + abs(old_bomb[1] - new_pos[1])

            if new_dist > old_dist:
                rewards += 1
            elif new_dist <= old_dist:
                rewards += -1

            if new_threats < last_threats:
                rewards += 1
            
            elif new_threats >= last_threats:
                rewards += -1

        rewards += reward_from_events(self, events)

        self.bombstate_action[bombstate_action] += rewards


    




def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    
    if e.SURVIVED_ROUND in events:
        for state in self.bombstate_action_log:
            self.bombstate_action[state] += 2
    
    elif e.KILLED_SELF in events:
        for state in self.last_three_bombstates:
            if state != None:
                self.bombstate_action[state] += -2

    elif e.GOT_KILLED in events:
        for state in self.last_three_bombstates:
            self.bombstate_action[state] += -1
    
    
    with open("bombstate_action.pt", "wb") as file:
        pickle.dump(self.bombstate_action, file)

    with open("seen_bombstates.pt", "wb") as file:
        pickle.dump(self.seen_bombstates, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -50,
        e.SURVIVED_ROUND: 5,
        e.INVALID_ACTION: -5,
        e.SURVIVED_ROUND: 10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
