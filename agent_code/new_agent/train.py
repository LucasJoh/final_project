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
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    try:
        self.coin_states_actions = pickle.load(open("coin_states_actions.pt", "rb"))
    except:
        self.coin_states_actions = None
        self.logger.debug(f"coin_states_actions could not be loaded")

    self.preoldpos = None

    if self.coin_states_actions == None:
        self.coin_states_actions = {}
        for i in range(1, 9):
            for j in range(1, 9):
                for k in range(1, 16):
                    for l in range(1, 16):
                        self.coin_states_actions[str(i) + str(j) + str(k) + str(l) + "LEFT"] = 1
                        self.coin_states_actions[str(i) + str(j) + str(k) + str(l) + "RIGHT"] = 1
                        self.coin_states_actions[str(i) + str(j) + str(k) + str(l) + "UP"] = 1
                        self.coin_states_actions[str(i) + str(j) + str(k) + str(l) + "DOWN"] = 1



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

    reward = 0
    if self_action in ["LEFT", "RIGHT", "UP", "DOWN"]:

        # transforming old game state and new game state to rotated game states
        old_game_state, oldstate = game_state_transformer(self, old_game_state)
 
        new_game_state, new_state = game_state_transformer(self, new_game_state)

        # getting oldposition and new position on rotated board
        oldpos = old_game_state['self'][3]
        newpos = new_game_state['self'][3]

        # get nearest coin positionb and distance to it on rotated board
        oldcoin, oldmin = get_nearest_coin_position(oldpos, old_game_state["coins"])
        newcoin, newmin = get_nearest_coin_position(newpos, new_game_state["coins"])

        # if there is a coin on the board reward the bot according to rule set
        if oldcoin != None:
            self.oldcoin = oldcoin

            # punish not lowering distance
            if newmin == oldmin:
                reward += -0.5

            #reward lowering distance
            elif newmin < oldmin:
                reward += 1

            # punish increasing distance
            elif newmin > oldmin:
                reward += -1 

            # if reward < 0:
            #     reward = 40* reward

            # punish visiting tile twice (to try and minimize getting stuck between two tiles)
            if self.preoldpos != None:
                if newpos == self.preoldpos:
                    reward += -2

            self.preoldpos = oldpos
            reward += reward_from_events(self, events)

            self.logger.info(f"Awarded {reward}")
            self.logger.info(f"mins {oldmin}, {newmin}")
            self.logger.info(f"position {oldpos}")
            self.coin_states_actions[str(oldpos[0]) + str(oldpos[1]) + str(oldcoin[0]) + str(oldcoin[1]) + self_action] += reward



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

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    with open("coin_states_actions.pt", "wb") as file:
        pickle.dump(self.coin_states_actions, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -5,
        e.SURVIVED_ROUND: 5,
        e.INVALID_ACTION: -5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
