import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features

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

    # try:
    #     self.coin_states = pickle.load(open("coin_states.pt", "rb"))
    # except:
    #     self.coin_states = None
    #     self.logger.debug(f"coin_states could not be loaded")

    try:
        self.coin_states_actions = pickle.load(open("coin_states_actions.pt", "rb"))
    except:
        self.coin_states_actions = None
        self.logger.debug(f"coin_states_actions could not be loaded")


    # if self.coin_states == None:
    #     self.coin_states = {}
    #     for i in range(1, 8):
    #         for j in range(1, 8):
    #             for k in range(1, 15):
    #                 for l in range(1, 15):
    #                     self.coin_states[str(i) + str(j) + str(k) + str(l)] = np.sqrt(np.square(i - k) + np.square(j - k))

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

    # # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # # state_to_features is defined in callbacks.py
    # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    if self_action in ["LEFT", "RIGHT", "UP", "DOWN"]:
        ownpos = game_state_transformer(self, old_game_state)[0]['self'][3]
        
        oldcoin, oldmin = get_nearest_coin_position(ownpos, old_game_state["coins"])
        newmin = np.linalg.norm(np.array(oldcoin) - np.array(new_game_state['self'][3]))


        if newmin == oldmin:
            reward = 0
        else:
            reward = (oldmin - newmin)
        reward += reward_from_events(self, events)

        self.logger.info(f"Awarded {reward}")

    if self_action == "WAIT":
        if self.last_a != None:
            self.coin_states_actions[str(ownpos[0]) + str(ownpos[1]) + str(oldcoin[0]) + str(oldcoin[1]) + self_action] += -5


        self.coin_states_actions[str(ownpos[0]) + str(ownpos[1]) + str(oldcoin[0]) + str(oldcoin[1]) + self_action] += reward

    

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

    # with open("coin_states.pt", "wb") as file:
    #     pickle.dump(self.coin_states, file)

    with open("coin_states_actions.pt", "wb") as file:
        pickle.dump(self.coin_states_actions, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1000,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -5,
        e.SURVIVED_ROUND: 5,
        #e.KILLED_OPPONENT: 5,
        #PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

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
