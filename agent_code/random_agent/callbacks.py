import numpy as np
from agent_funcs import *



def setup(self):
    np.random.seed()


def act(agent, game_state: dict):

    bombstate = find_threats(self, game_state)
    self.logger.debug(f"{bombstate}")

    agent.logger.info('Pick action at random')
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
