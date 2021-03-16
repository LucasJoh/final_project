from agent_funcs import *

def setup(self):
    setup_coords(self)


def act(self, game_state: dict):
    
    new_game_state, state= game_state_transformer(self, game_state)
    bombstate = threat_transformer(self, new_game_state)
    self.logger.debug(f"{game_state}")
    self.logger.info(f"{bombstate}")
    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']
