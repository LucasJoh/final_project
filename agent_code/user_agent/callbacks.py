from agent_funcs import *

def setup(self):
    setup_coords(self)


def act(self, game_state: dict):
    
    self.logger.debug(f"{game_state['coins']}")
    new_game_state, state= game_state_transformer(self, game_state)
    ownpos = new_game_state["self"][3]
    self.logger.debug(f"{game_state['coins']}")
    self.logger.debug(f"{new_game_state['coins']}")
    coin, min = get_nearest_coin_position(ownpos, new_game_state["coins"])
    self.logger.debug(f"{coin}, {min}, {ownpos}")
    bombstate = threat_transformer(self, new_game_state)
    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']
