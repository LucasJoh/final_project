from agent_funcs import *

def setup(self):
    setup_coords(self)


def act(self, game_state: dict):
    
    new_game_state, state= game_state_transformer(self, game_state)
    ownpos = new_game_state["self"][3]
    coin, min = get_nearest_coin_position(ownpos, new_game_state["coins"])
    bombstate = threat_transformer(self, new_game_state)
    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']
