import numpy as np

def game_state_transformer(self, game_state):
    new_game_state = copy.deepcopy(game_state)
    own_pos = game_state['self'][3]
    new_pos = list(own_pos)
    state = ""

    for i in range(len(new_game_state['bombs'])):
        new_game_state['bombs'][i] = list(new_game_state['bombs'][i])
    
    if own_pos[0] > 8:

        new_game_state['coins'] = []

        if own_pos[1] > 8:

            new_game_state['field'] = np.rot90(new_game_state['field'])
            new_game_state['field'] = np.rot90(new_game_state['field'])

            new_game_state['explosion_map'] = np.rot90(new_game_state['explosion_map'])
            new_game_state['explosion_map'] = np.rot90(new_game_state['explosion_map'])

            new_pos[0] = self.rd_x[own_pos[1] - 1][own_pos[0] - 1]
            new_pos[1] = self.rd_y[own_pos[1] - 1][own_pos[0] - 1]

            for i in range(len(game_state['coins'])):
                new_game_state['coins'].append((self.rd_x[game_state['coins'][i][1] - 1][game_state['coins'][i][0] - 1], self.rd_y[game_state['coins'][i][1] - 1][game_state['coins'][i][0] - 1]))

            for i in range(len(new_game_state['bombs'])):
                self.logger.debug(f"rotating bombs")
                new_game_state['bombs'][i][0] = (self.rd_x[new_game_state['bombs'][i][0][1] - 1][new_game_state['bombs'][i][0][0] - 1], self.rd_y[new_game_state['bombs'][i][0][1] - 1][new_game_state['bombs'][i][0][0] - 1])
            
            state = "rd"
        else:

            new_game_state['field'] = np.rot90(new_game_state['field'])
            
            

            new_game_state['explosion_map'] = np.rot90(new_game_state['explosion_map'])
            
            

            new_pos[0] = self.ru_x[own_pos[1] - 1][own_pos[0] - 1]
            new_pos[1] = self.ru_y[own_pos[1] - 1][own_pos[0] - 1]

            for i in range(len(game_state['coins'])):
                new_game_state['coins'].append((self.ru_x[game_state['coins'][i][1] - 1][game_state['coins'][i][0] - 1],self.ru_y[game_state['coins'][i][1] - 1][game_state['coins'][i][0] - 1]))

            for i in range(len(new_game_state['bombs'])):
                self.logger.debug(f"rotating bombs")
                new_game_state['bombs'][i][0] = (self.ru_x[new_game_state['bombs'][i][0][1] - 1][new_game_state['bombs'][i][0][0] - 1], self.ru_y[new_game_state['bombs'][i][0][1] - 1][new_game_state['bombs'][i][0][0] - 1])

            state = "ru"
            
            
    elif own_pos[1] > 8:
        new_game_state['coins'] = []
        
        new_game_state['field'] = np.rot90(new_game_state['field'])
        new_game_state['field'] = np.rot90(new_game_state['field'])
        new_game_state['field'] = np.rot90(new_game_state['field'])

        new_game_state['explosion_map'] = np.rot90(new_game_state['explosion_map'])
        new_game_state['explosion_map'] = np.rot90(new_game_state['explosion_map'])
        new_game_state['explosion_map'] = np.rot90(new_game_state['explosion_map'])

        new_pos[0] = self.ld_x[own_pos[1] - 1][own_pos[0] - 1]
        new_pos[1] = self.ld_y[own_pos[1] - 1][own_pos[0] - 1]

        for i in range(len(game_state['coins'])):
            new_game_state['coins'].append((self.ld_x[game_state['coins'][i][1] - 1][game_state['coins'][i][0] - 1], self.ld_y[game_state['coins'][i][1] - 1][game_state['coins'][i][0] - 1]))

        for i in range(len(new_game_state['bombs'])):
            self.logger.debug(f"rotating bombs")
            new_game_state['bombs'][i][0] = (self.ld_x[new_game_state['bombs'][i][0][1] - 1][new_game_state['bombs'][i][0][0] - 1], self.ld_y[new_game_state['bombs'][i][0][1] - 1][new_game_state['bombs'][i][0][0] - 1])

        state = "ld"

    new_game_state['self'] = list(new_game_state['self'])
    new_game_state['self'][3] = tuple(new_pos)

    return new_game_state, state