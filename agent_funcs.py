import numpy as np


#turn board
def game_state_transformer(self, game_state):
    new_game_state = game_state
    own_pos = game_state['self'][3]
    new_pos = list(own_pos)
    state = ""

    for i in range(len(new_game_state['bombs'])):
        new_game_state['bombs'][i] = list(new_game_state['bombs'][i])

    for i in range(len(new_game_state['coins'])):
        new_game_state['coins'][i] = list(new_game_state['coins'][i])
 
    if own_pos[0] > 8:

        if own_pos[1] > 8:

            new_game_state['field'] = np.array(list(zip(*new_game_state['field']))[::-1])
            new_game_state['field'] = np.array(list(zip(*new_game_state['field']))[::-1])

            new_game_state['explosion_map'] = np.array(list(zip(*new_game_state['explosion_map']))[::-1])
            new_game_state['explosion_map'] = np.array(list(zip(*new_game_state['explosion_map']))[::-1])

            new_pos[0] = self.rd_x[own_pos[1] - 1][own_pos[0] - 1]
            new_pos[1] = self.rd_y[own_pos[1] - 1][own_pos[0] - 1]

            for i in range(len(new_game_state['coins'])):
                new_game_state['coins'][i][0] = self.rd_x[new_game_state['coins'][i][0] - 1][new_game_state['coins'][i][1] - 1]
                new_game_state['coins'][i][1] = self.rd_y[new_game_state['coins'][i][0] - 1][new_game_state['coins'][i][1] - 1]

            for i in range(len(new_game_state['bombs'])):
                new_game_state['bombs'][i][0][0] = self.rd_x[new_game_state['bombs'][i][0][0] - 1][new_game_state['bombs'][i][0][1] - 1]
                new_game_state['bombs'][i][0][1] = self.rd_y[new_game_state['bombs'][i][0][0] - 1][new_game_state['bombs'][i][0][1] - 1]
            
            state = "rd"
        else:

            new_game_state['field'] = np.array(list(zip(*new_game_state['field']))[::-1])
            

            new_game_state['explosion_map'] = np.array(list(zip(*new_game_state['explosion_map']))[::-1])
            
            new_pos[0] = self.ru_x[own_pos[1] - 1][own_pos[0] - 1]
            new_pos[1] = self.ru_y[own_pos[1] - 1][own_pos[0] - 1]

            for i in range(len(new_game_state['coins'])):
                new_game_state['coins'][i][0] = self.ru_x[new_game_state['coins'][i][0] - 1][new_game_state['coins'][i][1] - 1]
                new_game_state['coins'][i][1] = self.ru_y[new_game_state['coins'][i][0] - 1][new_game_state['coins'][i][1] - 1]

            for i in range(len(new_game_state['bombs'])):
                new_game_state['bombs'][i][0][0] = self.ru_x[new_game_state['bombs'][i][0][0] - 1][new_game_state['bombs'][i][0][1] - 1]
                new_game_state['bombs'][i][0][1] = self.ru_y[new_game_state['bombs'][i][0][0] - 1][new_game_state['bombs'][i][0][1] - 1]

            state = "ru"
            
            
    elif own_pos[1] > 8:

        new_game_state['field'] = np.array(list(zip(*new_game_state['field']))[::-1])
        new_game_state['field'] = np.array(list(zip(*new_game_state['field']))[::-1])
        new_game_state['field'] = np.array(list(zip(*new_game_state['field']))[::-1])

        new_game_state['explosion_map'] = np.array(list(zip(*new_game_state['explosion_map']))[::-1])
        new_game_state['explosion_map'] = np.array(list(zip(*new_game_state['explosion_map']))[::-1])
        new_game_state['explosion_map'] = np.array(list(zip(*new_game_state['explosion_map']))[::-1])

        new_pos[0] = self.ld_x[own_pos[1] - 1][own_pos[0] - 1]
        new_pos[1] = self.ld_y[own_pos[1] - 1][own_pos[0] - 1]

        for i in range(len(new_game_state['coins'])):
            new_game_state['coins'][i][0] = self.ld_x[new_game_state['coins'][i][0] - 1][new_game_state['coins'][i][1] - 1]
            new_game_state['coins'][i][1] = self.ld_y[new_game_state['coins'][i][0] - 1][new_game_state['coins'][i][1] - 1]

        for i in range(len(new_game_state['bombs'])):
            new_game_state['bombs'][i][0][0] = self.ld_x[new_game_state['bombs'][i][0][0] - 1][new_game_state['bombs'][i][0][1] - 1]
            new_game_state['bombs'][i][0][1] = self.ld_y[new_game_state['bombs'][i][0][0] - 1][new_game_state['bombs'][i][0][1] - 1]

        state = "ld"

    new_game_state['self'] = list(new_game_state['self'])
    new_game_state['self'][3] = tuple(new_pos)

    return new_game_state, state

    # gets position of nearest coin
def get_nearest_coin_position(own_pos, coin_pos):
    min = 1000
    coin = (0,0)

    for c in coin_pos:
        dist = abs(c[0] - own_pos[0]) + abs(c[1] - own_pos[1])

        if dist < min:
            min = dist
            coin = c
    if coin == (0,0):
        return None, None
    else:
        return coin, min

def setup_coords(self):
    # coordinate setup
    coords_x = np.array([[i for i in range(1, 16)] for j in range(1,16)])
    coords_y = np.array([[j for i in range(1, 16)] for j in range(1,16)])
    
    self.ld_x = np.rot90(coords_x, k=1, axes=(0, 1))
    self.ld_y = np.rot90(coords_y, k=1, axes=(0, 1))

    self.rd_x = np.rot90(self.ld_x)
    self.rd_y = np.rot90(self.ld_y)

    
    self.ru_x = np.rot90(self.rd_x)
    self.ru_y = np.rot90(self.rd_y)


    self.order_ru = {"LEFT": "UP", "RIGHT": "DOWN", "UP": "RIGHT", "DOWN": "LEFT"}
    self.order_rd = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
    self.order_ld = {"LEFT": "DOWN", "UP": "LEFT", "RIGHT": "UP", "DOWN": "RIGHT"}

# def find_bombs(self, game_state):
#     if game_state["bombs"] != []:
#         for bomb in game_state["bombs"]:
