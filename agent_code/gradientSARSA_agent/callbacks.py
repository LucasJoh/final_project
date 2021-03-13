import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(10)
        self.model = np.full(37,0.1)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


#initial guess
def q_hat(S,A,w):
    #print(w)
    S_temp=S.copy()
    p = S_temp['self']
    ###before applying new step, test whether step is possible
    if p[3][0]%2==1:
        if A=='UP':
            if p[3][1]>=1:
                S_temp['self']=(p[0],p[1],p[2],(p[3][0],p[3][1]-1))
        if A=='DOWN':
            if p[3][1]<=14:
                S_temp['self']=(p[0],p[1],p[2],(p[3][0],p[3][1]+1))
    if p[3][1]%2==1:
        if A=='RIGHT':
            if p[3][0]<=14:
                S_temp['self']=(p[0],p[1],p[2],(p[3][0]+1,p[3][1]))
        if A=='LEFT':
            if p[3][0]>=2:
                S_temp['self']=(p[0],p[1],p[2],(p[3][0]-1,p[3][1]))
    if A=='BOMB':
        S_temp['bombs'].append((p[3],4))
    #print(p[3], A,S_temp['self'][3])
    X=state_to_features(S_temp)
    #print(X[:7])
    #print(X)
    #print(S_temp,S)
    #print(len(X))
    assert len(w)==len(X)
    return w@X


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    

    #S= state_to_features(game_state)
    S=game_state
    action = range(1,len(ACTIONS)+1)
    """
    if len(self.model)!=len(S):
        
        w=np.concatenate((np.array(self.model),np.full(np.abs(len(S)-len(self.model)),0.0)))
    else:
    """
    w=self.model
    
    #assert len(S)==len(w)
    greedy_ind = np.argmax(np.array([q_hat(S,a,w) for a in ACTIONS]))
    greedy=ACTIONS[greedy_ind]
    l=[q_hat(S,a,w) for a in ACTIONS]
    #print(w)
    
          
    # todo Exploration vs exploitation
    epsilon = 0.01
    
    if self.train:
        #self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        return np.random.choice([greedy,np.random.choice(ACTIONS,1)],1,p=[1-epsilon,epsilon])

    self.logger.debug("Querying model for action")
    
    return greedy


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    #Work with polynomial Basis (as seen in 9.5.1, p.210f.) (LS)

    # For example, you could construct several channels of equal shape, ...
    features = []
    channels = []
    #for collecting coins we consider our own position and the position of coins
    
    coins=game_state['coins']
    player=game_state['self'][3]
    bombs = [b[0] for b in game_state['bombs']]
    l=4-len(bombs)
    for i in range(l): #get full list, with (0,0) as placeholder
        bombs.append((0,0))
    
    if coins== None: #if there are no coins to collect, we don't want to be confused
        nextcoin=0
    else:
        p = np.full_like(coins,player[3])
        nextcoin = np.argmin(np.linalg.norm(p-coins,axis=1))
    
    #
    l2=9-len(coins)
    if l2!=9:
        for i in range(l2):
            coins.append((0,0))
    """
    channels.append(player[3])
    for coin in coins:
        channels.append(coin)
    #channels.append(nextcoin)
    for bomb in bombs:
        channels.append(bomb)
    """
    ###define 3 neighborhoods of the player. To maximize the chance for a coin it is helpful to maximize the number of coins inside the nh.
    ### find directly reachable coins
    within_one=0
    for coin in coins:
        if coin[0]!=0:
            if player[0] in range(coin[0]-1,coin[0]+1) and player[1] in range(coin[1]-1,coin[1]+1):
                within_one+=1
    features.append(within_one)
    ###find close coins
    nearby=0
    for coin in coins:
        if coin[0]!=0:
            if np.linalg.norm(np.asarray(player)-np.asarray(coin))<=3:
                nearby+=1
    features.append(nearby)
    ###find somehow reachable coins
    reachable=0
    for coin in coins:
        if coin[0]!=0:
            if np.linalg.norm(np.asarray(player)-np.asarray(coin))<=6:
                reachable+=1
    features.append(reachable)
     ###coins above
    above=0
    for coin in coins:
        if coin[0]!=0:
            if coin[1]>player[1]:
                above+=1
    features.append(above)
     ###coins beneath
    ben=0
    for coin in coins:
        if coin[0]!=0:
            if coin[1]<player[1]:
                ben+=1
    features.append(ben)
     ###coins right
    right=0
    for coin in coins:
        if coin[0]!=0:
            if coin[0]>player[0]:
                right+=1
    features.append(right)
     ###coins left
    left=0
    for coin in coins:
        if coin[0]!=0:
            if coin[0]<player[0]:
                left+=1
    features.append(left)

    features.append(nextcoin)
    
    ### collect how many bombs are currently safe
    
    safe_bombs=0
    for i in range(len(bombs)):
        if bombs[i][0]!=0:
            if (player[0] not in range(bombs[i][0]-3,bombs[i][0]+3)) and (player[1] not in range(bombs[i][1]-3,bombs[i][1]+3)):
                safe_bombs+=1
    features.append(safe_bombs)
    
    for feature in features:
        channels.append(feature)
    for i in range(len(features)):
            for j in range(i):
                    channels.append(features[j]*features[i])
    channels.append(1)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    #print(len(stacked_channels))
    return stacked_channels.reshape(-1)
    
    #return states
