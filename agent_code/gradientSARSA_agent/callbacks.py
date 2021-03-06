import os
import pickle
import random

import numpy as np
import copy


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
        self.model = np.full(5,0.1)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


#initial guess
def q_hat(S,A,w):
    #print(w)
    #print(S['bombs'])
    S_temp=copy.deepcopy(S) ###avoid bug caused by shallow-copy (.copy())
    p = S_temp['self']
    f = S_temp['field']

    ###before applying new step, test whether step is possible
    if A=='UP':
        if f[p[3][0],p[3][1]-1]==0:
            S_temp['self']=(p[0],p[1],p[2],(p[3][0],p[3][1]-1))
    if A=='DOWN':
        if f[p[3][0],p[3][1]+1]==0:
            S_temp['self']=(p[0],p[1],p[2],(p[3][0],p[3][1]+1))
    if A=='RIGHT':
        if f[p[3][0]+1,p[3][1]]==0:
            S_temp['self']=(p[0],p[1],p[2],(p[3][0]+1,p[3][1]))
    if A=='LEFT':
        if f[p[3][0]-1,p[3][1]]==0:
            S_temp['self']=(p[0],p[1],p[2],(p[3][0]-1,p[3][1]))
    if A=='BOMB':
        if p[2]==True:
            S_temp['bombs'].append((p[3],4))
            S_temp['self']=(p[0],p[1],False,p[3]) ##avoid bug, that after dropping a bomb the agent is able to drop another
    #print(p[3], A,S_temp['self'][3])
    X=state_to_features(S_temp)
    #print(X[:7])
    print("X:",A,X)
    #print("w:",w)
    #print(S_temp['bombs'],S['bombs'])
    #print(len(X))
    assert len(w)==len(X)
    #print("q:",w@X)
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
    #print("Vorher",S['bombs'])
    tester = np.array([np.abs(q_hat(S,a,w)) for a in ACTIONS])
    #print("A1",np.array([q_hat(S,a,w) for a in ACTIONS]))
    #print("Nacher:",S['bombs'])
    if np.all(tester==tester[0]):###if all entries are equal the first entry is chosen by argmax
        greedy = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
    if np.all(tester[:4]==tester[0]) and tester[0]>tester[4] and tester[0]>tester[5]:
        greedy =np. random.choice(['RIGHT','LEFT','UP','DOWN'])
    else:
        greedy_ind = np.argmax(tester)
        greedy=ACTIONS[greedy_ind]
    
    #print(w)
    print("A",np.array([q_hat(S,a,w) for a in ACTIONS]))
    print("w", w)
    print(greedy)
    #print(tester, np.argmax(tester))
          
    # todo Exploration vs exploitation
    epsilon = 0.1
    
    if self.train:
        #self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        return np.random.choice([greedy,np.random.choice(ACTIONS,1)],1,p=[1-epsilon,epsilon])

    self.logger.debug("Querying model for action")
    
    return greedy


def in_range(bomb, player=None):
    """
    evaluates where a bomb is possibly threatening. 
    If a players and a bombs position are given it returns whether the player is in range of an exploding bomb.
    If player is not given, it returns an array with all threatend spaces.
    """
    if player != None:
        
        ###bomb range
        bomb_range=[[bomb[0],bomb[1]],]

        for i in range(1,4):
            bomb_range.append([bomb[0]+i,bomb[1]])
            bomb_range.append([bomb[0]-i,bomb[1]])
            bomb_range.append([bomb[0],bomb[1]+i])
            bomb_range.append([bomb[0],bomb[1]-i])
        #print(player,bomb_range)
        assert len(bomb_range)==13
        bombs = np.array(bomb_range)
        p = np.asarray(player)
        in_range = [np.all(b==p) for b in bombs]
        #if not np.any(np.array(in_range)==True):
        #    print(player,bomb_range,in_range)

        return np.any(np.array(in_range)==True)
    else:
        ###bomb range
        bomb_range=[[bomb[0],bomb[1]],]

        for i in range(1,4):
            bomb_range.append([bomb[0]+i,bomb[1]])
            bomb_range.append([bomb[0]-i,bomb[1]])
            bomb_range.append([bomb[0],bomb[1]+i])
            bomb_range.append([bomb[0],bomb[1]-i])
        #print(player,bomb_range)
        assert len(bomb_range)==13
        bombs = np.array(bomb_range)
        return bombs

def is_in(point, point_list, index=False):
    """
    short auxillary function to check whether a 2-dim coordinate is in a list of 2-dim coordinates.
    In default it simply gives back a boolean value.
    With index=True, it gives back a list of indices where the point was found or False if the point is not contained.
    """
    IN = False
    index = []
    for i in range(len(point_list)):
        element=point_list[i]
        if element[0]==point[0] and element[1]==point[1]:
            IN=True
            index.append(i)
    #print(IN)
    if index==False:
        return IN
    else:
        if IN==True:
            return index
        else:
            return False

def find_path(starting_point, end_point, field, maxiter=200):
    """
    Inserting two arrays with starting and end point and the underlaying field as an array.
    Gives back the amount of steps that have to be taken, False if there is no path or True if starting point equals end point.
    """
    path=[starting_point,]
    rounds = [0,]
    #print(starting_point,end_point)
    if starting_point[0]==end_point[0] and starting_point[1]==end_point[1]:
        return True
    while (not is_in([end_point[0],end_point[1]],path)) and (len(path)<maxiter and len(path)>0):
        #print(path,rounds)
        point = path[-1]
        steps = [[point[0]+1,point[1]],[point[0]-1,point[1]],[point[0],point[1]+1],[point[0],point[1]-1]]
        rank = rounds[-1]
        if rank<4:
            steplen = 1
            while len(steps)!=0:
                arr_steps = np.array(steps)
                end = np.full_like(arr_steps,end_point)
                diff = np.linalg.norm(arr_steps-end, axis=1)
                for i in range(rank): # the rank-th best step
                    
                    min_ind = np.argmin(diff)
                    diff[min_ind] = 200
                    
                min_ind = np.argmin(diff)
                best_step = steps[min_ind]
                #print(best_step)
                if field[best_step[0],best_step[1]]==0: #step is possible
                    path.append(best_step)
                    rounds.append(0)
                    steplen = len(steps)
                    steps=[]
                else:
                    steps.remove(best_step)
            #if we are trapped on a circle, remove the circle and give sign to take another step
            if is_in(path[-1],path[:-1]):
                if steplen == 1 or rounds[-1]==4:
                    remove = is_in(path[-1],path,index=True)[0]
                    del path[-2:]
                    del rounds[-2:]
                    rounds[-1]+=1
                else:
                    del path[-1]
                    del rounds[-1]
                    rounds[-1]+=1
        else:
            path=[]
        #print(path,rounds)
    
    if path==False:
        return False
    if starting_point[0]==end_point[0] and starting_point[1]==end_point[1]:
        return 0
    else:
        return len(path)



        


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
    """
    if coins== None: #if there are no coins to collect, we don't want to be confused
        nextcoin=0
    else:
        p = np.full_like(coins,player)
        nextcoin = (np.argmin(np.linalg.norm(np.asarray(player)-np.asarray(coins),axis=1)))
    """
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
    
    """
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
    """

    """
     ###coins above
    above=0
    for coin in coins:
        if coin[0]!=0:
            if coin[1]<player[1]:
                above+=1
    features.append(above/9)
     ###coins beneath
    ben=0
    for coin in coins:
        if coin[0]!=0:
            if coin[1]>player[1]:
                ben+=1
    features.append(ben/9)
     ###coins right
    right=0
    for coin in coins:
        if coin[0]!=0:
            if coin[0]>player[0]:
                right+=1
    features.append(right/9)
     ###coins left
    left=0
    for coin in coins:
        if coin[0]!=0:
            if coin[0]<player[0]:
                left+=1
    features.append(left/9)
    """

    """
    ### find directly reachable coins
    within_one=0
    for coin in coins:
        if coin[0]!=0:
            if player[0] in range(coin[0]-1,coin[0]+1) and player[1] in range(coin[1]-1,coin[1]+1):
                within_one+=1
    features.append(within_one)
    ###find next-to coins
    nextcoin=0
    for coin in coins:
        if coin[0]!=0:
            if np.linalg.norm(np.asarray(player)-np.asarray(coin))<3:
                nextcoin+=1
    features.append(nextcoin)
    """
    ###define continous potential but avoid 1/r with r->0
    dis=[]
    diag = 20 #20 is next int for diag of playboard
    """
    try:
        coins[0]
    except:
        #print(game_state['coins'])
        #print(coins)
    """
    if (coins == None) or (coins == []): #if there are no coins to collect, we don't want to be confused
        nextcoin=0
    else:
        for i in range(len(coins)):
            if coins[i][0]!=0:
                #dis.append(np.linalg.norm(np.asarray(player)-np.asarray(coins[i])))
                path_it = find_path(np.asarray(player),np.asarray(coins[i]), game_state['field'])
                
                if path_it == False:
                    dis.append(200)
                else:
                    dis.append(path_it)
            else:
                dis.append(201)
        nextcoin=min(dis)
        
    #inv_dis = diag-nextcoin ##>0

    if nextcoin==0:
        inv_dis=2
    else:
        inv_dis = 1/nextcoin #<=1 and >0
    features.append(inv_dis)
    
    ### collect how many bombs are currently safe
    #print(not in_range(player,bombs[0]))
    safe_bombs=0
    for i in range(len(bombs)):
        if bombs[i][0]!=0:
            #print(player[0],range(bombs[i][0]-3,bombs[i][0]+3),player[1],range(bombs[i][1]-3,bombs[i][1]+3))
            if not in_range(player,bombs[i]):
                safe_bombs+=1
                
    #features.append(safe_bombs)
    features.append(0)

    ###look for next safe_space
    field = np.copy(game_state['field'])

    
    closest_spot = 200
    ###max 4*13=52 iterations, probably won't take to long
    if game_state['bombs']!=[]:
        for bomb in game_state['bombs']:
            if bomb[1]<=4:
                bomb_range = in_range(bomb[0])
                for s in bomb_range:
                    if np.all(s<=16) and np.all(s>=0):
                        if field[s[0],s[1]]!=-1:
                            field[s[0],s[1]]=2

        explosions = game_state['explosion_map']
        expl = np.argwhere(explosions!=0)
        for e in expl:
            
            if field[e[0],e[1]]!=-1:
                field[e[0],e[1]]=3

        free_s = np.argwhere(field==0)
        pos = np.full_like(free_s,np.asarray(player))
        dises = np.linalg.norm(pos-free_s,axis=1)
        test_s = []
        for i in range(len(dises)):
            if dises[i]<=6:
                test_s.append(free_s[i])
        diss=[]
        max_iter=1000
        for s in test_s:
            path_iter = find_path(np.asarray(player),s, game_state['field'],max_iter)            
            if path_iter == False:
                diss.append(200)
                max_iter=200
            elif path_iter == True:
                diss.append(0)
                max_iter=0
            else:
                diss.append(path_iter)
                max_iter=path_iter
        #print(diss)
        closest_spot = min(diss)
    
    if closest_spot==0 and game_state['bombs']!=[]:
        
        inv_close=2
    elif (player,4) in game_state['bombs']:
        inv_close = 0
    elif game_state['bombs']==[]:
        
        inv_close=0
    else:
        inv_close=1/closest_spot

    features.append(inv_close)

    
    ###count destroyable crates by a dropped bomb
    
    if (player,4) in game_state['bombs']:
        #print(player,game_state['bombs'],np.transpose(game_state['field']))
        reachable_crates=0
        reachable = in_range(player)
        for r in reachable:
            if np.all(r<=16) and np.all(r>=0):
                if game_state['field'][r[0],r[1]]==1:
                    reachable_crates+=1
        #print(reachable_crates)
        #print(game_state['self'][2])
        if reachable_crates==0:
            crates=0
        else:
            crates=reachable_crates/13
    else:
        crates=0
    features.append(crates)


    ####next_crate

    field=np.copy(game_state['field'])
    crate_ind = np.argwhere(field==1)
    player = np.asarray(game_state['self'][3])
    if len(crate_ind)!=0:
        p = np.full_like(crate_ind,player)
        diff=np.linalg.norm(p-crate_ind,axis=1)
        mini = np.argmin(diff)
        #print(player,crate_ind[mini])
        cim = crate_ind[mini]
        field[cim[0],cim[1]]=0
        min_dis = find_path(player,crate_ind[mini], field)
        field[cim[0],cim[1]]=1
        #print(crate_ind[mini])
        #print(player,crate_ind,diff)
        assert min_dis != False #obviously a crate has to be reachable
        assert min_dis != True #by definition one can't stand on a crate spot. Thus diff can't be 0.
        min_dis-=1 #to get min_dis we virtually removed the crate, but now the crate is back. Therefore it is reached one step earlier.
        if field[player[0],player[1]]==0: #don't walk in danger
            invert_dis = 1/min_dis
        else:
            invert_dis = 0
        features.append(invert_dis)
    else:
        features.append(0)



    
    
    for feature in features:
        channels.append(feature)
    """
    for i in range(len(features)):
            for j in range(i):
                    channels.append(features[j]*features[i])
    channels.append(1)
    """
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    #print(len(stacked_channels))
    return stacked_channels.reshape(-1)
    
    #return states
