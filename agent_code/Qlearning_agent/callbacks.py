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


#q-function
def q_hat(S,A,w):
    """
    Estimated q-function that that maps q_hat: (States,Actions) -> R. It gives back the value of A in state S
    It is estimated by the weights w, that gives a special weight to each feature.
    The weights w are learned during Training process and determine the final policy.

    :param S: A dictionary with current game state
    :param A: A string with an action out of list ACTIONS
    :param w: An array with current weights
    :return: float
    """
    
    S_temp=copy.deepcopy(S) ###avoid bug caused by shallow-copy (.copy())
    p = S_temp['self']
    f = S_temp['field']

    ###before applying new step, test whether step is possible (field-entry equals zero)
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
        if p[2]==True:#test if bombing is possible
            S_temp['bombs'].append((p[3],4))
            S_temp['self']=(p[0],p[1],False,p[3]) ##avoid bug, that after dropping a bomb the agent is able to drop another
    
    X=state_to_features(S_temp)
    
    #print("X:",A,X)
    #print("w:",w)

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
    
    S=game_state
    action = range(1,len(ACTIONS)+1)
    
    w=self.model
    
    ###find greedy action (maximizes q_hat for given S and w)
    tester = np.array([np.abs(q_hat(S,a,w)) for a in ACTIONS])
    if np.all(tester==tester[0]):###if all entries are equal the first entry is chosen by argmax
        greedy = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
    #if all step-entries have equal value and value of bomb is less, do random step
    if np.all(tester[:4]==tester[0]) and tester[0]>tester[4] and tester[0]>tester[5]: 
        greedy =np. random.choice(['RIGHT','LEFT','UP','DOWN'])
    else:
        greedy_ind = np.argmax(tester)
        greedy=ACTIONS[greedy_ind]
    
    #print("A",np.array([q_hat(S,a,w) for a in ACTIONS]))
    #print("w", w)
    #print(greedy)
          
    #Exploration vs exploitation
    # definie hyperparameter for epsilon-greedy-policy (lecture 2, p.3)
    epsilon = 0.1
    
    if self.train:
        #in training-mode use epsilon-greedy-policy
        return np.random.choice([greedy,np.random.choice(ACTIONS,1)],1,p=[1-epsilon,epsilon])

    self.logger.debug("Querying model for action")
    #in game-mode take greedy action
    return greedy

#TODO: not sure if there is a bug that explosions are reaching through walls in this function, might be good to check someday
def in_range(bomb, player=None):
    """
    evaluates where a bomb is possibly threatening. 
    If a players and a bombs position are given it returns whether the player is in range of an exploding bomb.
    If player is not given, it returns an array with all threatend spaces.

    :param bomb: list, array or tuple with 2D-coordinates of the bomb
    :param player (optional): list, array or tuple with 2D-coordinates of the agent

    :return: bool or list, depending on arguments handed in
    """
    if player != None:
        
        ###initialize list of coordinates that are in bomb range
        bomb_range=[[bomb[0],bomb[1]],]

        for i in range(1,4):
            bomb_range.append([bomb[0]+i,bomb[1]])
            bomb_range.append([bomb[0]-i,bomb[1]])
            bomb_range.append([bomb[0],bomb[1]+i])
            bomb_range.append([bomb[0],bomb[1]-i])
        
        assert len(bomb_range)==13 

        bombs = np.array(bomb_range)
        p = np.asarray(player)
        in_range = [np.all(b==p) for b in bombs]

        return np.any(np.array(in_range)==True)
    else:
        ###initialize list of coordinates that are in bomb range
        bomb_range=[[bomb[0],bomb[1]],]

        for i in range(1,4):
            bomb_range.append([bomb[0]+i,bomb[1]])
            bomb_range.append([bomb[0]-i,bomb[1]])
            bomb_range.append([bomb[0],bomb[1]+i])
            bomb_range.append([bomb[0],bomb[1]-i])
        
        assert len(bomb_range)==13

        bombs = np.array(bomb_range)

        return bombs


def is_in(point, point_list, index=False):
    """
    short auxillary function to check whether a 2-dim coordinate is in a list of 2-dim coordinates.
    In default it simply gives back a boolean value.
    With index=True, it gives back a list of indices where the point was found or False if the point is not contained.

    :param point: list, array or tuple with 2D-coordinates of one point
    :param point_list: list with 2D-coordinates
    :param index (optional): boolean if list of indices where a matching entry was found should be returned

    :return: bool or list
    """
    IN = False
    index = []
    
    #check if a matching point can be found
    for i in range(len(point_list)):
        element=point_list[i]
        if element[0]==point[0] and element[1]==point[1]:
            IN=True
            index.append(i)
    
    if index==False:
        return IN
    else:
        if IN==True:
            return index
        else:
            return False

#TODO speed that function up, maybe there is a special algorithm that fits better for that challenge @Laurin
def find_path(starting_point, end_point, field, maxiter=200):
    """
    Inserting starting and end point and the underlaying field as an array.
    Gives back the amount of steps that have to be taken, False if there is no path or True if starting point equals end point.

    :param starting_point: list, array or tuple with 2D-coordinates of the starting point
    :param end_point: list, array or tuple with 2D-coordinates of the end point
    :param field: array of the underlaying field
    :param maxiter (optional): int with maximum of iterations

    :return: the amount of steps that have to be taken as int
    """

    #initialize the list of coordinates that build the path, and auxilliary list to check how often path has crossed a point
    path=[starting_point,]
    rounds = [0,]
    
    if starting_point[0]==end_point[0] and starting_point[1]==end_point[1]: #catch that easy case before looping
        return True

    ##start loop to find a path
    while (not is_in([end_point[0],end_point[1]],path)) and (len(path)<maxiter and len(path)>0):
        
        point = path[-1]
        #initialize a step in every direction
        steps = [[point[0]+1,point[1]],[point[0]-1,point[1]],[point[0],point[1]+1],[point[0],point[1]-1]]
        rank = rounds[-1]

        if rank<4: #if we didn't already tried every direction (catch loops caused by nonreachable end points)
            steplen = 1
            while len(steps)!=0: #loop until there are no more steps to try
                arr_steps = np.array(steps)
                end = np.full_like(arr_steps,end_point)
                diff = np.linalg.norm(arr_steps-end, axis=1)

                #in a previous try we tried the best step that led to a uncomplete path, now take the next best step
                for i in range(rank): # the rank-th best step
                    
                    min_ind = np.argmin(diff)
                    diff[min_ind] = 200 #don't delete the entry not to get problems with indices
                    
                min_ind = np.argmin(diff)
                best_step = steps[min_ind] #choose the best step out of all steps that are left
                
                if field[best_step[0],best_step[1]]==0: #check whether step is possible
                    path.append(best_step)
                    rounds.append(0)
                    steplen = len(steps)
                    steps=[] #can forget the remaining options
                else:
                    steps.remove(best_step) #remove that option and try with another

            #if we are trapped on a circle, remove the circle and give sign to take another step
            if is_in(path[-1],path[:-1]): #if this is true we are crossing our path somewhere ->circle

                if steplen == 1 or rounds[-1]==4: #if this is true we are trapped
                    #go back two steps
                    del path[-2:] 
                    del rounds[-2:]
                    rounds[-1]+=1
                else:
                    #go back one step
                    del path[-1]
                    del rounds[-1]
                    rounds[-1]+=1
        else:
            path=[]
        
    
    if path==False:
        return False
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

    features = []
    channels = []

    #initialize some lists
    coins=game_state['coins']
    player=game_state['self'][3]
    bombs = [b[0] for b in game_state['bombs']]

    l=4-len(bombs)
    for i in range(l): #get full list, with (0,0) as placeholder
        bombs.append((0,0))
    
    l2=9-len(coins)
    if l2!=9:
        for i in range(l2):
            coins.append((0,0))
    

    ###Feature 1: Define continous potential but avoid 1/r with r->0
    dis=[]
    
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

    if nextcoin==0:
        inv_dis=2
    else:
        inv_dis = 1/nextcoin #<=1 and >0
    features.append(inv_dis)
    

    ###Remark: I will remove that soon
    features.append(0)


    ###Feature 2: look for next safe_space

    field = np.copy(game_state['field'])

    closest_spot = 200

    ###max 4*13=52 iterations, probably won't take to long
    
    if game_state['bombs']!=[]: #only if bombs are active on the field

        #determine all spaces that are currently threatend by a bomb and insert entry 2 in our copy
        for bomb in game_state['bombs']:
            if bomb[1]<=4:
                bomb_range = in_range(bomb[0])
                for s in bomb_range:
                    if np.all(s<=16) and np.all(s>=0):
                        if field[s[0],s[1]]!=-1:
                            field[s[0],s[1]]=2

        #determine all spaces that are threatend by a remaining explosion and insert entry 3 in our copy
        explosions = game_state['explosion_map']
        expl = np.argwhere(explosions!=0)
        for e in expl:
            
            if field[e[0],e[1]]!=-1:
                field[e[0],e[1]]=3

        #now all spaces with entry 0 are safe spaces for the current state
        free_s = np.argwhere(field==0)
        pos = np.full_like(free_s,np.asarray(player))
        dises = np.linalg.norm(pos-free_s,axis=1)
        test_s = []

        #just consider near spaces (otherwise it would take to long)
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
            elif path_iter == True: #if we are already safe, we don't have to check the rest
                diss.append(0)
                max_iter=0
            else:
                diss.append(path_iter) 
                max_iter=path_iter #we are only interessted in better choice, therefore we can speed calculation up
        
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

    
    ###Feature 3: Count destroyable crates by a dropped bomb
    
    if (player,4) in game_state['bombs']: #only active if a bomb is dropped by our agent

        reachable_crates=0
        reachable = in_range(player)
        
        for r in reachable:
            if np.all(r<=16) and np.all(r>=0):
                if game_state['field'][r[0],r[1]]==1:
                    reachable_crates+=1
        
        if reachable_crates==0:
            crates=0
        else:
            crates=reachable_crates/13
    else:
        crates=0

    features.append(crates)


    ###Feature 4: Find next crate
    #TODO: not only add the euclidian next, but a handfull of near crates to the feature spaces

    field=np.copy(game_state['field'])
    crate_ind = np.argwhere(field==1) #find all crates on the field
    player = np.asarray(game_state['self'][3])

    if len(crate_ind)!=0: #if there are no crates we can't find any distances

        p = np.full_like(crate_ind,player)
        diff=np.linalg.norm(p-crate_ind,axis=1)
        mini = np.argmin(diff)
        cim = crate_ind[mini]

        #we need to virtually remove the crate to make the find_path function work
        field[cim[0],cim[1]]=0
        min_dis = find_path(player,crate_ind[mini], field) #safe computation time by only calulating the euclidian closest 
        field[cim[0],cim[1]]=1

        assert min_dis != True #by definition one can't stand on a crate spot. Thus diff can't be 0.

        if min_dis==False: #the euclidian next crate is not reachable, therefore we ignore him this round
            invert_dis = 0

        elif field[player[0],player[1]]==0: #don't walk in danger

            min_dis-=1 #to get min_dis we virtually removed the crate, but now the crate is back. Therefore it is reached one step earlier.

            invert_dis = 1/min_dis

        else:
            invert_dis = 0
        features.append(invert_dis)

    else:
        features.append(0)

    #I kept this distinction between features and channels if a feature augmentation would be neccessary at some point (Sutton, 9.5)
    for feature in features:
        channels.append(feature)
    
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    
    return stacked_channels.reshape(-1)
    
    
