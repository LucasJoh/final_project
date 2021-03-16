import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features, q_hat

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
THREATEN_BY_ONE = "THREATEN_BY_ONE"
GETTING_CLOSER = "GETTING_CLOSER"
NEXT_TO_COIN = "NEXT_TO_COIN"
LOOSING_COIN = "LOOSING_COIN"
GOOD_STEP = "GOOD_STEP"
SAFE_BOMB = "SAFE_BOMB"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
WELL_PLACED_BOMB = "WELL_PLACED_BOMB"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    

def grad_q_hat(S,A,w):
    grad=np.zeros_like(w)
    for i in range(len(w)):#use that q is just linear wrt w
        w_temp=np.zeros_like(w)
        w_temp[i]=1
        grad[i]= q_hat(S,A,w_temp)
    return grad


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

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    if old_game_state == None:
        pass
    else:
        #S=state_to_features(old_game_state)
        S=old_game_state
        #S_prime = state_to_features(new_game_state)
        S_prime = new_game_state
        #A=np.where(self_action==ACTIONS)[0]
        A=self_action
        old_features = state_to_features(S)
        new_features = state_to_features(S_prime)
        
        #threat=0
        #new events
        #print(S_prime[0:2],S_prime[20:28])
        """
        for i in range(10,14):
            
            if S_prime[i*2]!=0: #don't get threaten by nonexisting bombs
                
                if (S_prime[0] in range(S_prime[i*2]-3,S_prime[i*2]+3)) or (S_prime[1] in range(S_prime[(i*2)+1]-3,S_prime[(i*2)+1]+3)):
                    events.append(THREATEN_BY_ONE)
        """
        #print(new_features[:6])
        for i in range(int(new_features[1])):
            events.append(SAFE_BOMB)
        #if threat==1:
        #    events.append(THREATEN_BY_ONE)

        #like a multiplier for getting closer to coins
        """
        old_p = S[0:2]
        new_p =S_prime[:2]
        old_cs = S[2:20]
        new_cs = S_prime[2:20]
        """
        """
        for i in range(9):
            old_c = old_cs[i*2:(i*2)+2]
            new_c = new_cs[i*2:(i*2)+2]
            if old_c[0]!=0 or new_c[0]!=0: #if coins appear or disappear its useless to talk about distance
                if np.linalg.norm(old_p-old_c)>np.linalg.norm(new_p-new_c):
                    events.append(GETTING_CLOSER)
        """
        """
        ind=1
        old_c = old_cs[ind*2:(ind*2)+2]
        new_c = new_cs[ind*2:(ind*2)+2]
        if old_c[0]!=0 or new_c[0]!=0: #if coins appear or disappear its useless to talk about distance
            if np.linalg.norm(old_p-old_c)>np.linalg.norm(new_p-new_c):
                events.append(GETTING_CLOSER)
        """
        if new_features[0]>old_features[0]:
            events.append(GETTING_CLOSER)
            
            

        #if new_features[0]>old_features[0]:
        #    events.append(NEXT_TO_COIN)

        #if agent takes a step away of available coins or an enemy is faster
        #by reward it is guranteed that the reward for taking the coin and therefore out of range is higher than the loss for not taking it
        if (new_features[0]<old_features[0]):
            events.append(LOOSING_COIN)

        if old_features[3]>=1 and self_action=="BOMB":
            events.append(WELL_PLACED_BOMB)

        #agents moves towards the right direction?
        """
        if e.INVALID_ACTION not in events:
            if A=="LEFT" and new_features[3]>new_features[2]:
                events.append(GOOD_STEP)
            if A=="RIGHT" and new_features[3]<new_features[2]:
                events.append(GOOD_STEP)
            if A=="UP" and new_features[0]>new_features[1]:
                events.append(GOOD_STEP)
            if A=="DOWN" and new_features[0]<new_features[1]:
                events.append(GOOD_STEP)
            #if new_features[7]>old_features[7]:
            #    events.append(CLOSER_TO_COIN)
            #if new_features[7]<old_features[7]:
            #    events.append(LOOSING_COIN)
        """

        gamma=0.7
        alpha=0.2
        
        """
        if (self.model).all() ==None:
            w=np.full(0.0,len(S))
        elif len(self.model)!=len(S):
            
            w=np.concatenate((np.array(self.model),np.full(np.abs(len(S)-len(self.model)),0.0)))
        else:
        """
        w=self.model
        
        action=range(1,len(ACTIONS)+1)
        epsilon=0.05
        tester = np.array([np.abs(q_hat(S,a,w)) for a in ACTIONS])
        if np.all(tester==tester[0]):###if all entries are equal the first entry is chosen by argmax
            greedy = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
        else:
            greedy_ind = np.argmax(tester)
            greedy=ACTIONS[greedy_ind]
        #A_prime=np.random.choice((greedy,np.random.choice(ACTIONS,1)),1,[1-epsilon,epsilon])###SARSA
        A_prime = greedy ###Q-learning
        
        
        #use gradient SARSA (p.244)
        #print(f"Grad: {grad_q_hat(S,A,w)}")
        #print(q_hat(S_prime,A_prime,w))
        #print(q_hat(S,A,w))
        #print(reward_from_events(self,events))
        #print(new_features[0:9])
        #print(0,q_hat(S_prime,A_prime,w) ,q_hat(S,A,w))
        #print(1,gamma*q_hat(S_prime,A_prime,w)-q_hat(S,A,w))
        #print(alpha*(reward_from_events(self,events)-gamma*q_hat(S_prime,A_prime,w)-q_hat(S,A,w)))
        #print(1,w)
        #print(2,alpha*(reward_from_events(self,events)-gamma*q_hat(S_prime,A_prime,w)-q_hat(S,A,w))*grad_q_hat(S,A,w))


        #print(w)
        #gradient method
        #print(state_to_features(S_prime)[:6],grad_q_hat(S,A,w)[0:6])
        ####Iteration
        R=reward_from_events(self,events)
        w=w+alpha*(R+gamma*q_hat(S_prime,A_prime,w)-q_hat(S,A,w))*grad_q_hat(S,A,w)
        #w=w+alpha*(R)*grad_q_hat(S,A,w)
        #print(2,R)
        #print(3,w)

        self.model=w

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

    #S=state_to_features(last_game_state)
    S=last_game_state   
        
    #A=np.where(last_action==ACTIONS)[0]+1
    A=last_action
        
    gamma=0.8
    alpha=0.4

    """    
    if (self.model).all() ==None:
        w=np.full(0.0,len(S))
    elif len(self.model)!=len(S):
            
        w=np.concatenate((np.array(self.model),np.full(np.abs(len(S)-len(self.model)),0.0)))
    else:
    """
    w=self.model
        
    action=range(1,len(ACTIONS)+1)
    epsilon=0.1
    greedy_ind = np.argmax(np.array([q_hat(S,a,w) for a in ACTIONS]))
    
    
    #iteration
    w=w+alpha*(reward_from_events(self,events)-gamma*q_hat(S,A,w)-q_hat(S,A,w))*grad_q_hat(S,A,w)

    self.model=w

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        #e.KILLED_OPPONENT: 5,
        e.MOVED_DOWN: -.1,
        e.MOVED_LEFT: -.1,
        e.MOVED_RIGHT: -.1,
        e.MOVED_UP: -.1,
        e.WAITED: -.1,
        e.INVALID_ACTION: -1,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -5,
        e.CRATE_DESTROYED: 0.2,
        THREATEN_BY_ONE: -0.1,
        GETTING_CLOSER: 3,
        CLOSER_TO_COIN: 5,
        NEXT_TO_COIN: 10,
        LOOSING_COIN: -2,
        GOOD_STEP: 0.3,
        SAFE_BOMB: 0.5,
        WELL_PLACED_BOMB: 1,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
