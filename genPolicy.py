
# coding: utf-8

# In[288]:

import numpy as np

bases = np.array([20, 20, 20, 5, 5, 5, 5, 1])
mods = [np.prod(bases[idx:]) for idx in range(len(bases))]

def idxToState(idx):
    state = np.zeros((len(bases)-1, 1))
    for i in range(len(mods)-2,-1,-1):
        state[i] = (idx % mods[i]) / mods[i+1]
        idx -= (idx % mods[i])
    return state
    
def stateToIdx(state):
    idx = 0
    for i, el in enumerate(state):
        toAdd = np.prod(bases[i+1:]) * el
        idx += toAdd
    return int(idx)


# In[289]:

def actionToIdx(action):
    if max(action) == 0 and min(action) == 0: 
        return 0
    return int((action[0] + 1)/2 + (2 * action[1]) + 1)

def idxToAction(idx):
    if idx == 0: 
        return (0,0)
    idx -= 1
    return ((idx%2)*2 - 1, idx/2)


# In[290]:

archive = np.load('all_A_B_data.npz')


# In[291]:

rawRewards = np.copy(archive['rewards'])
rawActions = np.copy(archive['actions'])
rawStates = np.copy(archive['discreteStates'])
binScales = np.copy(archive['binScales'])
filterBooleans = np.copy(archive['filtered'])


# In[292]:

print rawRewards.shape
print rawStates.shape
print rawActions.shape
print filterBooleans.shape


# In[296]:

states = [stateToIdx(state) for state in rawStates]
actions = [actionToIdx(action) for action in rawActions]
rewards = {states[t]:rawRewards[t] for t in range(len(states))}


# In[297]:

from collections import defaultdict
import copy
import sys
from scipy import stats
from scipy import sparse
import scipy.sparse as sp
import scipy.sparse.linalg


# In[330]:

def getTransitions():
    observed_states = set()

    valid_actions = defaultdict(list)
    
    transitions = defaultdict(lambda: defaultdict(int))
    for t in range(len(states)-1):
        if filterBooleans[t] and filterBooleans[t+1]:
            s = states[t]
            a = actions[t]
            sp = states[t+1]
            transitions[(s, a)][sp] += 1
            observed_states.add(s)
            
            valid_actions[s].append(a)

    for sa in transitions:
        denom = sum(transitions[sa].values()) * 1.0
        transitions[sa] = {x: transitions[sa][x]/denom for x in transitions[sa]}

    return transitions, list(observed_states), valid_actions


# In[331]:

transitions, observed_states, valid_actions = getTransitions()


# In[332]:

def valueIteration(gamma, transitions, observed_states, valid_actions, numStates, U=None, k=0):
    if U is None:
        U = defaultdict(int)
    initial_k = k
    while True:
        k += 1
        old_U = copy.copy(U)
        for s in observed_states:
            poss_rewards = [(rewards[s] + gamma * sum([transitions[(s, a)][sa] * float(old_U[sa])                                     for sa in transitions[(s, a)]])) for a in valid_actions[s]]
            U[s] = max(poss_rewards)
        diff = np.sqrt(np.sum(np.square(np.array([old_U[x]-U[x] for x in U]))))
        print 'Iteration {} has diff {}'.format(k, diff)

        if k % 100 == 0:
            print "Saving U at iter {}".format(k)
            np.save('checkpoint_value_iter_U_{}'.format(k), U)

        if diff < 0.1 or k > (initial_k+1000):
            break
    print str(k) + ' iterations required for convergence (or we just did too many)'
    return U


# In[333]:

NUM_STATES = (20**3) * (5**4)
U = valueIteration(.99, transitions, observed_states, valid_actions, NUM_STATES)


# In[334]:

U


# In[335]:

def extractPolicy(U, gamma, valid_actions, observed_states, transitions):
    pi = {}
    for s in observed_states:
       
        poss_rewards = [(rewards[s] + gamma * sum([transitions[(s, a)][sa] * float(U[sa])                                     for sa in transitions[(s, a)]])) for a in valid_actions[s]]
        
        pi[s] = valid_actions[s][np.argmax(poss_rewards)]
    return pi


# In[336]:

pi = extractPolicy(U, .99, valid_actions, observed_states, transitions)


# In[337]:

len(observed_states)


# In[338]:

pi


# In[354]:

num_unique_states = len(observed_states)
inputs = np.zeros((num_unique_states, 7))
y = np.zeros((num_unique_states, 15))

for idx, s in enumerate(pi):
    fullState = idxToState(s)
    inputs[idx, :] = fullState.squeeze()
    y[idx, pi[s]] = 1
    


# In[357]:

inputs


# In[ ]:



