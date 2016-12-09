
# coding: utf-8

# In[82]:

import numpy as np

bases = np.array([20, 20, 20, 5, 5, 5, 5, 1])
mods = [np.prod(bases[idx:]) for idx in range(len(bases))]

def idxToState(idx):
    state = np.zeros((len(bases)-1, 1))
    for i in range(len(mods)-2,-1,-1):
        state[i] = (idx % mods[i]) / mods[i+1]
        idx -= state[i]
        print idx
    return state
    
def stateToIdx(state):
    idx = 0
    for i, el in enumerate(state):
        toAdd = np.prod(bases[i+1:]) * el
        idx += toAdd
    return idx


# In[83]:

test = [5,14,12,3,1,2,1]
stateToIdx(test)
idxToState(stateToIdx(test)).astype(int)

