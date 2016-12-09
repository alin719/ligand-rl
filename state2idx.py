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
    return idx

def actionToIdx(action):
    if max(action) == 0 and min(action) == 0: 
        return 0
    return (action[0] + 1)/2 + (2 * action[1]) + 1

def idxToAction(idx):
    if idx == 0: 
        return (0,0)
    idx -= 1
    return ((idx%2)*2 - 1, idx/2)



test = [5,14,12,3,1,2,1]
print stateToIdx(test)
print idxToState(stateToIdx(test)).astype(int)

action = (-1, 3)
print actionToIdx(action)
print idxToAction(actionToIdx(action))
