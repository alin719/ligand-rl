import numpy as np
import state2idx as s2i
import generalize

def createActionDict():
    actionDict = {}
    for i in xrange(15):
        actionDict[i] = s2i.idxToAction(i)

def getRandomAction(actionDict):
    idx = np.random.randint(0, len(actionDict))
    return actionDict[idx]

def actionToVector(action):
    direction = action[0]
    dim = action[1]
    actionVector = np.zeros((1,7))
    actionVector[dim] = direction
    return actionVector

def isStateInbounds(state, numBins):
    isInbounds = True
    for i in xrange(state.shape[1]):
        if (curState[i] >= numBins[i]) or (curState[i] < 0):
            isInbounds = False
            break
    return isInbounds

def getNextEnumAction(G, state, actionDict):
	i = np.random.rand()
	epsilon = 0.2
	if i <= epsilon:
		return getRandomAction(actionDict)
	return G.getAction(state)

def rolloutPolicy(G, startState, numBins, numIters=1000, edgeThreshold=200):
    curState = startState
    statesVisited = []
    edgeCounter = 0
    actionDict = createActionDict()
    for i in xrange(numIters):
        curAction = actionDict[getNextEnumAction(G, curState, actionDict)]
        actionVector = actionToVector(curAction)
        nextState = curState + actionVector

        if not isStateInbounds(nextState):
            edgeCounter += 1
            while not isStateInbounds(nextState):
                newAction = getRandomAction(actionDict)
                newActionVector = actionToVector(newAction)
                nextState = curState + newActionVector

        statesVisited.append(curState)
        curState = nextState
        if edgeCounter > edgeThreshold:
            break
    return statesVisited

if __name__ == "__main__":
	weightsFilename = 'all_A_B_05_05_05_data_policynpz_weights'
	G = generalize.Generalizer(weightsFilename)
    numBins = np.array([20, 20, 20, 5, 5, 5, 5])

	startState = np.array([])
	
	trajectory = rolloutPolicy(G, startState, numBins, 1000, 200)
	print trajectory
    np.savez('/home/rbedi/cs238/ligand-rl/' + 'rollout_trajectory',
         trajectory=trajectory)