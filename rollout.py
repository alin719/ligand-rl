import numpy as np
import state2idx as s2i

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

def isStateInbounds(state, ranges):
    isInbounds = True
    for i in xrange(state.shape[1]):
        if (curState[i] >= numBins[i]) or (curState[i] < 0):
            isInbounds = False
            break
    return isInbounds

def rolloutPolicy(startState, ranges, numIters=1000, edgeThreshold=200):
    curState = startState
    statesVisited = []
    edgeCounter = 0
    actionDict = createActionDict()
    for i in xrange(numIters):
        curAction = actionDict[getNextEnumAction(curState)]
        actionVector = actionToVector(curAction)
        nextState = curState + actionVector

        #Policy-given 
        if not isStateInbounds(nextState):
            edgeCounter += 1
            while not isStateInbounds(nextState):
                newAction = getRandomAction(actionDict)
                newActionVector = actionToVector(newAction)
                nextState = curState + actionVector

        statesVisited.append(curState)
        curState = nextState
        if edgeCounter > edgeThreshold:
            break
    return statesVisited