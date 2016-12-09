import numpy as np
import mdptoolbox
import state2idx as s2i
import scipy.sparse as sparse
from collections import Counter
import time

def getStateSpaceSize():
    bases = np.array([20, 20, 20, 5, 5, 5, 5, 1])
    n = 1
    for elem in bases:
        n*=elem
    return n

def enumerateVectors(vectors, rewards):
    vectorToRewardDict = {}
    vectorSet = set()
    numTotalVectorss = vectors.shape[0]
    for i in xrange(numTotalVectorss):
        curVector = tuple(vectors[i,:])
        vectorToRewardDict[curVector] = rewards[i]
        vectorSet.add(curVector)

    enumVectorToVectorDict = {}
    vectorToEnumVectorDict = {}
    counter = 0
    for vector in vectorSet:
        enumVectorToVectorDict[counter] = vector
        vectorToEnumVectorDict[vector] = counter
        counter += 1

    return vectorToEnumVectorDict, enumVectorToVectorDict, vectorToRewardDict

# stateToEnumStateDict, enumStateToStateDict, stateToRewardDict = \
#     enumerateVectors(states, rewards)

def enumActionStateToNewVectorState(enumAction, enumState, \
        enumStateToStateDict, enumActionToActionDict, binScales):
    actualAction = enumActionToActionDict[enumAction]
    actualState = np.array(enumStateToStateDict[enumState])
    affectedDimension = actualAction[1]
    affectedAmount = binScales[affectedDimension]*actualAction[0]   
    actualState[affectedDimension] += affectedAmount
    computedSp = tuple(actualState)
    return computedSp

def getStateIdxMappingDicts(n):
    stateToIdxDict = {}
    idxToStateDict = {}
    print 'Precomputing state/idx mappings'
    for i in xrange(n):
        if i % 1000000 == 0:
            print i, '/', n
        state = np.transpose(s2i.idxToState(i))[0]
        stateToIdxDict[tuple(state)] = i
        idxToStateDict[i] = state
    return stateToIdxDict, idxToStateDict

def getObservedTransitionCounters(states, actions, filterBooleans, m, stateToIdxDict, idxToStateDict):
    '''

    '''
    numObservedStates = states.shape[0]
    print 'Computing observed states...'
    start = time.time()
    observedTransitionCounters = {}
    for i in xrange(m):
        observedTransitionCounters[i] = {}
    for i in xrange(numObservedStates - 1):
        if filterBooleans[i] or filterBooleans[i+1]:
            continue
        s = tuple(states[i])
        sp = tuple(states[i+1])
        a = actions[i]
        
        sIdx = stateToIdxDict[s]
        spIdx = stateToIdxDict[sp]
        if sIdx >= 0 and spIdx >= 0:
            aIdx = s2i.actionToIdx(a)
            if sIdx not in observedTransitionCounters[aIdx]:
                observedTransitionCounters[aIdx][sIdx] = Counter()
            observedTransitionCounters[aIdx][sIdx][spIdx] += 1
    end = time.time()
    print 'Finished computing observed states.  Elapsed time: ', end - start
    return observedTransitionCounters

def createTransitionMatrix(rewards, actions, states, filterBooleans):
    n = getStateSpaceSize()
    m = 15 #numActions

    stateToIdxDict, idxToStateDict = getStateIdxMappingDicts(n)
    observedTransitionCounters = \
        getObservedTransitionCounters(states, actions, filterBooleans, m, stateToIdxDict, idxToStateDict)

    T = [sparse.lil_matrix((n, n), dtype=np.float32)]*m
    observedStates = [set()]*m

    print 'Adding observed states to sparse matrices...'
    start = time.time()
    for aIdx in observedTransitionCounters:
        sIdxSum = 0
        for sIdx in observedTransitionCounters[aIdx]:
            for spIdx in observedTransitionCounters[aIdx][sIdx]:
                sIdxSum += observedTransitionCounters[aIdx][sIdx][spIdx]
            for spIdx in observedTransitionCounters[aIdx][sIdx]:
                T[aIdx][sIdx, spIdx] = \
                    observedTransitionCounters[aIdx][sIdx][spIdx] / sIdxSum
                observedStates[aIdx].add(sIdx)
    end = time.time()
    print 'Finished adding observed states.  Elapsed time: ', end - start

    for i in xrange(m):
        print 'Adding transitions for action ', i
        start = time.time()
        a = s2i.idxToAction(i)
        for j in xrange(n):
            if j % 100000 == 0:
                end = time.time()
                print j, '/', n, '  Time elapsed: ', end - start
            if j not in observedStates[i]:
                s = idxToStateDict[j]
                spCalc = s
                spCalc[a[1]] += a[0]
                sp = stateToIdxDict[tuple(spCalc)]
                T[i][j, sp] = 1
        end = time.time()
        print 'Finished addding transitions for action ', i, ' - ', end-start
    return T

if __name__ == "__main__":
     #Load data in
    data = np.load('trajA_mdp_data.npz')
    rewards = data['rewards']
    actions = data['actions']
    states = data['discreteStates']
    filterBooleans = data['filtered']
    
    T = createTransitionMatrix(rewards, actions, states, filterBooleans)
    np.savez('trajA_transitions', transitionMatrix=T)