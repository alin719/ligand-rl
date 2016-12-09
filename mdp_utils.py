def enumerateVectors(vectors, rewards)
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

stateToEnumStateDict, enumStateToStateDict, stateToRewardDict = \
    enumerateVectors(states, rewards)

def enumActionStateToNewVectorState(enumAction, enumState, \
        enumStateToStateDict, enumActionToActionDict, binScales):
    actualAction = enumActionToActionDict[enumAction]
    actualState = np.array(enumStateToStateDict[enumState])
    affectedDimension = actualAction[1]
    affectedAmount = binScales[affectedDimension]*actualAction[0]   
    actualState[affectedDimension] += affectedAmount
    computedSp = tuple(actualState)
    return computedSp