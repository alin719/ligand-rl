import numpy as np
import math
from copy import deepcopy
import time
import sklearn.neighbors
from collections import defaultdict
import operator
import sys

defaultRanges = np.array([(10, -20), (0, -20), (10, -20),
                          (math.pi, 0), (math.pi, -math.pi),
                          (math.pi, 0), (math.pi, -math.pi)])
defaultNumBins = np.array([20, 20, 20, 5, 5, 5, 5])


def Translate(points, direction, magnitude):
    M = np.zeros((3, 3))
    print M[:, direction]
    M[:, direction] = np.array([1, 1, 1])*magnitude
    return points + M


def Rotate(points, pointIndex, axis, angle):
    pts = deepcopy(points)
    origin = pts[0, :]
    targetPoint = pts[pointIndex, :]
    zeroedTargetPoint = targetPoint - origin
    a = np.deg2rad(angle)
    sin, cos = np.sin(a), np.cos(a)
    if axis == 0:
        Rx = np.array([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
        zeroedTargetPoint = np.matmul(zeroedTargetPoint, Rx)
    elif axis == 1:
        Ry = np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
        zeroedTargetPoint = np.matmul(zeroedTargetPoint, Ry)
    elif axis == 2:
        Rz = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        zeroedTargetPoint = np.matmul(zeroedTargetPoint, Rz)
    else:
        print "Ya fukt up."
    zeroedTargetPoint += origin
    pts[pointIndex, :] = zeroedTargetPoint
    return pts


def getCenterOfMass(points):
    '''
    Finds the center of mass of the receptor points
    '''
    return np.array([np.mean(points[:, 0]),
                     np.mean(points[:, 1]),
                    np.mean(points[:, 2])])


def getTransformationMatrix(center, points):
    '''
    Given the points on the receptor, defines the axes
    '''
    primaryPoint = points[0, :]
    secondaryPoint = points[1, :]
    v1 = primaryPoint - center
    helper = secondaryPoint - center
    v2 = np.cross(v1, helper)
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    return np.array([v1, v2, v3])


def frameToState(p1, p2, p3, l1, l2, l3):
    '''
    Converts 6 points into the R^7 state (x,y,z,angles)
    '''
    proteinBase = np.array([p1, p2, p3])
    center = getCenterOfMass(proteinBase)
    M = getTransformationMatrix(center, proteinBase)

    tL1 = np.matmul(M, l1)
    tL2 = np.matmul(M, l2)
    tL3 = np.matmul(M, l3)

    tCenter = np.matmul(M, center)
    ligandPosition = tL1 - tCenter
    zeroedP2 = tL2 - tL1
    zeroedP3 = tL3 - tL1

    r2, theta2, phi2 = cartesian_to_spherical(zeroedP2)
    r3, theta3, phi3 = cartesian_to_spherical(zeroedP3)

    state = np.array([ligandPosition[0], ligandPosition[1], ligandPosition[2],
                      theta2, phi2, theta3, phi3])
    return state


def discretizeStates(states, binScales):
    '''
    Takes R^7 raw values and discretizes.
    Discretizes to bins of sizes binScales.
    This is for computing rewards
    '''
    states = np.round(states/binScales)
    states = states*binScales
    return states


def discretizeUnscaledStates(states, ranges, binScales):
    '''
    Shifts up by minimum (so nothing is negative)
    Rounds to the nearest bin
    Divides so that index starts at 0 and increments
    This is to be used for input to transition matrices etc
    '''
    mins = ranges[:, 1]
    print states.shape
    print 'MIN SHAPE: ', mins.shape
    states -= np.transpose(mins)
    states = np.round(states/binScales)
    return states


def getBinScales(ranges, numBins):
    binWidths = np.abs(ranges[:, 0]) + np.abs(ranges[:, 1])
    binScales = binWidths/numBins
    return binScales

def shiftAngle(states, axis, amount):
    n = states.shape[0]
    for i in xrange(n):
        if states[i, axis] < 0:
            states[i, axis] += amount
    return states

def stateToLigandPos(state, ranges, binScales):
    r = 8

    scaledState = state*binScales
    minRanges = np.transpose(ranges[:,1])
    shiftedState = scaledState + minRanges

    x, y, z, theta1, phi1, theta2, phi2 = shiftedState
    zeroL2 = sphericalToCartesian(np.array([r, theta1, phi1]))
    zeroL3 = sphericalToCartesian(np.array([r, theta2, phi2]))

    L1 = np.array([x, y, z])
    L2 = L1 + zeroL2
    L3 = L1 + zeroL3

    return L1, L2, L3

def createStates(data):
    '''
    Takes in a saved data file and constructs a state
    for each frame, but does not do any discretization.
    Returns the raw R^7 numpy array for each frame.
    '''
    n = data['receptor_1_pos'].shape[0]
    states = np.zeros((n, 7))

    r1Pos = data['receptor_1_pos']
    r2Pos = data['receptor_2_pos']
    r3Pos = data['receptor_3_pos']
    cLPos = data['center_ligand_pos']
    s1Pos = data['side1_ligand_pos']
    s2Pos = data['side2_ligand_pos']

    for i in xrange(n):
        p1 = r1Pos[i]
        p2 = r2Pos[i]
        p3 = r3Pos[i]
        l1 = cLPos[i]
        l2 = s1Pos[i]
        l3 = s2Pos[i]
        state = frameToState(p1, p2, p3, l1, l2, l3)
        states[i, :] = state 
    return states


def filterDiscreteStates(discreteStates, ranges, binScales):
    '''
    Filter scaled states.
    Unused.
    '''
    filterBooleans = np.ones((discreteStates.shape[0],)).astype(bool)
    scaledRanges = np.transpose(np.round(ranges.T/binScales)*binScales)
    for i in xrange(discreteStates.shape[0]):
        curState = discreteStates[i, :]
        for j in xrange(ranges.shape[0]):
            if (curState[j] > scaledRanges[j][0]) or curState[j] < (scaledRanges[j][1]):
                filterBooleans[i] = False
                break
    filteredStates = discreteStates[filterBooleans]
    return filteredStates


def filterUnscaledDiscreteStates(discreteStates, numBins):
    '''
    Filter UNscaled states (ie not real coordinates)
    Rejects points outside the range as defined by numBins.
    '''
    filterBooleans = np.ones((discreteStates.shape[0],)).astype(bool)
    n = discreteStates.shape[0]
    m = discreteStates.shape[1]
    for i in xrange(n):
        curState = discreteStates[i, :]
        for j in xrange(m):
            if (curState[j] >= numBins[j]) or (curState[j] < 0):
                filterBooleans[i] = False
                break
    # filteredStates = discreteStates[filterBooleans]
    return discreteStates, filterBooleans


# http://svn.gna.org/svn/relax/1.3/maths_fns/coord_transform.py
def cartesian_to_spherical(vector):
    """Convert the Cartesian vector [x, y, z] to spherical coordinates [r, theta, phi].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """

    # The radial distance.
    r = np.linalg.norm(vector)

    # Unit vector.
    unit = vector / r

    # The polar angle.
    theta = math.acos(unit[2])

    # The azimuth.
    phi = math.atan2(unit[1], unit[0])

    # Return the spherical coordinate vector.
    return np.array([r, theta, phi], np.float64)


def spherical_to_cartesian(spherical_vect, cart_vect):
    """Convert the spherical coordinate vector [r, theta, phi] to the Cartesian vector [x, y, z].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param spherical_vect:  The spherical coordinate vector [r, theta, phi].
    @type spherical_vect:   3D array or list
    @param cart_vect:       The Cartesian vector [x, y, z].
    @type cart_vect:        3D array or list
    """

    # Trig alias.
    sin_theta = math.sin(spherical_vect[1])

    # The vector.
    cart_vect[0] = spherical_vect[0] * math.cos(spherical_vect[2]) * sin_theta
    cart_vect[1] = spherical_vect[0] * math.sin(spherical_vect[2]) * sin_theta
    cart_vect[2] = spherical_vect[0] * math.cos(spherical_vect[1])


def lambdaDistance(s1, s2, **kwargs):
    '''
    Returns the weighted sum of the Euclidean distances between the first 3 elements
    of s1 and s2 and the remaining elements, weighting the latter by angleWeight
    '''
    return np.linalg.norm(s1[0:3]-s2[0:3]) + kwargs['angleWeight'] * np.linalg.norm(s1[3:]-s2[3:])


def nnTest():
    '''
    Verifies successful performance of the custom distance metric.
    '''
    testData = np.array([[0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 3, 3, 3, 3]])
    angleWeight = 0.5
    tree = sklearn.neighbors.NearestNeighbors(n_neighbors=5,
                                              algorithm='ball_tree',
                                              metric=lambdaDistance,
                                              metric_params={'angleWeight':angleWeight})
    tree.fit(testData)
    dist, ind = tree.kneighbors(X=testData[0, :], n_neighbors=2, return_distance=True)
    print dist, ind


def generateDistTree(discreteStates, angleWeight):
    '''
    Generates a ball-tree backed distance tree for fast kNN lookup.
    '''
    tree = sklearn.neighbors.NearestNeighbors(n_neighbors=5,
                                              algorithm='ball_tree',
                                              metric=lambdaDistance,
                                              metric_params={'angleWeight': angleWeight})
    tree.fit(discreteStates)
    return tree


def mostFrequentState(discreteStates):
    '''
    Returns the most frequent state observed
    '''
    seen = defaultdict(int)
    for s in discreteStates:
        seen[tuple(s)] += 1

    return np.array(max(seen.iteritems(), key=operator.itemgetter(1))[0]).reshape(1, -1)


def generateRewards(discreteStates, tree, l1, l2, angleWeight):
    '''
    Returns a vector with a reward for every state.
    l1: (-) weighting of neighboring state proximity reward
    l2: (-) weighting of distance to most frequent state reward
    '''
    neighboringStateRewards = np.zeros((discreteStates.shape[0],))
    for s in range(discreteStates.shape[0]):
        if s % 100 == 0:
            print 'On state {}'.format(s)
        dist, ind = tree.kneighbors(X=discreteStates[s, :].reshape(1, -1), n_neighbors=6, return_distance=True)
        neighboringStateRewards[s] = np.mean(dist)

    mostFreqRewards = np.zeros((discreteStates.shape[0],))
    mostFreqState = mostFrequentState(discreteStates)
    for s in range(discreteStates.shape[0]):
        dist = lambdaDistance(discreteStates[s, :].reshape(1, -1), mostFreqState, angleWeight=angleWeight)
        mostFreqRewards[s] = dist

    rewards = (-l1 * neighboringStateRewards) + (-l2 * mostFreqRewards)
    return rewards


def computeActions(episode):
    actions = []
    for i in range(episode.shape[0]-1):
        s = episode[i:i+1, :]
        sp = episode[i+1:i+2, :]
        actions.append((int(np.sign(sp-s)[0][np.argmax(abs(sp-s))]), np.argmax(abs(sp - s))))
    actions.append((0, -1))
    return actions


def reconstructStates(discreteStates, ranges, binScales):
    reconstructStates = discreteStates*binScales
    mins = ranges[:, 1]
    reconstructedStates = reconstructedStates - mins
    return reconstructedStates


def loadFiles(MAX_STATE):
    import glob
    PATH = '/home/rbedi/cs238/ligand-rl/data/*npz'
    files = glob.glob(PATH)
    allStates = np.zeros((0, 7))
    for file in files:
        print file
        data = np.load(file)
        start = time.time()
        states = createStates(data)[0:MAX_STATE]
        allStates = np.vstack((allStates, states))
        end = time.time()
        print end - start
    return allStates


def countFilteredOut(states, numBins):
    count = 0
    n = 700
    print states.shape
    booleanFiltered = np.ones((n, 1))
    for i in xrange(n):
        valid = True
        for j in xrange(states.shape[1]):
            if (states[i][j] >= numBins[j]) or (states[i][j] < 0):
                valid = False
                booleanFiltered[i] = 0
                break
        if valid:
            count += 1
            print 1, '              ', i
        else:
            print 0
    return count, booleanFiltered


def loadSingleFile(filepath, maxState):
    dataFile = sys.argv[1]
    MAX_STATE = int(sys.argv[2])

    dataId = ''.join(dataFile.split('/')[-1].split('-')[1:3])

    data = np.load(dataFile)
    start = time.time()
    states = createStates(data)[0:MAX_STATE]
    end = time.time()
    print end - start
    return states, dataId


if __name__ == "__main__":

    # states, dataId = loadSingleFile(sys.argv[1], 2000)

    states = loadFiles(2000)

    angleWeight = float(sys.argv[1])
    neighboringStateWeight = float(sys.argv[2])
    mostFreqWeight = float(sys.argv[3])

    dataId = 'all_A_B_' + str(angleWeight) + '_' + str(neighboringStateWeight) + '_' + str(mostFreqWeight)

    testStates = np.copy(states)
    testStates = shiftAngle(testStates, 6, 2*math.pi)

    ranges = np.array([(45, 15), (15, -20), (25, -5),
                       (math.pi, 0.2), (math.pi + 0.5, -math.pi + 0.4),
                       (math.pi, 0.2), (2*math.pi, 0)])
    numBins = np.array([20, 20, 20, 5, 5, 5, 5])
    binWidths = np.abs(ranges[:, 0]) + np.abs(ranges[:, 1])
    binScales = binWidths/numBins
    print binScales

    discreteStates = discretizeUnscaledStates(testStates, ranges, binScales)
    filteredStates, filterBooleans = filterUnscaledDiscreteStates(discreteStates, numBins)

    print 'Maximum state values'
    print np.max(filteredStates[filterBooleans], axis=0)
    print 'Minimum state values'
    print np.min(filteredStates[filterBooleans], axis=0)

    realStates = discretizeStates(states, binScales)
    print "Original states discretized"
    tree = generateDistTree(realStates, angleWeight)
    print "Tree generated"
    rewards = generateRewards(realStates, tree, neighboringStateWeight, mostFreqWeight, angleWeight)
    print "Rewards generated"
    actions = computeActions(realStates)
    print "actions generated"

    np.savez('/home/rbedi/cs238/ligand-rl/' + dataId + '_data',
             rewards=rewards,
             actions=actions,
             discreteStates=discreteStates,
             binScales=binScales,
             filtered=filterBooleans)
