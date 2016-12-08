import numpy as np
import math
from copy import deepcopy
import time

defaultRanges = np.array([(10, -20), (0, -20), (10,-20), (math.pi, 0), (math.pi, -math.pi), (math.pi, 0), (math.pi, -math.pi)])
defaultNumBins = np.array([20, 20, 20, 5, 5, 5, 5])

def Translate(points, axis, magnitude):
    M = np.zeros((3,3))
    print M[:,direction]
    M[:,direction] = np.array([1, 1, 1])*magnitude
    return points + M


 def Rotate(points, pointIndex, axis, angle):
    pts = deepcopy(points)
    origin = pts[0,:]
    targetPoint = pts[pointIndex, :]
    zeroedTargetPoint = targetPoint - origin
    a = np.deg2rad(angle)
    sin, cos = np.sin(a), np.cos(a)
    if axis == 0:
        Rx = np.array([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
        zeroedTargetPoint = np.matmul(zeroedTargetPoint,Rx)
    elif axis == 1:
        Ry = np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
        zeroedTargetPoint = np.matmul(zeroedTargetPoint,Ry)
    elif axis == 2:
        Rz = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        zeroedTargetPoint = np.matmul(zeroedTargetPoint,Rz)
    else:
        print "Ya fukt up."
    zeroedTargetPoint += origin
    pts[pointIndex, :] = zeroedTargetPoint
    return pts

def createStates(data):
    n = data['receptor_1_pos'].shape[0]
    states = np.zeros((n,7))
    xsize = 50
    xsteps = 20
    ysize = 50
    ysteps = 20
    zsize = 50
    zsteps = 20
    rotstep = 5

    for i in xrange(n):
        p1 = data['receptor_1_pos'][i]
        p2 = data['receptor_2_pos'][i]
        p3 = data['receptor_3_pos'][i]
        l1 = data['center_ligand_pos'][i]
        l2 = data['side1_ligand_pos'][i]
        l3 = data['side2_ligand_pos'][i]
        state = frameToState(p1, p2, p3, l1, l2, l3)
        scaleVector = getScaleVector(xsize, xsteps, ysize, ysteps, zsize, zsteps, rotstep)
        discreteState = discretizeState(state, scaleVector)
        states[i,:] = discreteState   
    return states

def getCenterOfMass(points):
    return np.array([np.mean(points[:,0]), np.mean(points[:,1]), np.mean(points[:,2])])

def getTransformationMatrix(center, points):
    primaryPoint = points[0,:]
    secondaryPoint = points[1,:]
    v1 = primaryPoint - center
    helper = secondaryPoint - center
    v2 = np.cross(v1, helper)
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    return np.array([v1, v2, v3])
    
def frameToState(p1, p2, p3, l1, l2, l3):
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
    
    state = np.array([ligandPosition[0], ligandPosition[1], ligandPosition[2], theta2, phi2, theta3, phi3])
    return state
    
    
def discretizeStates(states, binScales):
    states = np.round(states/binScales)
    states = states*binScales
    return states

def getBinScales(ranges, numBins):
    binWidths = np.abs(ranges[:,0]) + np.abs(ranges[:,1])
    binScales = binWidths/numBins
    return binScales

def createStates(data):
    n = data['receptor_1_pos'].shape[0]
    states = np.zeros((n,7))
    xsize = 50
    xsteps = 20
    ysize = 50
    ysteps = 20
    zsize = 50
    zsteps = 20
    rotstep = 5
    
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
        states[i,:] = state   
    return states

def filterDiscreteStates(discreteStates, ranges, binScales):
    filterBooleans = np.ones((discreteStates.shape[0],)).astype(bool)
    scaledRanges = np.transpose(np.round(ranges.T/binScales)*binScales)
    for i in xrange(discreteStates.shape[0]):
        curState = discreteStates[i,:]
        for j in xrange(ranges.shape[0]):
            if (curState[j] > scaledRanges[j][0]) or curState[j] < (scaledRanges[j][1]):
                filterBooleans[i] = False
                break
    filteredStates = discreteStates[filterBooleans]
    return filteredStates

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