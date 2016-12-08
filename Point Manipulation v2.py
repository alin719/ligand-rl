
# coding: utf-8

# In[14]:


# In[15]:

import numpy as np
import math
from copy import deepcopy
import time
import sklearn.neighbors

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as a3d

import colorlover as cl
import plotly.offline as po
import plotly.graph_objs as go
import plotly
plotly.offline.init_notebook_mode()
# In[19]:



# In[2]:

def Translate(points, direction, magnitude):
    M = np.zeros((3,3))
    print M[:,direction]
    M[:,direction] = np.array([1, 1, 1])*magnitude
    return points + M
    


# In[3]:

def Rotate(pts, pointIndex, axis, angle):
    points = deepcopy(pts)
    origin = points[0,:]
    targetPoint = points[pointIndex, :]
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
    points[pointIndex, :] = zeroedTargetPoint
    return points


# In[ ]:

def PlotLines(A):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.hold(True)
    for i in xrange(1, 3):
        xs1 = np.array([A[0,0], A[i,0]])
        ys1 = np.array([A[0,1], A[i,1]])
        zs1 = np.array([A[0,2], A[i,2]])
        print xs
        print ys
        print zs
        ax.plot(xs, ys, zs, '-o')
    plt.show()


# In[ ]:

def PlotlyPlot(A):
    xs1 = np.array([A[0,0], A[1,0]])
    ys1 = np.array([A[0,1], A[1,1]])
    zs1 = np.array([A[0,2], A[1,2]])
    xs2 = np.array([A[0,0], A[2,0]])
    ys2 = np.array([A[0,1], A[2,1]])
    zs2 = np.array([A[0,2], A[2,2]])

    trace1 = go.Scatter3d(
        x=[xs1[0],xs1[1]],
        y=[ys1[0],ys1[1]],
        z=[zs1[0],zs1[1]],
        mode='lines+markers',
        name='start ref',
        marker=dict(
            size=5,
            line=dict(
                color='green',
                width=2
            ),
            color=cl.scales['3']['qual']['Dark2'],
            opacity=0.8
        )
    )

    trace2 = go.Scatter3d(
        x=[xs2[0],xs2[1]],
        y=[ys2[0],ys2[1]],
        z=[zs2[0],zs2[1]],
        mode='lines+markers',
        name='start ref',
        marker=dict(
            size=5,
            line=dict(
                color='green',
                width=2
            ),
            color=cl.scales['3']['qual']['Dark2'],
            opacity=0.8
        )
    )

    layout = go.Layout(
        scene=dict(
            aspectratio=dict(
                y=1,
                x=1,
                z=1
                ),
            xaxis=dict(
                autorange=False,
                range=[-5, 5]
            ),
            yaxis=dict(
                autorange=False,
                range=[-5, 5]
            ),
            zaxis=dict(
                autorange=False,
                range=[-5, 5]
            )
        )
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    po.iplot(fig)


# In[4]:

def getCenterOfMass(points):
    return np.array([np.mean(points[:,0]), np.mean(points[:,1]), np.mean(points[:,2])])


# In[5]:

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
    


# In[6]:

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


# In[7]:

def frameToState(p1, p2, p3, l1, l2, l3):
    proteinBase = np.array([p1, p2, p3])
    center = getCenterOfMass(proteinBase)
    M = getTransformationMatrix(center, proteinBase)
    
    tP1 = np.matmul(M, p1)
    tP2 = np.matmul(M, p2)
    tP3 = np.matmul(M, p3)
    tL1 = np.matmul(M, l1)
    tL2 = np.matmul(M, l2)
    tL3 = np.matmul(M, l3)

    tCenter = np.matmul(M, center)
    ligandPosition = tL1 - tCenter  
    zeroedP2 = tL2 - tL1
    zeroedP3 = tL3 - tL1
#     print 'Center at:', tCenter

    r2, theta2, phi2 = cartesian_to_spherical(zeroedP2)
    r3, theta3, phi3 = cartesian_to_spherical(zeroedP3)
    
    state = np.array([ligandPosition[0], ligandPosition[1], ligandPosition[2], theta2, phi2, theta3, phi3])
    return state
    


# In[8]:

def getScaleVector(xsize, xsteps, ysize, ysteps, zsize, zsteps, rotsteps):
    #Divide by 2 to account for positive and negative
    xscale = xsize/(xsteps/2)
    yscale = ysize/(ysteps/2)
    zscale = zsize/(zsteps/2)
    rotscale = math.pi/(rotsteps/2)
    return np.array([xscale, yscale, zscale, rotscale, rotscale, rotscale, rotscale])


# In[9]:

def plotPointDistributions(curPoints, ranges, numBins):
    fieldNames = ['X', 'Y', 'Z', 'Theta1', 'Phi1', 'Theta2', 'Phi2']
    thisX = curPoints[:,0]
    thisY = curPoints[:,1]
    thisZ = curPoints[:,2]
    print 'IN PLOT'
    print curPoints
    thisTheta1 = curPoints[:,3]
    thisPhi1 = curPoints[:,4]
    thisTheta2 = curPoints[:,5]
    thisPhi2 = curPoints[:,6]
    axes = [thisX, thisY, thisZ, thisTheta1, thisPhi1, thisTheta2, thisPhi2]

    filteredPoints = np.copy(curPoints)
    for i in xrange(len(axes)):
        curAxis = axes[i]
        print curAxis
        plt.figure(i)
        plt.title(fieldNames[i])
        curAxis = curAxis[0:2000]
        plt.scatter(range(len(curAxis)), curAxis)
        plt.ylim(ranges[i])
        plt.show()
        plt.figure(2*i)
        curRange = ranges[i]
        filteredAxis = curAxis[(curAxis <= curRange[0]) & (curAxis >= curRange[1])]

    count = 0
    for i in xrange(len(filteredPoints)):
        valid = True
        for j in xrange(3):
            curRange = ranges[j]
            if not (filteredPoints[i][j] <= curRange[0] and filteredPoints[i][j] >= curRange[1]):
                valid = False
                break
        if valid:
            count = count + 1
    print count


# In[10]:

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



# In[12]:

def discretizeStates(states, binScales):#, binNumbers):
    states = np.round(states/binScales)
    states = states*binScales
    return states



def lambdaDistance(s1, s2, **kwargs):
    '''
    '''
    return np.linalg.norm(s1[0:3]-s2[0:3]) +  kwargs['lambda'] * np.linalg.norm(s1[3:]-s2[3:]) 

def nnTest():
    testData = np.array([[0,0,0,0,0,0,0],[2,2,2,3,3,3,3]])
    angleWeight = 0.5
    tree = sklearn.neighbors.NearestNeighbors(n_neighbors=5, 
                                              algorithm='ball_tree', 
                                              metric=lambdaDistance, 
                                              metric_params={'lambda':angleWeight})
    tree.fit(testData)
    dist, ind = tree.kneighbors(X=testData[0, :], n_neighbors=2, return_distance=True)
    print dist, ind

def generateDistTree(discreteStates, angleWeight):
    tree = sklearn.neighbors.NearestNeighbors(n_neighbors=5, 
                                              algorithm='ball_tree', 
                                              metric=lambdaDistance, 
                                              metric_params={'lambda':angleWeight})
    tree.fit(discreteStates)
    return tree

def generateRewards(discreteStates, tree):
    rewards = np.zeros((discreteStates.shape[0],))
    for s in range(discreteStates.shape[0]):
        if s % 100 == 0:
            print 'On state {}'.format(s)
        dist, ind = tree.kneighbors(X=discreteStates[s, :], n_neighbors=6, return_distance=True)
        rewards[s] = np.sum(dist)
    return rewards

data = np.load('trajA_coords.npz')
start = time.time()
states = createStates(data)
end = time.time()
print end - start

testStates = np.copy(states)
ranges = np.array([(10, -20), (0, -20), (10,-20), (math.pi, 0), (math.pi, -math.pi), (math.pi, 0), (math.pi, -math.pi)])
numBins = np.array([20, 20, 20, 5, 5, 5, 5])
binWidths = np.abs(ranges[:,0]) + np.abs(ranges[:,1])
binScales = binWidths/numBins
print binScales
discreteStates = discretizeStates(testStates, binScales)

discreteRanges = np.array([(numBin, 0) for numBin in numBins])
print discreteStates
# plotPointDistributions(discreteStates, ranges, numBins)
print np.min(discreteStates, 0)

tree = generateDistTree(discreteStates, 0.5)
rewards = generateRewards(discreteStates, tree)

np.save('trajArewards', rewards)

