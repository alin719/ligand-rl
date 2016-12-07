import numpy as np
from copy import deepcopy



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