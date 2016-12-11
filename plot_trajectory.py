import numpy as np
import matplotlib.pyplot as plt

def plotPointDistributions(curPoints, numBins):
    fieldNames = ['X', 'Y', 'Z', 'Theta1', 'Phi1', 'Theta2', 'Phi2']
    thisX = curPoints[:,0]
    thisY = curPoints[:,1]
    thisZ = curPoints[:,2]
    thisTheta1 = curPoints[:,3]
    thisPhi1 = curPoints[:,4]
    thisTheta2 = curPoints[:,5]
    thisPhi2 = curPoints[:,6]
    axes = [thisX, thisY, thisZ, thisTheta1, thisPhi1, thisTheta2, thisPhi2]

    filteredPoints = np.copy(curPoints)
    for i in xrange(len(axes)):
        curAxis = axes[i]
        plt.figure(i)
        plt.title(fieldNames[i])
        curAxis = curAxis[0:2000]
        plt.scatter(range(len(curAxis)), curAxis)
        plt.ylim(0, numBins[i])
        plt.show()
        plt.figure(2*i)

def plotTrajectory(filename):
    traj = np.load(filename)['trajectory']
    numBins = np.array([20, 20, 20, 5, 5, 5, 5])
    plotPointDistributions(traj, numBins)
    
if __name__ == "__main__":
	plotTrajectory('rollout_trajectory.npz')