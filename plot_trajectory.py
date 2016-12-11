import numpy as np
import matplotlib.pyplot as plt

def plotPointDistributions(curPoints, numBins, prefixName):
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
        plt.title(fieldNames[i], fontsize=20)
        plt.xlabel('Frame #')
        plt.ylabel('State #')
        plt.ylim(0, numBins[i])
        plt.xlim(0, 2000)
        curAxis = curAxis[0:2000]
        plt.scatter(range(len(curAxis)), curAxis)
        plt.ylim(0, numBins[i])
        figureFileName = 'trajectory_plots/'+ prefixName + '_' + fieldNames[i] + '.png'
        plt.savefig(figureFileName)
        # plt.show()

def plotSubplotsPointDistributions(curPoints, numBins, prefixName):
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
    f, axarr = plt.subplots(len(axes), sharex=True, figsize=(8,24))

    for i in xrange(len(axes)):
        curAxis = axes[i]
        axarr[i].set_title(fieldNames[i], fontsize=20)        
        curAxis = curAxis[0:2000]
        axarr[i].scatter(range(len(curAxis)), curAxis)
        axarr[i].set_ylim(0, numBins[i])
        axarr[i].set_xlim(0, 2000)
        axarr[i].set_ylabel('State #')
    axarr[6].set_ylabel('Frame #')
    figureFileName = 'trajectory_plots/'+ prefixName + '_all.png'
    
    plt.savefig(figureFileName)
    # plt.show()
    
def plotSplitSubplotsPointDistributions(curPoints, numBins, prefixName):
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
    
    f, axarr = plt.subplots(3, sharex=True, figsize=(8,18))

    for i in xrange(3):
        curAxis = axes[i]
        axarr[i].set_ylabel('State #')
        axarr[i].set_title(fieldNames[i], fontsize=20)
        curAxis = curAxis[0:2000]
        axarr[i].scatter(range(len(curAxis)), curAxis)
        axarr[i].set_ylim(0, numBins[i])
        axarr[i].set_xlim(0, 2000)
    axarr[2].set_xlabel('Frame #')

    figureFileName = 'trajectory_plots/'+ prefixName + '_xyz.png'
    plt.savefig(figureFileName)
    # plt.show()
    
    f, axarr = plt.subplots(4, sharex=True, figsize=(8,24))
    for i in xrange(3, 7):
        j = i - 3
        curAxis = axes[i]
        axarr[j].set_ylabel('State #')
        axarr[j].set_title(fieldNames[i], fontsize=20)
        curAxis = curAxis[0:2000]
        axarr[j].scatter(range(len(curAxis)), curAxis)
        axarr[j].set_ylim(0, numBins[i])
        axarr[j].set_xlim(0, 2000)
    axarr[3].set_xlabel('Frame #')

    figureFileName = 'trajectory_plots/'+ prefixName + '_angles.png'
    
    plt.savefig(figureFileName)
    # plt.show()

def plotTrajectory(filename):
    traj = np.load(filename)['trajectory']
    numBins = np.array([20, 20, 20, 5, 5, 5, 5])
    prefixName = filename.split('.npz')[0]
    plotPointDistributions(traj, numBins, prefixName)
    plotSubplotsPointDistributions(traj, numBins, prefixName)
    plotSplitSubplotsPointDistributions(traj, numBins, prefixName)

if __name__ == "__main__":
	files = ['rollout_trajectory_2k_rand.npz']
	for filename in files:
		plotTrajectory(filename)