import vmd
import atomsel
import numpy as np
import sys

def align_frames(reference, target):
    '''
    Aligns frames of target molecule to first frame of reference molecule
    '''
    num_frames = vmd.molecule.numframes(target)

    s1 = atomsel.atomsel('protein and name CA', reference, 0)
    all_s1 = atomsel.atomsel('all', reference, 0)

    for frame in range(1, num_frames):
        s2 = atomsel.atomsel('protein and name CA', target, frame)
        mat = s2.fit(s1)

        all_s2 = atomsel.atomsel('all', target, frame)
        old_rmsd = all_s2.rmsd(all_s1)
        all_s2.move(mat)
        new_rmsd = all_s2.rmsd(all_s1)

        print "RMSD btw frames 0 and {:d}: {:.2f} -> {:.2f}".format(
            frame, old_rmsd, new_rmsd)


def export_sel(molid, sel_descriptor):
    '''
    Exports coordinates from molid over all frames corresponding to
    the selection of atoms specified by sel
    '''
    sel_pos = []
    num_frames = 4  #vmd.molecule.numframes(molid)
    for frame in range(num_frames):
        frame_data = np.copy(vmd.vmdnumpy.timestep(molid, frame))
        sel = vmd.vmdnumpy.atomselect(molid, frame, sel_descriptor)
        sel_pos.append(np.take(frame_data, np.nonzero(sel), 0)[0])

    return np.array(sel_pos).squeeze()

sim_id = sys.argv[1]
sim_path = sys.argv[2]

# Load the trajectory
ROOT = '/scratch/PI/rondror/DesRes-Simulations/Downloaded/M2R_Nature_Dror_Desres'
TRAJ = 'DESRES-Trajectory_nature2013-' + sim_id + '/DESRES-Trajectory_' + sim_path + '/' + sim_path
PATH = ROOT+'/'+TRAJ+'/'
traj_name = TRAJ.split('/')[-1]

vmd.molecule.load('mae', PATH+traj_name+'.mae', 'dcd', PATH+traj_name+'-000.dcd')

# Align the frames to the first frame of the simulation
align_frames(0, 0)


if sim_id == 'A' or sim_id == 'B':
    # Extract three coordinates of the ligand
    # These values are specific to trajectory A
    ligand_pos = export_sel(0, 'resname C73P')
    center_ligand_pos = ligand_pos[:, 18, :]    # corresponds to C2 on C73P (index 2230 in A.mae)
    side1_ligand_pos = ligand_pos[:, 20, :]    # corresponds to C on C73P (index 2232 in A.mae)
    side2_ligand_pos = ligand_pos[:, 0, :]     # corresponds to C on C73P (index 2212 in A.mae)

    # Extract the centroid of the binding site
    # These values are also specific to trajectory A
    receptor_1_pos = export_sel(0, 'resid 410 and name CA')  # CA on ASN residue 410
    receptor_2_pos = export_sel(0, 'resid 127 and name CA')  # CA on TYR residue 177
    receptor_3_pos = export_sel(0, 'resid 422 and name CA')  # CA on TRP residue 422


np.savez(sim_path + '_coords',
         center_ligand_pos=center_ligand_pos,
         side1_ligand_pos=side1_ligand_pos,
         side2_ligand_pos=side2_ligand_pos,
         receptor_1_pos=receptor_1_pos,
         receptor_2_pos=receptor_2_pos,
         receptor_3_pos=receptor_3_pos)
