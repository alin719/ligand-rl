import vmd
import atomsel
import numpy as np


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
    num_frames = vmd.molecule.numframes(molid)
    for frame in range(num_frames):
        frame_data = vmd.vmdnumpy.timestep(molid, frame)
        sel = vmd.vmdnumpy.atomselect(molid, frame, sel_descriptor)
        sel_pos.append(np.take(frame_data, np.nonzero(sel), 0)[0][0])

    return np.array(sel_pos)


# Load the trajectory
ROOT = '/scratch/PI/rondror/DesRes-Simulations/Downloaded/M2R_Nature_Dror_Desres'
TRAJ = 'DESRES-Trajectory_nature2013-A/DESRES-Trajectory_nature2013-A-01-sel1/nature2013-A-01-sel1'
PATH = ROOT+'/'+TRAJ+'/'
traj_name = TRAJ.split('/')[-1]

vmd.molecule.load('mae', PATH+traj_name+'.mae', 'dcd', PATH+traj_name+'-000.dcd')

# Align the frames to the first frame of the simulation
align_frames(0, 0)

# Extract three coordinates of the ligand
# These values are specific to trajectory A
center_ligand_pos = export_sel(0, 'index 2230')  # corresponds to a C2 on C73P
side1_ligand_pos = export_sel(0, 'index 2232')  # corresponds to a C on one side of C73P
side2_ligand_pos = export_sel(0, 'index 2212')  # corresponds to a C on the other side of C73P

# Extract the centroid of the binding site
# These values are also specific to trajectory A

receptor_1_pos = export_sel(0, 'index 1837')  # CA on ASN residue 410
receptor_2_pos = export_sel(0, 'index 1241')  # CA on TYR residue 177
receptor_3_pos = export_sel(0, 'index 1924')  # CA on TRP residue 422

np.savez('trajA_coords',
         center_ligand_pos=center_ligand_pos,
         side1_ligand_pos=side1_ligand_pos,
         side2_ligand_pos=side2_ligand_pos,
         receptor_1_pos=receptor_1_pos,
         receptor_2_pos=receptor_2_pos,
         receptor_3_pos=receptor_3_pos)
