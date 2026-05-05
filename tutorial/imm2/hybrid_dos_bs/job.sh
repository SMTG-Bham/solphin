#!/bin/bash
#SBATCH -n 152
#SBATCH --qos bbdefault
#SBATCH --constraint sapphire
#SBATCH -A scanlodo-pv-defects
#SBATCH -t 240:00:0
#SBATCH --nodes 3
#SBATCH --mem-per-cpu=6750M


module purge
module load bluebear
module load bear-apps/2022a
module load VASP/6.5.0-intel-2022a

mpirun -np ${SLURM_NTASKS} vasp_std
