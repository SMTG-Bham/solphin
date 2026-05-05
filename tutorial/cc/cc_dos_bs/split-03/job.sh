#!/bin/bash
#SBATCH -n 60
#SBATCH --qos bbdefault
#SBATCH --constraint=icelake
#SBATCH -A scanlodo-pv-defects
#SBATCH -t 90:30:0
#SBATCH --nodes 1

module purge
module load bluebear
module load bear-apps/2022a
module load VASP/6.4.2-foss-2022a

mpirun -np ${SLURM_NTASKS} vasp_std
