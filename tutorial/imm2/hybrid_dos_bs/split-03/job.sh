#!/bin/bash
#SBATCH -n 52
#SBATCH --constraint icelake
#SBATCH --qos bbdefault
#SBATCH -A scanlodo-pv-defects
#SBATCH -t 200:00:00
#SBATCH --cpus-per-task=2
#SBATCH --nodes 2

module purge
module load bluebear
module load bear-apps/2022a
module load VASP/6.4.2-foss-2022a

mpirun -np ${SLURM_NTASKS} vasp_std
