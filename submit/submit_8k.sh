#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=benchmark7B
## filename for job standard output (stdout)
## %j is the job id, %u is the user id

#SBATCH --output=/data/home/beidic/zhuoming/log-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/data/home/beidic/zhuoming/log-%j.err

#SBATCH --time=6:00:00

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --nodes=1

## number of tasks per node
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=8
#SBATCH --no-requeue

source /data/home/beidic/.bashrc
source /data/home/beidic/miniconda/etc/profile.d/conda.sh
source activate base
conda activate zoo-torch20
which python
cd /data/home/beidic/hanshi/LongContextInfer

python benchmark_flash.py --datalen 8000
