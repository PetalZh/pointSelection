#!/bin/bash
#SBATCH --cpus-per-task=8
###SBATCH --nodelist=node5
###SBATCH --exclude=node[1-4, 22-24]
#SBATCH -J petal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -A OD-221915

# init tensorflow env
source activate py38
#export NUMBAPRO_NVVM=/home/zha374/miniconda3/envs/py38/nvvm/lib64/libnvvm.so
#export NUMBAPRO_LIBDEVICE=/home/zha374/miniconda3/envs/py38/nvvm/libdevice/
# export CUDA_HOME=/home/zha374/miniconda3/envs/py38/

#source activate tick-dev

# MAIN BATCH COMMANDS
echo "RUNNING $@"
python $@

#cd -