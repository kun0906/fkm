#!/bin/bash

#SBATCH --job-name='sbatch'         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
## SBATCH --mem-per-cpu=40G         # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)\
#SBATCH --priority=10             # Only Slurm operators and administrators can set the priority of a job.
# # SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --output=out_sbatch.txt
#SBATCH --error=err_sbatch.txt
# #SBATCH --mail-user=kun.bj@cloud.com # not work \
# #SBATCH --mail-user=<YourNetID>@princeton.edu
#SBATCH --mail-user=ky8517@princeton.edu

#module purge
module load anaconda3/5.0.1
source activate tf1-gpu

# check cuda and cudnn version for tensorflow_gpu==1.13.1
# https://www.tensorflow.org/install/source#linux
module load cudatoolkit/10.0
module load cudnn/cuda-10.0/7.5.0

whereis nvcc
which nvcc
nvcc --version

cd /scratch/gpfs/ky8517/leaf-torch/data/femnist
pwd
python3 -V
#python vgg16_zero.py
./preprocess.sh

#
#module load anaconda3/2021.5
#cd /scratch/gpfs/ky8517/fkm/afkm
#PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main.py

#
#
#   Please run fit() on GPU if you want save yourself.
#			Commands:
#				### Create GPU environment
#				ssh ky8517@tigergpu.princeton.edu
#				$ module load anaconda3/2021.11
#				$ conda create --name tf2_10_0-gpu-py397 python=3.9.7
#				$ conda env list
#				$ conda activate tf2_10_0-gpu-py397
#				$ cd /scratch/gpfs/ky8517/fkm
#				$ pip3 install -r requirement.txt       (install on the login machine)
#
#				### pip install tensorflow-gpu==2.10.0 --no-cache-dir
#				### conda deactivate
#
#				### log into a GPU node.
#				$ srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
#				$ cd /scratch/gpfs/ky8517/fkm/fkm
#				### module purge
#				$module load anaconda3/2021.11
#				$source activate tf2_10_0-gpu-py397
#
#				# check cuda and cudnn version for tensorflow_gpu==1.13.1
#				# https://www.tensorflow.org/install/source#linux
#				module load cudatoolkit/11.2
#				module load cudnn/cuda-11.x/8.2.0
#
#				whereis nvcc
#				which nvcc
#				nvcc --version
#
#				cd /scratch/gpfs/ky8517/fkm/fkm
#				PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 datasets/mnist.py
#
#				### download the fitted model from HPC to local
#				scp ky8517@tigergpu.princeton.edu:/scratch/gpfs/ky8517/fkm/fkm/datasets/MNIST/vgg16_zero.h5 ~/Downloads/
