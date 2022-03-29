#!/usr/bin/env bash

: ' multiline comment
  running the shell with source will change the current bash path
  e.g., source stat.sh

  check cuda and cudnn version for tensorflow_gpu==1.13.1
  https://www.tensorflow.org/install/source#linux
'
#ssh ky8517@tigergpu.princeton.edu
#srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
cd /scratch/gpfs/ky8517/leaf/data/femnist
module load anaconda3/5.0.1
module load cudatoolkit/10.0
module load cudnn/cuda-10.0/7.5.0
source activate tf1-gpu

