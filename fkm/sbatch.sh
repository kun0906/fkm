#!/bin/bash

module load anaconda3/2021.5
cd /scratch/gpfs/ky8517/fkm/fkm
PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 sbatch.py
