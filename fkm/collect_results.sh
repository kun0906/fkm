#!/bin/bash

# load environment and collect results
module load anaconda3/2021.5
cd /scratch/gpfs/ky8517/fkm/fkm
PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 collect_results.py



