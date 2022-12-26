#!/bin/bash

# load environment and collect results
#module load anaconda3/2021.11
cd /scratch/gpfs/ky8517/fkm/fkm
#PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 vis/collect_results.py
PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 vis/collect_K_results.py
#PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 collect_table_results.py



