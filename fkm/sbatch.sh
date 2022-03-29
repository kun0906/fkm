#!/bin/bash

module load anaconda3/2021.11
cd /scratch/gpfs/ky8517/fkm/fkm
PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_all.py
