#!/bin/bash

for i in `seq 1 5`; do
  srun \
    -N1 \
    --mem=10G \
    --time=10:00:00 \
    --pty bash -i \
    Rscript my_script.R --subset $i --file $1 > "$OUTPUT-$i" &

    #module purge
  module load anaconda3/2021.5
  cd /scratch/gpfs/ky8517/fkm/fkm
  # PYTHONPATH = '..' python3 Our_greedy_initialization.py > greedy.txt 2 > & 1 &
  PYTHONPATH='..' python3 Our_greedy_initialization.py -n '00' > Our_greedy_initialization_00.txt 2>&1 &

done

wait