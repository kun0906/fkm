#!/bin/bash

##cd fkm/fkm
for no in '00' '10' '20'; do
  PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 Stanford_random_initialization.py -n $no >~tmp/random_$no.txt 2>&1 &
done

for no in '00' '01' '10' '11' '20' '21'; do
  PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 Stanford_average_initialization.py -n $no >~tmp/average_$no.txt 2>&1 &
done

for no in '00' '01' '10' '11' '20' '21'; do
  PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 Our_greedy_initialization.py -n $no >~tmp/greedy_$no.txt 2>&1 &
done