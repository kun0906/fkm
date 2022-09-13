### A Python project for comparing the proposed greedy centroid initialization with other initialization methods of federated K-means, which includes a centralized/traditional K-means (CKM) and four federated K-means (FKM).    

## Environment requirement
  - Conda 4.10.3 # Conda -V
  - Python 3.9.7 # Python3 -V
  - Pip3 21.2.4 # Pip3 -V
`
## Installation instruction
  - pip3 install -r requirements.txt
  
## Source code:
- fkm 
  - datasets
  - clustering
  - utils
  - results

## Dataset:
- FEMINIST: _downloaded from 'LEAF: https://leaf.cmu.edu/'_
- GAUSSIAN2: _simulated 2 clusters from 2 Gaussian distributions._
- GAUSSIAN3: _simulated 3 clusters from 3 Gaussian distributions._
- GAUSSIAN5: _simulated 5 clusters from 5 Gaussian distributions._

## Execution
  ```shell
  cd fkm/fkm/
  PYTHONPATH='..' python3 centralized_kmeans.py
  PYTHONPATH='..' python3 federated_server_init_first.py
  PYTHONPATH='..' python3 federated_client_init_first.py
  PYTHONPATH='..' python3 federated_greedy_kmeans.py  
#  PYTHONPATH='..' python3 Our_greedy_center.py  
#  PYTHONPATH='..' python3 Our_greedy_2K.py  
#  PYTHONPATH='..' python3 Our_greedy_K_K.py  
#  PYTHONPATH='..' python3 Our_greedy_concat_Ks.py
#  PYTHONPATH='..' python3 Our_weighted_kmeans_initialization.py
  ```



[//]: # ()
[//]: # (# Author: Kun)

[//]: # (# Email: kun.bj@outlook.com)