
# Dataset:
- FEMINIST are downloaded from 'LEAF: https://leaf.cmu.edu/'

# Source code: 
- fkm 
  - clustering
  - datasets
  - utils
  - results

# Execute
  ```python3
  cd fkm/fkm/
  PYTHONPATH='..' python3 Centralized_Kmeans.py
  PYTHONPATH='..' python3 Stanford_random_initialization.py
  PYTHONPATH='..' python3 Stanford_average_initialization.py
  PYTHONPATH='..' python3 Our_greedy_initialization.py  
  ```