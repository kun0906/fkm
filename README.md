
# Dataset:
- FEMINIST: _downloaded from 'LEAF: https://leaf.cmu.edu/'_
- GAUSSIAN2: _simulated 2 clusters from 2 Gaussian distributions._

# Source code:
- fkm 
  - clustering
  - datasets
  - utils
  - results

# Execute
  ```shell
  cd fkm/fkm/
  PYTHONPATH='..' python3 Centralized_Kmeans.py
  PYTHONPATH='..' python3 Stanford_random_initialization.py
  PYTHONPATH='..' python3 Stanford_average_initialization.py
  PYTHONPATH='..' python3 Our_greedy_initialization.py  
  PYTHONPATH='..' python3 Our_greedy_center.py  
  PYTHONPATH='..' python3 Our_greedy_2K.py  
  PYTHONPATH='..' python3 Our_greedy_K_K.py  
  PYTHONPATH='..' python3 Our_greedy_concat_Ks.py
  PYTHONPATH='..' python3 Our_weighted_kmeans_initialization.py
  ```


# Updates: version 0.3.3 (20220216)
- Add 'Our_greedy_concat_Ks.py'
- Add 'Our_weighted_kmeans_initialization.py'
- Add 'save plot' in 'gaussian3.py, gaussian2.py, and gaussian5.py'
- Add 'tol' in out_dir 
- Reduce tolerance from 1e-6 to 1e-10
- 




 