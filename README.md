
# Dataset:
- FEMINIST: _downloaded from 'LEAF: https://leaf.cmu.edu/'_
- GAUSSIAN2: _simulated 2 clusters from 2 Gaussian distributions._
- GAUSSIAN3: _simulated 3 clusters from 3 Gaussian distributions._
- GAUSSIAN5: _simulated 5 clusters from 5 Gaussian distributions._
- 
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
  PYTHONPATH='..' python3 Stanford_server_random_initialization.py
  PYTHONPATH='..' python3 Stanford_client_initialization.py
  PYTHONPATH='..' python3 Our_greedy_initialization.py  
  PYTHONPATH='..' python3 Our_greedy_center.py  
  PYTHONPATH='..' python3 Our_greedy_2K.py  
  PYTHONPATH='..' python3 Our_greedy_K_K.py  
  PYTHONPATH='..' python3 Our_greedy_concat_Ks.py
  PYTHONPATH='..' python3 Our_weighted_kmeans_initialization.py
  ```


# Updates: version 0.3.4-2 (20220223)
- Modified collect_results.py and added '...-std.png' into xlsx  
- Remove the np.sqrt from euclidean_dist = np.sum(np.square(x - centroids[labels]), axis=1) in utils_stats.py.
- 