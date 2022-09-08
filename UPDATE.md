

# Updates: version 0.3.7 (20220908)
- Add 6 new datasets: 'BITCOIN', 'CHARFONT', 'GASSENSOR','DRYBEAN', 'SELFBACK','MNIST', 
- Add PCA for MNIST to reduce the feature dimension from 28*28=784  to ~110 (n_components = 0.95 for PCA) 


# Updates: version 0.3.6 (20220906)
- Modify the solution for empty clusters according to the initial centroids. 
  Assign the farthest points to the centroids that have no points (https://github.com/klebenowm/cs540/blob/master/hw1/KMeans.java)
- Update latex_plot.py 
- Turn off the email notification of sbatch due to too many notification. 
  


# Updates: version 0.3.5 (20220823)
- Rewrite the whole code and adjust the project structure.  
- Add config.yaml
- Add log
- Assign the farthest points to the centroids that have no points (https://github.com/klebenowm/cs540/blob/master/hw1/KMeans.java)

# Updates: version 0.3.4-2 (20220223)
- Modified collect_results.py and added '...-std.png' into xlsx  
- Remove the np.sqrt from euclidean_dist = np.sum(np.square(x - centroids[labels]), axis=1) in utils_stats.py.
- 