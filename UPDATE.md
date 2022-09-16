V0.2.3: Add different ratios for 3Gaussians and NBAIOT

1. Delete "if 'Centralized' in params['p2']:  # for Centralized Kmeans, ratios should have no impact."
	if 'Centralized' in params['p2']:  # for Centralized Kmeans, ratios should have no impact.
        pass
    elif 2 * ratio <= 0 or 2 * ratio >= 1:
        pass
    else:



V0.2.2: add IS_PCA and IS_REMOVE_OUTLIERS

1. Add IS_PCA and IS_REMOVE_OUTLIERS for NBAIOT
2. Add different ratios for 3GAUSSIANS and NBAIOT
3. Update collect_results.py

V0.2.1:
- Repeat for generating data with different SEED_DATA; however, model seed is fixed to 42. 
- Add weighted silhouette score and plot the silhouette score 

V0.1.1-(single_train_test_data) (20220908)
- single train and test set, but different seeds for model training.

V0.0.3 (20220908)
- Add 6 new datasets: 'BITCOIN', 'CHARFONT', 'GASSENSOR','DRYBEAN', 'SELFBACK','MNIST', 
- Add PCA for MNIST to reduce the feature dimension from 28*28=784  to ~110 (n_components = 0.95 for PCA) 


V0.0.3 (20220906)
- Modify the solution for empty clusters according to the initial centroids. 
  Assign the farthest points to the centroids that have no points (https://github.com/klebenowm/cs540/blob/master/hw1/KMeans.java)
- Update latex_plot.py 
- Turn off the email notification of sbatch due to too many notification. 
  


V0.0.3 (20220823)
- Rewrite the whole code and adjust the project structure.  
- Add config.yaml
- Add log
- Assign the farthest points to the centroids that have no points (https://github.com/klebenowm/cs540/blob/master/hw1/KMeans.java)

V0.0.3-0 (20220223)
- Modified collect_results.py and added '...-std.png' into xlsx  
- Remove the np.sqrt from euclidean_dist = np.sum(np.square(x - centroids[labels]), axis=1) in utils_stats.py.
- 