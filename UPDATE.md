V0.2.8: add the evolution of centroids. 

1. add main_single_demo.py to generate single result.
2. add plot_evolution_centroids.py


V0.2.7: Add different size of 3Gaussians

1. Add n1 = [50, 100, 500, 1000, 2000, 3000]:
2. Update the collect_results.py and collect_table_results.py


V0.2.6: Add different Ks and corresponding collecting functions

1. Add different Ks e.g., [1, 2,3, ..., 10]
2. Correct the shape error. 
   i.e., if self.centroids.shape != self.true_centroids[split].shape:
         continue
3. Add collect_K_results and collect_table_K_results. 


V0.2.5: Add latex_plot_datasets for synthetic datasets

1. Add score_comparison.py for DB score and Silhouette score and ARI
2. Add latex_plot datasets for synthetic datasets


V0.2.4: Generate dataset for each algorithm

1. Generate dataset for each algorithm for safe even it's very slow. 
2. Use sample_fraction = 1 for all experiments
3. Add ignore.py and still need to double-check.

V0.2.3-5.1: Print each client data distribution for debugging. 


V0.2.3-4: Modify the way to sample the data for each client and change sample_fraction to 1.0. 

1. Modify the way to sample the data for all four datasets (only train without test) 
2. Add ratio for MNIST.
3. Change sample_fraction = 1 from 0.5. 


V0.2.3-3: Reduce the evaluation metrics for saving time

1. Reduce the evaluation metrics for saving time, especifically for silhouette plot and add plot.close() 
2. Add centralized_minibatch_kmeans.py (not finished yet)



V0.2.3-2: Add VGG_16 for MNIST and address the generated data effected by ratios and train_test_spilt.

1. Centralized kmeans should not be effected by ratios
2. Modify the ways to obtain test set, which could be an issue for the dataset caused by train_test_spilt
   E.g., train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=2, shuffle=True, stratify=y,
                                                            random_state=random_state) 
        where one class is 503 data points, and another one is 505; however, we expect both is 504.
3. Add vgg_16 for MNIST and reduce the dimension for 28*28 to 100 
4. Update main_all.py for MNIST 



V0.2.3-1: Add different ratios for 3Gaussians and NBAIOT

1. Delete "if 'Centralized' in params['p2']:  # for Centralized Kmeans, ratios should have no impact."
    if 'Centralized' in params['p2']:  # for Centralized Kmeans, ratios should have no impact.
        pass
    elif 2 * ratio <= 0 or 2 * ratio >= 1:
        pass
    else:
2. Add the setting for ['DRYBEAN', 'SELFBACK', 'GASSENSOR', 'MNIST'] and fix some issues in the setting. 


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