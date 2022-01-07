import os


def get_experiment_params(out_dir='results', p0='FEMNIST', p1='1client_1writer_multidigits',
                          p2='Centralized_random'):
    """

    Parameters
    ----------
    out_dir: data_name/data_details/server_average-Clients_random/
    p0
    p1
    p2

    Returns
    -------

    """
    repeats = 30 if '2GAUSSIANS' == p0 else 10
    client_epochs = 5
    out_dir = os.path.join(out_dir, f'repeats_{repeats}-client_epochs_{client_epochs}')
    ##############################################################################################################
    ### p0 = 'FEMNIST'
    if p0 == 'FEMNIST':
        if p1 == '1client_1writer_multidigits':
            # each client only includes one writer, and each writer includes 10 digits.
            """ Dataset: 
                1) We randomly select 10% writers from FEMNIST, i.e., 0.1* 3600 = 360.
                2) Each client only includes one writer, i.e., we have 360 clients.
                3) For each writer, we choose 70% data points for training and 30% data points for testing. 
        
            """
            DATASET = 'FEMNIST'
            writer_ratio = 0.1
            data_ratio_per_writer = 0.3
            data_ratio_per_digit = None
            n_clusters = 10
            n_clients = None
            data_name = f'{DATASET}-Writers_{writer_ratio}-Testset_{data_ratio_per_writer}-Clusters_{n_clusters}-Clients_{n_clients}-{p1}'
        elif p1 == '1client_multiwriters_multidigits':
            # each client includes multiwriters, and each writer includes 10 digits.
            """ Dataset:
                    1) We randomly select 10% writers from FEMNIST, i.e., 0.1* 3600 = 360.
                    2) Divide the selected writers into 20 groups. Each group is a client, i.e., we have 20 clients.
                    3) For each writer among a group, we choose 70% data points for training and 30% data points for testing. 
    
                """
            DATASET = 'FEMNIST'
            writer_ratio = 0.1
            data_ratio_per_writer = 0.3
            data_ratio_per_digit = None
            n_clusters = 10
            n_clients = 20
            data_name = f'{DATASET}-Writers_{writer_ratio}-Testset_{data_ratio_per_writer}-Clusters_{n_clusters}-Clients_{n_clients}-{p1}'

        elif p1 == '1client_multiwriters_1digit':
            # each client includes multi writers, and each client only includes 1 digit.
            """ Dataset:
                    1) We randomly select 10% writers from FEMNIST, i.e., 0.1* 3600 = 360.
                    2) Divide the selected writers' data points into 10 groups. Each group is a client and only includes one digit, i.e., we have 10 clients.
                    3) For each client, we choose 70% data points for training and 30% data points for testing. 
    
                """
            DATASET = 'FEMNIST'
            writer_ratio = 0.1
            data_ratio_per_writer = 0.3
            data_ratio_per_digit = 0.3
            n_clusters = 10
            n_clients = 10
            data_name = f'{DATASET}-Writers_{writer_ratio}-Testset_{data_ratio_per_digit}-Clusters_{n_clusters}-Clients_{n_clients}-{p1}'
        else:
            return {}

    elif p0 == '2GAUSSIANS':
        writer_ratio = None
        data_ratio_per_writer = None
        data_ratio_per_digit = None

        if p1 == '1client_1cluster':
            """ Dataset: 
                1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                2) Each client only includes one cluster, i.e., we have 2 clients.
                3) For each client, we choose 70% data points for training and 30% data points for testing. 

            """
            DATASET = '2GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 2
            n_clients = 2
            data_name = f'{DATASET}-{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        elif p1 == '1client_0.7cluster1_0.3cluster2':

            """ Dataset: 
                        1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points. 
                        2) client1 has 70% data from cluster 1 and 30% data from cluster2
                        3) client2 has 30% data from cluster 1 and 70% data from cluster2
                    """
            DATASET = '2GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 2
            n_clients = 2
            data_name = f'{DATASET}-{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'

        elif p1 == '1client_ylt0':
            # lt0 means all 'y's are larger than 0
            """ Dataset: 
                            1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                            2) client 1 has all data (y>0) from cluster1 and cluster2
                            3) client 2 has all data (y<=0) from cluster1 and cluster2
                            4) For each client, we choose 70% data points for training and 30% data points for testing. 

                        """
            DATASET = '2GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 2
            n_clients = 2
            data_name = f'{DATASET}-{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        elif p1 == '1client_xlt0':
            # lt0 means all 'x's are larger than 0
            """ Dataset: 
                            1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                            2) client 1 has all data (x>0) from cluster1 and cluster2
                            3) client 2 has all data (x<=0) from cluster1 and cluster2
                            4) For each client, we choose 70% data points for training and 30% data points for testing. 

                        """
            DATASET = '2GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 2
            n_clients = 2
            data_name = f'{DATASET}-{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        elif p1 == '1client_1cluster_diff_sigma':
            """ Dataset: 
                1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                    cluster 1: sigma = 0.5
                    cluster 2: sigma = 1
                2) Each client only includes one cluster, i.e., we have 2 clients.
                3) For each client, we choose 70% data points for training and 30% data points for testing. 

            """
            DATASET = '2GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 2
            n_clients = 2
            data_name = f'{DATASET}-{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        elif p1 == '1client_1cluster_diff_sigma_n':
            """ Dataset: 
                1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                    cluster 1: sigma = 0.5 and n_points = 5000
                    cluster 2: sigma = 1    and n_points = 15000
                2) Each client only includes one cluster, i.e., we have 2 clients.
                3) For each client, we choose 70% data points for training and 30% data points for testing. 

            """
            DATASET = '2GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 2
            n_clients = 2
            data_name = f'{DATASET}-{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        else:
            return {}
    else:
        raise NotImplementedError(f'{p0}, {p1}')

    ##############################################################################################################
    ### p2 = 'Centralized_random'
    if p2 == 'Centralized_random':
        """
            4) We use the centralized kmeans, 
                i.e., collect all the data to the server and then use the centralized kmeans. 
                a) Randomly initialize centroids, 
        """
        server_init_centroids = 'random'
        client_init_centroids = None
        is_federated = False
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Centralized-{server_init_centroids}'

    elif p2 == 'Centralized_kmeans++':
        """
            4) We use the centralized kmeans. 
                b) Initialize centroids with kmeans++. 
        """
        server_init_centroids = 'kmeans++'
        client_init_centroids = None
        is_federated = False
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Centralized-{server_init_centroids}'

    elif p2 == 'Centralized_true':
        """
            4) We use the centralized kmeans. 
                b) Initialize centroids with true centroids. 
        """
        server_init_centroids = 'true'
        client_init_centroids = None
        is_federated = False
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Centralized-{server_init_centroids}'


    elif p2 == 'Federated-Server_random':
        """
            4) We use the federated kmeans. 
                a) First randomly initialize server' centroids, 
                b) Then broadcast the server's centroids to each client.
        """
        server_init_centroids = 'random'
        client_init_centroids = None
        is_federated = True
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Federated-Server_{server_init_centroids}-' \
                  f'Client_{client_init_centroids}'

    elif p2 == 'Federated-Server_true':
        """
            4) We use the federated kmeans. 
                a) First initialize server' centroids with true centroids, 
                b) Then broadcast the server's centroids to each client.
        """
        server_init_centroids = 'true'
        client_init_centroids = None
        is_federated = True
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Federated-Server_{server_init_centroids}-' \
                  f'Client_{client_init_centroids}'

    elif p2 == 'Federated-Server_average-Client_random':
        """
            4) We use the federated kmeans. 
                a) First randomly initialize each client's centroids, 
                b) Then average clients' centroids as server's centroids. 
        """
        server_init_centroids = 'average'
        client_init_centroids = 'random'
        is_federated = True
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Federated-Server_{server_init_centroids}-' \
                  f'Client_{client_init_centroids}'
    elif p2 == 'Federated-Server_average-Client_kmeans++':
        """
            4) We use the federated kmeans. 
                a) First initialize each client's centroids with Kmeans++, 
                b) Then average clients' centroids as server's centroids. 
        """
        server_init_centroids = 'average'
        client_init_centroids = 'kmeans++'
        is_federated = True
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Federated-Server_{server_init_centroids}-' \
                  f'Client_{client_init_centroids}'

    elif p2 == 'Federated-Server_average-Client_true':
        """
            4) We use the federated kmeans. 
                a) First initialize each client's centroids with true centroids, 
                b) Then average clients' centroids as server's centroids. 
        """
        server_init_centroids = 'average'
        client_init_centroids = 'true'
        is_federated = True
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Federated-Server_{server_init_centroids}-' \
                  f'Client_{client_init_centroids}'

    elif p2 == 'Federated-Server_greedy-Client_random':
        """
            4) We use the federated kmeans. 
                a) First randomly initialize each client's centroids, 
                b) Then average clients' centroids as server's centroids. 
        """
        server_init_centroids = 'greedy'
        client_init_centroids = 'random'
        is_federated = True
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Federated-Server_{server_init_centroids}-' \
                  f'Client_{client_init_centroids}'
    elif p2 == 'Federated-Server_greedy-Client_kmeans++':
        """
            4) We use the federated kmeans. 
                a) First initialize each client's centroids with Kmeans++, 
                b) Then greedy choose server's centroids from all clients' centroids. 
        """
        server_init_centroids = 'greedy'
        client_init_centroids = 'kmeans++'
        is_federated = True
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Federated-Server_{server_init_centroids}-' \
                  f'Client_{client_init_centroids}'

    elif p2 == 'Federated-Server_greedy-Client_true':
        """
            4) We use the federated kmeans. 
                a) First initialize each client's centroids with true centroids, 
                b) Then greedy choose server's centroids from all clients' centroids. 
        """
        server_init_centroids = 'greedy'
        client_init_centroids = 'true'
        is_federated = True
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Federated-Server_{server_init_centroids}-' \
                  f'Client_{client_init_centroids}'

    else:
        return {}

    params = {'p0': p0, 'p1': p1, 'p2': p2, 'repeats': repeats,'client_epochs': client_epochs,
              'is_crop_image': True, 'image_shape': (14, 14),  # For FEMNIST, crop 28x28 to 14x14
              'data_name': data_name,
              'writer_ratio': writer_ratio, 'data_ratio_per_writer': data_ratio_per_writer,
              'data_ratio_per_digit': data_ratio_per_digit if data_ratio_per_digit is not None else None,
              'n_clusters': n_clusters,
              'is_federated': is_federated,
              'n_clients': n_clients,
              'server_init_centroids': server_init_centroids, 'client_init_centroids': client_init_centroids,
              'out_dir': out_dir,
              }

    return params
