import os


def get_experiment_params(out_dir='results', p0='FEMNIST', p1='1client_1writer_multidigits',
                          p2='Centralized_random', p3='py_name', n_clusters = None, n_clients = None,
                          client_epochs=1, tolerance=1e-5, n_repeats = 5):
    """

    Parameters
    ----------
    out_dir: data_name/data_details/server_average-Clients_random/
    p0
    p1
    p2
    client_epochs:

    Returns
    -------

    """
    normalize_method = 'std'
    n_repeats = n_repeats
    # tolerance =f"{tolerance:.2e}"
    out_dir = os.path.join(out_dir, f'repeats_{n_repeats}-client_epochs_{client_epochs}-tol_{str(tolerance)}-normalize_{normalize_method}', p3)
    ##############################################################################################################
    # p0 = 'FEMNIST'
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
            n_clusters = 10 if not n_clusters else n_clusters
            data_name = f'Writers_{writer_ratio}-Testset_{data_ratio_per_writer}-Clusters_{n_clusters}-Clients_{n_clients}-{p1}'
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
            n_clusters = 10 if not n_clusters else n_clusters
            data_name = f'Writers_{writer_ratio}-Testset_{data_ratio_per_writer}-Clusters_{n_clusters}-Clients_{n_clients}-{p1}'

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
            n_clusters = 10 if not n_clusters else n_clusters
            data_name = f'Writers_{writer_ratio}-Testset_{data_ratio_per_digit}-Clusters_{n_clusters}-Clients_{n_clients}-{p1}'

        elif p1 == 'iid_data':
            # each client includes multi writers, and each client only includes 1 digit.
            """ Dataset:
                """
            DATASET = 'FEMNIST'
            writer_ratio = 0.1
            data_ratio_per_writer = 0.3
            data_ratio_per_digit = 0.3
            n_clusters = 10 if not n_clusters else n_clusters
            # data_name = f'Writers_{writer_ratio}-Testset_{data_ratio_per_digit}-Clusters_{n_clusters}-Clients_{n_clients}-{p1}'
            data_name = 'iid_data'
        elif p1 == 'femnist_user_percent':
            # each client includes multi writers, and each client only includes 1 digit.
            """ Dataset:
                """
            DATASET = 'FEMNIST'
            writer_ratio = None
            data_ratio_per_writer = None
            data_ratio_per_digit = None
            n_clusters = 62 if not n_clusters else n_clusters
            data_name = f'Clusters_{n_clusters}-Clients_{n_clients}-{p1}'
        else:
            msg = p1
            raise NotImplementedError(msg)
    elif p0 == 'NBAIOT':
        if p1 in ['nbaiot_user_percent']:
            DATASET = 'NBAIOT'
            writer_ratio = None
            data_ratio_per_writer = None
            data_ratio_per_digit = None
            n_clusters = 2 if not n_clusters else n_clusters # 1 benign + 10 attacks
            if n_clients != 2:
                raise ValueError(f'error: {n_clients}')
            data_name = f'Clusters_{n_clusters}-Clients_{n_clients}-{p1}'
        elif p1 in ['nbaiot_user_percent_client11']:
            DATASET = 'NBAIOT'
            writer_ratio = None
            data_ratio_per_writer = None
            data_ratio_per_digit = None
            n_clusters = 11 if not n_clusters else n_clusters # 1 benign + 10 attacks
            if n_clients != 11:
                raise ValueError(f'error: {n_clients}')
            data_name = f'Clusters_{n_clusters}-Clients_{n_clients}-{p1}'

    elif p0 == 'SENT140':
        if p1 == 'sent140_user_percent':
            DATASET = 'SENT140'
            writer_ratio = None
            data_ratio_per_writer = None
            data_ratio_per_digit = None
            n_clusters = 2 if not n_clusters else n_clusters # 1 benign + 10 attacks
            data_name = f'Clusters_{n_clusters}-Clients_{n_clients}-{p1}'
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
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        elif p1.split(':')[-1] == 'mix_clusters_per_client':

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
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'

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
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
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
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
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
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        elif p1.split(':')[-1] == 'diff_sigma_n':
            """ Dataset: 
                1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                    cluster 1: sigma = 0.1 and n_points = 5000
                    cluster 2: sigma = 1    and n_points = 15000
                2) Each client only includes one cluster, i.e., we have 2 clients.
                3) For each client, we choose 70% data points for training and 30% data points for testing. 

            """
            DATASET = '2GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 2
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'

        elif p1 == '1client_xlt0_2':
            """ Dataset: 
                1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                    cluster 1: sigma = 0.01 and n_points = 5000
                    cluster 2: sigma = 0.01   and n_points = 15000
                2) Each client only includes one cluster, i.e., we have 2 clients.
                3) For each client, we choose 70% data points for training and 30% data points for testing. 

            """
            DATASET = '2GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 2
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        else:
            return {}

    elif p0 == '3GAUSSIANS':
        writer_ratio = None
        data_ratio_per_writer = None
        data_ratio_per_digit = None

        if p1 == '1client_1cluster':
            """ Dataset: 
                1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                2) Each client only includes one cluster, i.e., we have 2 clients.
                3) For each client, we choose 70% data points for training and 30% data points for testing. 

            """
            DATASET = '3GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 3
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        elif p1.split(':')[-1] == 'mix_clusters_per_client':

            """ Dataset: 
                        1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points. 
                        2) client1 has 70% data from cluster 1 and 30% data from cluster2
                        3) client2 has 30% data from cluster 1 and 70% data from cluster2
                    """
            DATASET = '3GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 3
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'

        elif p1 == '1client_ylt0':
            # lt0 means all 'y's are larger than 0
            """ Dataset: 
                            1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                            2) client 1 has all data (y>0) from cluster1 and cluster2
                            3) client 2 has all data (y<=0) from cluster1 and cluster2
                            4) For each client, we choose 70% data points for training and 30% data points for testing. 

                        """
            DATASET = '3GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 3
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        elif p1 == '1client_xlt0':
            # lt0 means all 'x's are larger than 0
            """ Dataset: 
                            1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                            2) client 1 has all data (x>0) from cluster1 and cluster2
                            3) client 2 has all data (x<=0) from cluster1 and cluster2
                            4) For each client, we choose 70% data points for training and 30% data points for testing. 

                        """
            DATASET = '3GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 3
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        elif p1 == '1client_1cluster_diff_sigma':
            """ Dataset: 
                1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                    cluster 1: sigma = 0.5
                    cluster 2: sigma = 1
                2) Each client only includes one cluster, i.e., we have 2 clients.
                3) For each client, we choose 70% data points for training and 30% data points for testing. 

            """
            DATASET = '3GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 3
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        elif p1.split(':')[-1] == 'diff_sigma_n':
            """ Dataset: 
                1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                    cluster 1: sigma = 0.1 and n_points = 5000
                    cluster 2: sigma = 1    and n_points = 15000
                2) Each client only includes one cluster, i.e., we have 2 clients.
                3) For each client, we choose 70% data points for training and 30% data points for testing. 

            """
            DATASET = '3GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 3
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'

        elif p1 == '1client_xlt0_2':
            """ Dataset: 
                1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                    cluster 1: sigma = 0.01 and n_points = 5000
                    cluster 2: sigma = 0.01   and n_points = 15000
                2) Each client only includes one cluster, i.e., we have 2 clients.
                3) For each client, we choose 70% data points for training and 30% data points for testing. 

            """
            DATASET = '3GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 3
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        else:
            return {}

    elif p0 == '4GAUSSIANS':
        writer_ratio = None
        data_ratio_per_writer = None
        data_ratio_per_digit = None

        if p1.split(':')[-1] == 'diff_sigma_n':
            """ Dataset: 
			"""
            DATASET = '4GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 2
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        else:
            return {}

    elif p0 == '5GAUSSIANS':
        writer_ratio = None
        data_ratio_per_writer = None
        data_ratio_per_digit = None

        if p1 == '5clients_5clusters':
            """ Dataset: 

            """
            DATASET = '5GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 5
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        elif p1 == '5clients_4clusters':
            """ Dataset:

            """
            DATASET = '5GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 4
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'
        elif p1 == '5clients_3clusters':
            """ Dataset:

            """
            DATASET = '5GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 3
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'

    elif p0 == '10GAUSSIANS':
        writer_ratio = None
        data_ratio_per_writer = None
        data_ratio_per_digit = None
        if p1.split(':')[-1] == 'diff_sigma_n':
            """ Dataset: 

            """
            DATASET = '10GAUSSIANS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            # n_clusters = 10
            # n_clusters = 10   # for case 4
            n_clusters = 3      # for case 5
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'

    elif p0 == '3GAUSSIANS-ADVERSARIAL':
        writer_ratio = None
        data_ratio_per_writer = None
        data_ratio_per_digit = None
        if p1.split(':')[-1] == 'diff_sigma_n':
            """ Dataset: 

            """
            DATASET = '3GAUSSIANS-ADVERSARIAL'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            # n_clusters = 10
            n_clusters = 3
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'


    elif p0 == '2MOONS':
        writer_ratio = None
        data_ratio_per_writer = None
        data_ratio_per_digit = None
        if p1 == '2moons':
            """ Dataset: 
                1) We generate 2 clusters ((-1,0), (1, 0)) in R^2, and each of them has 10000 data points.
                    cluster 1: sigma = 0.01 and n_points = 5000
                    cluster 2: sigma = 0.01   and n_points = 15000
                2) Each client only includes one cluster, i.e., we have 2 clients.
                3) For each client, we choose 70% data points for training and 30% data points for testing. 

            """
            DATASET = '2MOONS'
            # writer_ratio = 0.1
            test_size = 0.3
            # data_ratio_per_digit = None
            n_clusters = 2
            data_name = f'{p1}-Testset_{test_size}-Clusters_{n_clusters}-Clients_{n_clients}'

    else:
        raise NotImplementedError(f'{p0}, {p1}')

    ##############################################################################################################
    # p2 = 'Centralized_random'
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

    elif p2 == 'Federated-Server_random_min_max':
        """
            4) We use the federated kmeans. 
                a) First randomly initialize server' centroids, 
                b) Then broadcast the server's centroids to each client.
        """
        server_init_centroids = 'random_min_max'
        client_init_centroids = None
        is_federated = True
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Federated-Server_{server_init_centroids}-' \
                  f'Client_{client_init_centroids}'

    elif p2 == 'Federated-Server_gaussian':
        """
        """
        server_init_centroids = 'gaussian'
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

    elif p2 == 'Federated-Server_sorted-Client_true':
        """
            4) We use the federated kmeans. 
                a) First initialize each client's centroids with true centroids, 
                b) Then use the sorted method to choose server's centroids from all clients' centroids. 
        """
        server_init_centroids = 'sorted'
        client_init_centroids = 'true'
        is_federated = True
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Federated-Server_{server_init_centroids}-' \
                  f'Client_{client_init_centroids}'

    elif p2 == 'Federated-Server_sorted-Client_random':
        """
            4) We use the federated kmeans. 
                a) First randomly initialize each client's centroids, 
                b) Then use the sorted method to choose server's centroids from all clients' centroids. 
        """
        server_init_centroids = 'sorted'
        client_init_centroids = 'random'
        is_federated = True
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Federated-Server_{server_init_centroids}-' \
                  f'Client_{client_init_centroids}'

    elif p2 == 'Federated-Server_sorted-Client_kmeans++':
        """
            4) We use the federated kmeans. 
                a) First initialize each client's centroids with kmeans++, 
                b) Then use the sorted method to choose server's centroids from all clients' centroids. 
        """
        server_init_centroids = 'sorted'
        client_init_centroids = 'random'
        is_federated = True
        out_dir = f'{out_dir}/{DATASET}/{data_name}/Federated-Server_{server_init_centroids}-' \
                  f'Client_{client_init_centroids}'

    else:
        msg = p0 + p1 + p2
        raise NotImplementedError(msg)

    params = {'p0': p0, 'p1': p1, 'p2': p2, 'n_repeats': n_repeats, 'client_epochs': client_epochs,
              'tolerance': tolerance, 'normalize_method': normalize_method,
              'is_crop_image': True, 'image_shape': (28, 28),  # For FEMNIST, crop 28x28 to 14x14
              'data_name': data_name,
              'writer_ratio': writer_ratio, 'data_ratio_per_writer': data_ratio_per_writer,
              'data_ratio_per_digit': data_ratio_per_digit if data_ratio_per_digit is not None else None,
              'n_clusters': n_clusters,
              'is_federated': is_federated,
              'n_clients': n_clients,
              'server_init_centroids': server_init_centroids, 'client_init_centroids': client_init_centroids,
              'out_dir': out_dir, 'show_title': True,
              'n_consecutive': 5,
              'splits': ['train', 'test']  # ['train', 'test']
              }

    return params
