import numpy as np

for i in range(5):
    print()
    np.random.seed(i)
    for iteration in range(1, 1 + 10):
        # clients_in_round = random.sample(x, clients_per_round) # without replacement and random
        # r=np.random.RandomState(iteration)
        r = np.random.RandomState((i+1)*iteration)
        clients_in_round = r.choice(range(1, 100), size=3, replace=False)
        print(iteration, clients_in_round)
    # print(np.random.get_state())






