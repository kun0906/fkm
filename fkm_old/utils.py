import pickle
from time import time


def distance(a, b):
    return sum((x1 - x2) ** 2 for x1, x2 in zip(a, b))


def load(in_file):
    with open(in_file, 'rb') as f:
        data = pickle.load(f)
    return data


def dump(data, out_file):
    with open(out_file, 'wb') as out:
        pickle.dump(data, out)


def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func
