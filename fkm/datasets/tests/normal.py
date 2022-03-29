
import numpy as np
import numpy.random
from numpy.random import MT19937

for i in range(10):
	a = np.random.normal(loc=2, scale=3, size=1)
	# np.random.normal() is short for
	#   r = numpy.random.RandomState(seed=None)
	#   r.normal()

	r = numpy.random.RandomState(seed=42)
	b = r.normal(loc=2, scale=3, size=1)
	print(f'i={i}, a={a}, b={b}')

bit_generator = MT19937()
print(bit_generator)