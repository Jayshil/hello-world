import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

x = np.random.normal(0., 10., 100000)

plt.hist(x, bins=1000)
plt.show()