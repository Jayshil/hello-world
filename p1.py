import numpy as np
import matplotlib.pyplot as plt

print('This is my first Github repository')

x = np.random.normal(0,0.5,10000)
y = np.random.normal(1,0.2,10000)

plt.hist(x,bins=500)
plt.hist(y,bins=500)
plt.show()
