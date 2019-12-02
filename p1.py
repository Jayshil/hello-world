import numpy as np
import matplotlib.pyplot as plt

print('This is my first Github repository')

x = np.random.normal(0, 0.5, 10000)
y = x**2

plt.plot(x)
plt.savefig('x.pdf')
plt.plot(y)
plt.savefig('y.pdf')
