import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(0,1, 1000)
plt.hist(x)

y = input('Enter a numner: ')
print(int(y)**2)
