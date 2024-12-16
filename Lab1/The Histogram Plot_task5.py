import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5764)

mean_x = 0
variance_x = 1
mean_y = 5
variance_y = 2
number_samples = 1000

x = np.random.normal(mean_x, np.sqrt(variance_x), number_samples)
y = np.random.normal(mean_y, np.sqrt(variance_y), number_samples)

plt.figure()
plt.hist(x, bins=25, alpha=0.5, label='Variable x', color='blue')
plt.hist(y, bins=25, alpha=0.5, label='Variable y', color='red')
plt.title('The Histogram Plot of Random Variables x and y ')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid()
plt.legend()
plt.show()


