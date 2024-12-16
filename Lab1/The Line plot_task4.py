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
plt.plot(x, label='Variable x')
plt.plot(y, label='Variable y')
plt.title('The Line Plot of Random Variables x and y ')
plt.xlabel('number_samples')
plt.ylabel('Value')
plt.grid()
plt.legend()
plt.show()


