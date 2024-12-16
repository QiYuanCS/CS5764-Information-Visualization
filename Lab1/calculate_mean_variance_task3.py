import numpy as np

np.random.seed(5764)

def Pearson_mean_variance(x, y):

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    variance_x = np.var(x, ddof = 1)
    variance_y = np.var(y, ddof = 1)

    print(f'The sample mean of random variable x: {mean_x:.2f}')
    print(f'The sample mean of random variable y: {mean_y:.2f}')
    print(f'The sample variance of random variable x: {variance_x:.2f}')
    print(f'The sample variance of random variable y: {variance_y:.2f}')


mean_x = 0
variance_x = 1
mean_y = 5
variance_y = 2
number_samples = 1000

x = np.random.normal(mean_x, np.sqrt(variance_x), number_samples)
y = np.random.normal(mean_y, np.sqrt(variance_y), number_samples)

Pearson_mean_variance(x, y)


