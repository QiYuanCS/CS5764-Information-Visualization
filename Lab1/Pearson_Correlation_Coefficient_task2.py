import numpy as np
np.random.seed(5764)
#%%
def Pearson_Correlation_Coefficient(x, y):

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    if denominator == 0:
        return 0
    return numerator / denominator


mean_x = 0
variance_x = 1
mean_y = 5
variance_y = 2
number_samples = 1000
x = np.random.normal(mean_x, np.sqrt(variance_x), number_samples)
y = np.random.normal(mean_y, np.sqrt(variance_y), number_samples)
r = Pearson_Correlation_Coefficient(x, y)
print(f"Pearson correlation and coefficient is: {r:.2f}")




