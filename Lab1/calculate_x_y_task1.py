import numpy as np

np.random.seed(5764)

def calculate_x_y():

    mean_x = float(input("Enter mean  or press enter for the default 0 ") or 0)
    variance_x = float(input("Enter variance or press enter for the default 1 ") or 1)
    mean_y = float(input("Enter mean y ") or 5)
    variance_y = float(input("Enter variance y ") or 2)
    num_samples = int(input("Enter number of samples "))

    x = np.random.normal(mean_x, variance_x, num_samples)
    y = np.random.normal(mean_y, variance_y, num_samples)

    print("the first three values of x: ", [f"{val:.2f}" for val in x[:3]])
    print("the first three values of y: ", [f"{val:.2f}" for val in y[:3]])

calculate_x_y()
