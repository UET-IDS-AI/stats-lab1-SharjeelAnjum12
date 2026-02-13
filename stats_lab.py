import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.normal(0, 1, n)
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Normal Distribution Histogram (0,1)")
    plt.show()
    return data


def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.uniform(0, 10, n)
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Uniform Distribution Histogram (0,10)")
    plt.show()
    return data


def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.binomial(1, 0.5, n)
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Bernoulli Distribution Histogram (0.5)")
    plt.show()
    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    total = 0
    n = len(data)
    if n == 0:
        return None
    for x in data:
        total += x
    return total / n


def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    n = len(data)
    if n < 2:
        return None  # Cannot compute variance for n < 2
    mean = sample_mean(data)
    total = 0
    for x in data:
        total += (x - mean) ** 2
    return total / (n - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - 25th percentile (Q1)
    - 75th percentile (Q3)

    Use a consistent quartile definition. The tests for the fixed
    dataset [5,1,3,2,4] expect Q1=2 and Q3=4.
    """
    if len(data) == 0:
        return None, None, None, None, None
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    
    mid = n // 2
    median = sorted_data[mid]
    
    lower_half = sorted_data[:mid]
    upper_half = sorted_data[mid+1:]
    
    q1 = lower_half[-1]
    q3 = upper_half[0]
    
    return minimum, maximum, median, q1, q3


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    if len(x) != len(y):
        raise ValueError("Inputs must have the same length")
    
    n = len(x)
    if n < 2:
        return None  # Cannot compute covariance for n < 2
    
    mean_x = sample_mean(x)
    mean_y = sample_mean(y)
    
    total = 0
    for xi, yi in zip(x, y):
        total += (xi - mean_x) * (yi - mean_y)
    
    return total / (n - 1)


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    if len(x) != len(y):
        raise ValueError("Inputs x and y must have the same length")
    
    if len(x) < 2:
        return None  # Cannot compute covariance matrix for n < 2
    
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)
    
    return np.array([[var_x, cov_xy], 
                     [cov_xy, var_y]])
