# Compare pair-wise Euclidean distances and covariance
# 2017-03-22 jkang

import numpy as np
from scipy.spatial.distance import pdist

# data shape = (feature) x (example)
data = np.random.randint(5, size=(4, 3))
# cdata = # centering data

# pairwise Euclidean distance (sum)
p_dist_sum = np.sum(pdist(data))

# covariance (sum)
c_dist = np.cov(data.T)
u_triangle = np.triu(c_dist, k=1)
c_dist_sum = np.sum(u_triangle)
