from KDEpy import FFTKDE, TreeKDE, NaiveKDE
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
import pdb
from scipy.spatial import cKDTree
import typing

def calculate_kde(data, weights, bin_width, bandwidth: typing.Optional[float], max_x: typing.Optional[int], max_y: typing.Optional[int]):
    #print(f"max_x = {max_x}, max_y = {max_y}")
    # class KDEpyWrapper(BaseEstimator):
    #    def __init__(self, bandwidth):
    #        self.bandwidth = bandwidth
    #        self.kde = None
    #
    #    def fit(self, X, y=None, sample_weight=None):
    #        self.kde = TreeKDE(kernel='gaussian', bw=self.bandwidth, norm=2)
    #        self.kde.fit(X, weights=sample_weight)
    #        return self
    #
    #    def score(self, X, y=None):
    #        # Compute the density estimates for the input data
    #        density_estimates = self.kde.evaluate(X, eps=0.001)
    #        # Compute the log likelihood of the density estimates
    #        log_likelihood = np.log(density_estimates).sum()
    #        # Return the mean log likelihood
    #        return log_likelihood / len(density_estimates)
    #
    #    def get_params(self, deep=True):
    #        return {"bandwidth": self.bandwidth}
    #
    #    def set_params(self, **params):
    #        if "bandwidth" in params:
    #            self.bandwidth = params["bandwidth"]
    #        return self
    #
    ## Define the bandwidth grid
    # bandwidths = np.logspace(-1, 1, 19)
    #
    ## Initialize variables to store results
    # best_bandwidth = None
    # best_score = float('-inf')  # Update this based on your evaluation metric
    #
    ## Perform grid search
    # for bandwidth in bandwidths:
    #    # Create KDE instance with the current bandwidth
    #    kde = KDEpyWrapper(bandwidth)
    #
    #    # Compute scores using cross-validation
    #    scores = cross_val_score(kde, data, cv=5, fit_params={'sample_weight': weights})
    #
    #    # Compute mean score (you can use other aggregation functions)
    #    mean_score = np.mean(scores)
    #
    #    # Update best bandwidth if current bandwidth yields better performance
    #    if mean_score > best_score:
    #        best_score = mean_score
    #        best_bandwidth = bandwidth
    weights = np.ravel(weights)
    err_num = 0

    if len(data) != len(weights):
        err_num = 1

    half_bin_width = bin_width / 2.0

    if max_x is None:
        max_x = ((max(data[:, 0]) + bin_width) // bin_width) * bin_width
    else:
        max_x = (max_x + bin_width // bin_width) * bin_width

    if max_y is None:
        max_y = ((max(data[:, 1]) + bin_width) // bin_width) * bin_width
    else:
        max_y = (max_y + bin_width // bin_width) * bin_width

    num_points_x = (max_x / bin_width) + 1
    num_points_y = (max_y / bin_width) + 1

    positions = (
        (
            np.mgrid[0 : max_x : num_points_x * 1j, 0 : max_y : num_points_y * 1j]
            + half_bin_width
        )
        .reshape(2, -1)
        .T
    )
    # num_points_x = max(data[:,0]) - min(data[:,0]) + 3
    # num_points_y = max(data[:,1]) - min(data[:,1]) + 3
    # xx, yy = np.mgrid[xmin:xmax:num_points_x*1j, ymin:ymax:num_points_y*1j]
    # positions = np.vstack([xx.ravel(), yy.ravel()]).T
    # print(positions)

    # print("Best bandwidth:", best_bandwidth)
    best_bandwidth = 1.0 if bandwidth is None else bandwidth

    # Compute the kernel density estimate
    kde = NaiveKDE(kernel="gaussian", bw=best_bandwidth, norm=2)
    # pdb.set_trace()
    y = kde.fit(data, weights=weights).evaluate(positions)
    y /= y.sum()
    # total_sum = y.sum()
    # print("kde length: {}".format(len(y)))
    # print("python kde matrix: {}".format(y))
    # print("kde sum: {}".format(total_sum))

    # find the data indices in the positions vector
    tree = cKDTree(positions)
    distances, filtered_indices = tree.query(data, k=1)
    # print(distances)
    # threshold_distance = bin_width
    # filtered_indices = indices[distances < threshold_distance]

    if len(data) != len(filtered_indices):
        err_num = 2
    # print("error_number")
    # print(err_num)

    output = y[filtered_indices]
    # print("data length: {}".format(len(data)))
    # print("weight length: {}".format(len(weights)))
    # print("prob length: {}".format(len(output)))
    # print("output:")
    # print(output)

    return output
