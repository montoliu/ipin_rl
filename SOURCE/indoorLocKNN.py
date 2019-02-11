import numpy as np
import math
from scipy.spatial import distance


# ----------------------------------
# IndoorLoc class
# Get the location of a set of samples using a knn-based algorithm
# ----------------------------------
class IndoorLocKNN:
    # ---------------------------------------------------------
    # Class constructor
    # train_fingerprints is a np array of n_train_samples x n_aps.
    #    Each element i,j is the RSSI in the i-th location from the j-th AP
    # train_locations is a np array of train_samples x 2.
    #    Each element i is the coordinates x,y of i-th train fingerprint
    # If floor_estimation is true (and building_estimation = false),
    #    then train_locations is a np array of train_samples x 3: x,y,floor
    # If building_estimation is true, (and floor_estimation = false)
    #    then train_locations is a np array of train_samples x 3: x,y,building
    # If both, floor_estimation and building_estimation are true,
    #    then train_locations is a np array of train_samples x 4: x,y,floor,building
    # k is the number of neighbors in the knn algorithm
    # ---------------------------------------------------------
    def __init__(self, train_fingerprints, train_locations, k=3, floor_estimation=False, building_estimation=False):
        self.train_fingerprints = train_fingerprints
        self.train_locations = train_locations
        self.n_training_samples = train_fingerprints.shape[0]
        self.n_aps = train_fingerprints.shape[1]
        self.k = k
        self.floor_estimation = floor_estimation
        self.building_estimation = building_estimation

        if floor_estimation:
            self.n_floors = int(np.max(self.train_locations[:, 2]) + 1)
            if building_estimation:
                self.n_buildings = int(np.max(self.train_locations[:, 3]) + 1)
            else:
                self.n_buildings = 0
        else:
            self.n_floors = 0
            if building_estimation:
                self.n_buildings = int(np.max(self.train_locations[:, 2]) + 1)
            else:
                self.n_buildings = 0

    # ---------------------------------------------------------
    # get_accuracy
    # ---------------------------------------------------------
    # Get statistics of the accuracy of a set of test fingerprints
    # ---------------------------------------------------------
    def get_accuracy(self, test_fingerprints, test_locations):
        estimated_locations = self.get_locations(test_fingerprints)
        v_errors = self.estimate_accuracy(estimated_locations, test_locations)

        return np.mean(v_errors), np.percentile(v_errors, 75)

    # ---------------------------------------------------------
    # get_locations
    # ---------------------------------------------------------
    # Get the location of a set of test_fingerprints
    # ---------------------------------------------------------
    def get_locations(self, test_fingerprints):
        distances = distance.cdist(self.train_fingerprints, test_fingerprints, "euclidean")
        n_test = test_fingerprints.shape[0]

        locations = np.zeros([n_test, 2])
        for i in range(n_test):
            x, y = self.get_location(distances[:, i])
            locations[i, 0] = x
            locations[i, 1] = y

        return locations

    # ---------------------------------------------------------
    # get_location
    # ---------------------------------------------------------
    # Given a test fingerprint, return the estimated location (x,y) for this fingerprint
    # All the samples are in the same floor
    # ---------------------------------------------------------
    def get_location(self, distances):
        x = 0
        y = 0
        for i in range(self.k):
            best = np.argmin(distances)
            x = x + self.train_locations[best, 0]
            y = y + self.train_locations[best, 1]
            distances[best] = 1000000  # big number

        x = x / self.k
        y = y / self.k

        return x, y

    # ---------------------------------------------------------
    # estimate_accuracy
    # ---------------------------------------------------------
    # Estimate the location accuracy of the estimated locations given the true ones.
    # ---------------------------------------------------------
    def estimate_accuracy(self, estimated_locations, true_locations):
        n_samples = estimated_locations.shape[0]
        v_errors = np.zeros(n_samples)
        for i in range(n_samples):
            v_errors[i] = self.distance_space(estimated_locations[i, :], true_locations[i, :])

        return v_errors

    # ---------------------------------------------------------
    # distance_space
    # ---------------------------------------------------------
    # Euclidean distance between two 2D or 3D points
    # ---------------------------------------------------------
    def distance_space(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

