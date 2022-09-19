import numpy as np


def get_closest_point(point, array):
    """
    Find ID of the closest point from point to array.
    Using euclidian norm.
    Works in 2d.
    :param point:
    :param array:
    :return: id of the closest point
    """
    min_distance = np.inf
    min_id = 0
    for i in range(len(array)):
        distance = np.sqrt(pow(array[i][0] - point[0], 2.0) + pow(array[i][1] - point[1], 2.0))

        if distance < min_distance:
            min_distance = distance
            min_id = i
    return min_id


def get_closest_point_vectorized(point, array):
    """
    Find ID of the closest point from point to array.
    Using euclidian norm.
    Works in nd.
    :param point: np.array([x, y, z, ...])
    :param array: np.array([[x1, y1, z1, ...], [x2, y2, z2, ...], [x3, y3, z3, ...], ...])
    :return: id of the closest point
    """

    min_id = np.argmin(np.sum(np.square(array - point), 1))

    return min_id


def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    dist_from_segment_start = np.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], dist_from_segment_start, min_dist_segment
