import numpy as np
import matplotlib.pyplot as plt
from helpers.closest_point import get_closest_point_vectorized
from helpers.console_handling import color_map


def determine_side(a, b, p):
    """ Determines, if car is on right side of trajectory or on left side
    Arguments:
         a - point of trajectory, which is nearest to the car, geometry_msgs.msg/Point
         b - next trajectory point, geometry_msgs.msg/Point
         p - actual position of car, geometry_msgs.msg/Point

    Returns:
         1 if car is on left side of trajectory
         -1 if car is on right side of trajectory
         0 if car is on trajectory
    """
    side = (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    if side > 0:
        return -1
    elif side < 0:
        return 1
    else:
        return 0


class Track:
    def __init__(self, centerline_descriptor, track_width, reference_speed):
        """
        :param centerline_descriptor: np.array Nx5 [s_start, start_x, start_y, curvature, start angle]
        :param track_width:
        """
        self.cline_desc = centerline_descriptor
        self.track_width = track_width
        self.trajectory = np.empty((0, 7), float)  # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
        self.left_boundary = None
        self.right_boundary = None
        self.calculate_trajectory(reference_speed)
        self.calculate_trajectory_boundary()
        self.show_trajectory()

    def calculate_trajectory_boundary(self):
        # initialize vector [number_of_points, 2]
        vectors = np.zeros((self.trajectory[:, 1].shape[0], 2))
        # rotate each vector that it aims away from the trajectory (compute normals to the trajectory)
        # the length of vectors is track_width/2
        for i in range(self.trajectory[:, 1].shape[0]):
            R = self.get_rotation_matrix_2d(self.trajectory[i, 3] - np.pi / 2.0)
            vectors[i, :] = R @ np.array([self.track_width / 2.0, 0.0])

        self.right_boundary = np.array([self.trajectory[:, 1] + vectors[:, 0], self.trajectory[:, 2] + vectors[:, 1]]).T
        self.left_boundary = np.array([self.trajectory[:, 1] - vectors[:, 0], self.trajectory[:, 2] - vectors[:, 1]]).T

    def calculate_trajectory(self, reference_speed):
        max_distance_between_points = 8.0
        for i in range(self.cline_desc.shape[0] - 1):
            segment_length = self.cline_desc[i + 1, 0] - self.cline_desc[i, 0]
            num_points_on_segment = int(np.ceil(segment_length / max_distance_between_points))
            point_dist_on_segment = segment_length / num_points_on_segment

            new_point = np.zeros(7)

            if self.cline_desc[i, 3] == 0.0:
                # curvature = 0 -> line
                for j in range(num_points_on_segment):
                    # (s,x,y,yaw) trajectory waypoints
                    new_point[0] = self.cline_desc[i, 0] + j * point_dist_on_segment  # s
                    new_point[3] = yaw = self.cline_desc[i, 4]
                    new_point[1:3] = np.array([self.cline_desc[i, 1], self.cline_desc[i, 2]]) + self.get_rotation_matrix_2d(yaw) @ np.array(
                        [point_dist_on_segment * j, 0.0])
                    new_point[4] = self.cline_desc[i, 3]
                    new_point[5] = reference_speed
                    new_point[6] = 0.0
                    self.trajectory = np.append(self.trajectory, [new_point], 0)
            else:
                arc_start_angle = np.mod(self.cline_desc[i, 4] - np.pi / 2.0 * np.sign(self.cline_desc[i, 3]), 2 * np.pi)
                arc_end_angle = np.mod(self.cline_desc[i + 1, 4] - np.pi / 2.0 * np.sign(self.cline_desc[i, 3]), 2 * np.pi)
                arc_angle_length = np.mod(np.sign(self.cline_desc[i, 3]) * arc_end_angle - np.sign(self.cline_desc[i, 3]) * arc_start_angle,
                                          2 * np.pi)
                for j in range(num_points_on_segment):
                    # (s,x,y,yaw) trajectory waypoints
                    new_point[0] = self.cline_desc[i, 0] + j * point_dist_on_segment

                    new_point[3] = np.mod(
                        arc_start_angle + arc_angle_length / num_points_on_segment * j * np.sign(self.cline_desc[i, 3]) + np.pi / 2.0 * np.sign(
                            self.cline_desc[i, 3]), 2 * np.pi)

                    new_point[1:3] = self.find_arc_end(np.array([self.cline_desc[i, 1], self.cline_desc[i, 2]]),
                                                       1 / self.cline_desc[i, 3],
                                                       arc_start_angle,
                                                       np.mod(arc_angle_length / num_points_on_segment * j, 2 * np.pi))

                    new_point[4] = self.cline_desc[i, 3]
                    new_point[5] = reference_speed
                    new_point[6] = 0.0
                    self.trajectory = np.append(self.trajectory, [new_point], 0)

        self.trajectory = np.append(self.trajectory, [self.trajectory[0]], 0)
        self.trajectory[-1, 0] = self.trajectory[-2, 0] + np.linalg.norm(self.trajectory[-2, 1:3] - self.trajectory[0, 1:3])
        # print(self.trajectory)

    def get_reference_trajectory(self):
        return self.trajectory

    def get_curvature_at_s(self, s):
        # trajectory ... s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
        diff = self.trajectory[:, 0] - s
        segment_id = np.argmax(diff[diff <= 0])  # should be always id of the point that has smaller s than the point
        curvature = self.trajectory[segment_id, 4]
        return curvature

    def find_arc_end(self, start_point, radius, start_angle, arc_angle):
        # print(f"start angle: {start_angle}")
        angle = start_angle + arc_angle * np.sign(radius)
        # print(f"angle: {start_angle + np.pi/2.0 * np.sign(radius)}")
        C = self.find_center_of_arc(start_point, radius, start_angle + np.pi / 2.0 * np.sign(radius))
        arc_end_point = C + abs(radius) * np.array([np.cos(angle), np.sin(angle)])
        return arc_end_point

    def find_center_of_arc(self, point, radius, direction):
        """
        :param point: Point on the arc. np.array([x, y])
        :param radius: Radius of arc. - -> arc is going to the right, + -> arc is going to the left.
        :param direction: direction which way the arc continues from the point (angle 0-2pi)
        :return: center: np.array([x, y])
        """
        R = self.get_rotation_matrix_2d(direction + np.pi / 2.0 * np.sign(radius))
        # print(f"R: {R}")
        # print(f"R: {direction}")
        # print(f"F: {(R @ np.array([[abs(radius)], [0.0]]))}")
        C = np.squeeze(point + (R @ np.array([[abs(radius)], [0.0]])).T)
        return C

    def get_arc_length(self, start_angle, end_angle, radius):
        """
        :param start_angle: 0 - 2pi
        :param end_angle: 0 - 2pi
        :param radius:
        :return:
        """
        return np.absolute(end_angle - start_angle) * np.absolute(radius)

    def get_rotation_matrix_2d(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    def show_trajectory(self):
        plt.plot(self.trajectory[:, 1], self.trajectory[:, 2], "k--")

        # initialize vector [number_of_points, 2]
        vectors = np.zeros((self.trajectory[:, 1].shape[0], 2))
        # rotate each vector that it aims away from the trajectory (compute normals to the trajectory)
        # the length of vectors is track_width/2
        for i in range(self.trajectory[:, 1].shape[0]):
            R = self.get_rotation_matrix_2d(self.trajectory[i, 3] - np.pi / 2.0)
            vectors[i, :] = R @ np.array([self.track_width / 2.0, 0.0])

        # add and subtract these vectors from the trajectory to get track borders and plot them
        plt.plot(self.trajectory[:, 1] + vectors[:, 0], self.trajectory[:, 2] + vectors[:, 1], "b-")
        plt.plot(self.trajectory[:, 1] - vectors[:, 0], self.trajectory[:, 2] - vectors[:, 1], "b-")

        plt.axis('square')
        plt.show()

    def cartesian_to_frenet_arr(self, poses):  # TODO probably a good idea to rewrite this in batch mode
        """
        :param poses: list of pose -> pose = np.array([x,y,yaw])
        :return: list of poses -> pose = np.array([s,ey,eyaw])
        """
        poses_frenet = []
        for pose in poses:
            poses_frenet.append(self.cartesian_to_frenet(pose))
        return poses_frenet

    def cartesian_to_frenet(self, pose):
        """
        :param pose: np.array([x,y,yaw])
        :return:
        """
        # trajectory ... s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2

        min_id = get_closest_point_vectorized(point=pose[0:2], array=self.trajectory[:, 1:3])
        # print("----")
        # print(self.trajectory[min_id, :])
        # print(pose[0:2])
        # print(f"min_id {min_id}")
        # print(self.trajectory[min_id, 3] - np.pi / 2)

        # print(np.arctan2(pose[1] - self.trajectory[min_id, 2], pose[0] - self.trajectory[min_id, 1]))
        # print(self.trajectory[min_id, 3])

        a = np.mod(np.arctan2(pose[1] - self.trajectory[min_id, 2], pose[0] - self.trajectory[min_id, 1]) - (self.trajectory[min_id, 3] - np.pi / 2),
                   2.0 * np.pi)

        if a > np.pi:
            min_id = (min_id + self.trajectory.shape[0] - 2) % (self.trajectory.shape[0] - 1)

        # print(f"min_id corrected {min_id}")

        if self.trajectory[min_id, 4] == 0:
            # print("line")

            eyaw = pose[2] - self.trajectory[min_id, 3]

            vector = (self.trajectory[min_id + 1, 1:3] - self.trajectory[min_id, 1:3])[np.newaxis].T

            vector = vector / np.linalg.norm(vector)

            projector = vector @ vector.T
            # print(projector)
            projection = projector @ (pose[0:2] - self.trajectory[min_id, 1:3])

            point = self.trajectory[min_id, 1:3] + projection

            ey = np.linalg.norm(point - pose[0:2]) * determine_side(self.trajectory[min_id, 1:3], self.trajectory[min_id + 1, 1:3], pose[0:2])
            # print(self.trajectory[min_id, 0])
            # print(point)
            s = self.trajectory[min_id, 0] + np.linalg.norm(point - self.trajectory[min_id, 1:3])

            # print(ey)

            # print(s)

            # print(eyaw)
        else:
            # print("circle")
            # print(min_id)

            # print(self.trajectory[min_id, 3])
            # print(1.0 / self.trajectory[min_id, 4])
            # print(np.array([self.trajectory[min_id, 1], self.trajectory[min_id, 2]]))

            center = self.find_center_of_arc(point=np.array([self.trajectory[min_id, 1], self.trajectory[min_id, 2]]),
                                             radius=1.0 / self.trajectory[min_id, 4],
                                             direction=self.trajectory[min_id, 3])

            # print(center)  # correct
            # print(self.trajectory[min_id, 0])

            ey = (abs(1.0 / self.trajectory[min_id, 4]) - np.linalg.norm(center - pose[0:2])) * np.sign(self.trajectory[min_id, 4])

            # print(ey)

            start_point_angle = np.arctan2(self.trajectory[min_id, 2] - center[1], self.trajectory[min_id, 1] - center[0])
            end_angle = np.arctan2(pose[1] - center[1], pose[0] - center[0])

            if end_angle < 0.0:
                end_angle = end_angle + 2.0 * np.pi

            if start_point_angle < 0.0:
                start_point_angle = start_point_angle + 2.0 * np.pi

            angle = np.sign(self.trajectory[min_id, 4]) * end_angle - np.sign(self.trajectory[min_id, 4]) * start_point_angle

            if angle < 0.0:
                angle = angle + 2.0 * np.pi

            # print(f"start angle: {start_point_angle}")
            # print(f"end_angle: {end_angle}")

            # print(angle)
            # print(abs(1.0 / self.trajectory[min_id, 4]))

            # print(angle * abs(1.0 / self.trajectory[min_id, 4]))

            s = self.trajectory[min_id, 0] + angle * abs(1.0 / self.trajectory[min_id, 4])

            eyaw = pose[2] - (end_angle + np.pi / 2.0 * np.sign(self.trajectory[min_id, 4]))

            # print(s)
            #
            # print(eyaw)

        if eyaw > np.pi:
            eyaw -= 2.0 * np.pi
        if eyaw < -np.pi:
            eyaw += 2.0 * np.pi

        return np.array([s, ey, eyaw])

    def frenet_to_cartesian_arr(self, poses):  # TODO probably a good idea to rewrite this in batch mode
        """
        :param poses: list of pose -> pose = np.array([s,ey,eyaw])
        :return: list of poses -> pose = np.array([x,y,yaw])
        """
        poses_cartesian = []
        for pose in poses:
            poses_cartesian.append(self.frenet_to_cartesian(pose))
        return poses_cartesian

    def frenet_to_cartesian(self, pose):
        """
        :param pose: [s, ey, eyaw]
        :return:
        """
        # trajectory ... s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
        diff = self.trajectory[:, 0] - pose[0]

        # print(diff)

        segment_id = np.argmax(diff[diff <= 0])  # should be always id of the point that has smaller s than the point

        # print(segment_id)
        # print(self.trajectory[segment_id, :])

        if self.trajectory[segment_id, 4] == 0:
            # line
            yaw = np.mod(self.trajectory[segment_id, 3] + pose[2], 2.0 * np.pi)
            s_reminder = pose[0] - self.trajectory[segment_id, 0]
            R1 = self.get_rotation_matrix_2d(self.trajectory[segment_id, 3])
            R2 = self.get_rotation_matrix_2d(self.trajectory[segment_id, 3] + np.pi / 2.0 * np.sign(pose[1]))
            position = (self.trajectory[segment_id, 1:3] + (R1 @ np.array([[abs(s_reminder)], [0.0]])).T + (
                    R2 @ np.array([[abs(pose[1])], [0.0]])).T).squeeze()
        else:
            # circle
            center = self.find_center_of_arc(self.trajectory[segment_id, 1:3],
                                             1.0 / self.trajectory[segment_id, 4],
                                             self.trajectory[segment_id, 3])

            s_reminder = pose[0] - self.trajectory[segment_id, 0]

            start_angle = np.mod(self.trajectory[segment_id, 3] - np.pi / 2.0 * np.sign(self.trajectory[segment_id, 4]), 2 * np.pi)
            arc_angle = s_reminder / abs(1.0 / self.trajectory[segment_id, 4])

            trajectory_point = self.find_arc_end(self.trajectory[segment_id, 1:3],
                                                 1.0 / self.trajectory[segment_id, 4],
                                                 start_angle,
                                                 arc_angle)
            vector = trajectory_point - center

            position = trajectory_point + vector / np.linalg.norm(vector) * pose[1] * (-1) * np.sign(self.trajectory[segment_id, 4])
            yaw = np.mod(np.arctan2(vector[1], vector[0]) + np.pi / 2.0 * np.sign(self.trajectory[segment_id, 4]) + pose[2], 2 * np.pi)

        return np.array([position[0], position[1], yaw])


if __name__ == "__main__":
    # centerline_descriptor = np.array([[0.0, 25 * np.pi, 25 * np.pi + 50, 2 * 25 * np.pi + 50, 2 * 25 * np.pi + 100],
    #                                   [0.0, 0.0, -50.0, -50.0, 0.0],
    #                                   [0.0, 50.0, 50.0, 0.0, 0.0],
    #                                   [1 / 25, 0.0, 1 / 25, 0.0, 1 / 25],
    #                                   [0.0, np.pi, np.pi, 0.0, 0.0]]).T

    # centerline_descriptor = np.array([[0.0, 25 * np.pi, 25 * np.pi + 25, 25 * (3.0 * np.pi / 2.0) + 25, 25 * (3.0 * np.pi / 2.0) + 50,
    #                                    25 * (2.0 * np.pi + np.pi / 2.0) + 50, 25 * (2.0 * np.pi + np.pi / 2.0) + 125, 25 * (3.0 * np.pi) + 125,
    #                                    25 * (3.0 * np.pi) + 200],
    #                                   [0.0, 0.0, -25.0, -50.0, -50.0, -100.0, -100.0, -75.0, 0.0],
    #                                   [0.0, 50.0, 50.0, 75.0, 100.0, 100.0, 25.0, 0.0, 0.0],
    #                                   [1 / 25, 0.0, -1 / 25, 0.0, 1 / 25, 0.0, 1 / 25, 0.0, 1 / 25],
    #                                   [0.0, np.pi, np.pi, np.pi / 2.0, np.pi / 2.0, 3.0 * np.pi / 2.0, 3.0 * np.pi / 2.0, 0.0, 0.0]]).T

    centerline_descriptor = np.array([[0.0, 50.0, 25 * np.pi + 50, 25 * np.pi + 100, 25 * 2 * np.pi + 100],
                                      [0.0, -50.0, -50.0, 0.0, 0.0],
                                      [0.0, 0.0, 50.0, 50.0, 0.0],
                                      [0.0, -1 / 25, 0.0, -1 / 25, 0.0],
                                      [np.pi, np.pi, 0.0, 0.0, np.pi]]).T

    track = Track(centerline_descriptor=centerline_descriptor, track_width=10.0, reference_speed=5.0)

    # test_transformations = [np.array([-74.81885038777025, 25.34721413251105, np.pi + 0.2])]
    test_transformations = [np.array([-74.0, 25.0, np.pi])]

    plt.plot(track.trajectory[:, 1], track.trajectory[:, 2], "ko")
    plt.plot(test_transformations[0][0], test_transformations[0][1], "ro")
    # plt.show()

    for i in range(len(test_transformations)):
        print(f"***---*** {i}")
        print(f"Original point cartesian: {test_transformations[i]}")
        pose_frenet = track.cartesian_to_frenet(test_transformations[i])
        print(f"Point frenet: {pose_frenet}")
        pose = track.frenet_to_cartesian(pose_frenet)
        print(f"Point cartesian: {pose}")

        print(np.all(np.isclose(test_transformations[i], pose, rtol=5.e-3, atol=5.e-3, )))

        plt.plot(pose[0], pose[1], "go")

    plt.show()

    # centerline_descriptor1 = np.array([[0.0, 100 * np.pi, 100 * np.pi + 200, 2 * 100 * np.pi + 200, 2 * 100 * np.pi + 400],
    #                                    [0.0, 0.0, -200.0, -200.0, 0.0],
    #                                    [0.0, 200.0, 200.0, 0.0, 0.0],
    #                                    [1 / 100, 0.0, 1 / 100, 0.0, 1 / 100],
    #                                    [0.0, np.pi, np.pi, 0.0, 0.0]]).T
    #
    # print(centerline_descriptor1)
    # print(centerline_descriptor1.shape)
    #
    # track = Track(centerline_descriptor=centerline_descriptor1, track_width=1.0, reference_speed=5.0)
    #
    # # test find_center_of_arc
    # # [point, radius, direction, center (answer)]
    # test_center_of_arc = [[np.array([0.0, 0.0]), 3.0, 0.0, np.array([0.0, 3.0])],
    #                       [np.array([0.0, 0.0]), -3.0, 0.0, np.array([0.0, -3.0])],
    #                       [np.array([1.0, 1.0]), 3.0, np.pi / 2, np.array([-2.0, 1.0])],
    #                       ]
    # for i in range(len(test_center_of_arc)):
    #     center = track.find_center_of_arc(test_center_of_arc[i][0], test_center_of_arc[i][1], test_center_of_arc[i][2])
    #     if not np.all(np.isclose(center, test_center_of_arc[i][-1])):
    #         print(f"{color_map['red']}NOT OK{color_map['reset']}")
    #         print(f"{color_map['red']}{test_center_of_arc[i][0]}  {test_center_of_arc[i][1]} "
    #               f"{test_center_of_arc[i][2]}  {test_center_of_arc[i][-1]} {color_map['reset']}")
    #         print(f"Answer {center}")
    #         exit()
    # print(f"{color_map['green']}Center of arc test OK{color_map['reset']}")
    #
    # # track.find_arc_end(start_point, radius, start_angle, arc_angle)
    # # test find_arc_end
    # # [start_point, radius, start_angle, arc_angle, end_point (answer)]
    # test_arc_end = [[np.array([0.0, 0.0]), 3.0, np.pi + np.pi / 2, np.pi / 2, np.array([3.0, 3.0])],
    #                 [np.array([3.0, 3.0]), -6.0, 0.0, np.pi / 2, np.array([-3.0, -3.0])],
    #                 ]
    # for i in range(len(test_arc_end)):
    #     end = track.find_arc_end(test_arc_end[i][0], test_arc_end[i][1], test_arc_end[i][2], test_arc_end[i][3])
    #     if not np.all(np.isclose(end, test_arc_end[i][-1])):
    #         print(f"{color_map['red']}NOT OK{color_map['reset']}")
    #         print(f"{color_map['red']}{test_arc_end[i][0]}  {test_arc_end[i][1]} "
    #               f"{test_arc_end[i][2]} {test_arc_end[i][3]} {test_arc_end[i][-1]} {color_map['reset']}")
    #         print(f"Answer {end}")
    #         exit()
    # print(f"{color_map['green']}Arc end test OK{color_map['reset']}")
    #
    # # for i in range(5000):
    # # track.cartesian_to_frenet(np.array([-0.01, 0.11, 0]))
    #
    # test_transformations = [np.array([0.0, 0.0, 0.0]),
    #                         np.array([101, 100, np.pi/2]),
    #                         np.array([99, 100, np.pi/2]),
    #                         ]
    #
    # for i in range(len(test_transformations)):
    #     print(f"***---*** {i}")
    #     print(f"Original point cartesian: {test_transformations[i]}")
    #     pose_frenet = track.cartesian_to_frenet(test_transformations[i])
    #     print(f"Point frenet: {pose_frenet}")
    #     pose = track.frenet_to_cartesian(pose_frenet)
    #     print(f"Point cartesian: {pose}")
    #
    #     print(np.all(np.isclose(test_transformations[i], pose, rtol=5.e-3, atol=5.e-3,)))
