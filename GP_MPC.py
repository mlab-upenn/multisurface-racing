import time
import yaml
import gym
from argparse import Namespace
from regulators.pure_pursuit import *
from regulators.path_follow_mpc import *
from models.extended_kinematic import ExtendedKinematicModel
from models.GP_model_ensembling import GPEnsembleModel
from models.GP_model_ensembling_frenet import GPEnsembleModelFrenet
from helpers.closest_point import *
from helpers.track import Track
import torch
import gpytorch
import os
import numpy as np

from pyglet.gl import GL_POINTS
import pyglet
import copy
import json


@dataclass
class MPCConfigEXT:
    NXK: int = 7  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    NU: int = 2  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 15  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.000001, 2.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.000001, 2.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 0.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # state error cost matrix, for the next (T) prediction time steps
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 0.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # final state error matrix, penalty  for the final state constraints
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 3.0  # dist step [m] kinematic
    LENGTH: float = 4.298  # Length of the vehicle [m]
    WIDTH: float = 1.674  # Width of the vehicle [m]
    LR: float = 1.50876
    LF: float = 0.88392
    WB: float = 0.88392 + 1.50876  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 45.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 11.5  # maximum acceleration [m/ss]
    MAX_DECEL: float = -45.0  # maximum acceleration [m/ss]

    MASS: float = 1225.887  # Vehicle mass


@dataclass
class MPCConfigGP:
    NXK: int = 7  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    NU: int = 2  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 10  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.0000008, 2.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.0000008, 2.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.0, 0.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # state error cost matrix, for the next (T) prediction time steps
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.0, 0.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # final state error matrix, penalty  for the final state constraints
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 3.0  # dist step [m] kinematic
    LENGTH: float = 4.298  # Length of the vehicle [m]
    WIDTH: float = 1.674  # Width of the vehicle [m]
    LR: float = 1.50876
    LF: float = 0.88392
    WB: float = 0.88392 + 1.50876  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 45.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 11.5  # maximum acceleration [m/ss]
    MAX_DECEL: float = -45.0  # maximum acceleration [m/ss]

    MASS: float = 1225.887  # Vehicle mass


@dataclass
class MPCConfigGPFrenet:
    NXK: int = 7  # length of kinematic state vector: z = [s, ey, vx, eyaw, vy, yaw rate, steering angle]
    NU: int = 2  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 15  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.0000008, 2.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.0000008, 2.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([5.5, 10.5, 5.0, 8.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # state error cost matrix, for the next (T) prediction time steps
    Qfk: list = field(
        default_factory=lambda: np.diag([5.5, 10.5, 5.0, 8.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # final state error matrix, penalty  for the final state constraints
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 3.0  # dist step [m] kinematic
    LENGTH: float = 4.298  # Length of the vehicle [m]
    WIDTH: float = 1.674  # Width of the vehicle [m]
    LR: float = 1.50876
    LF: float = 0.88392
    WB: float = 0.88392 + 1.50876  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 45.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 11.5  # maximum acceleration [m/ss]
    MAX_DECEL: float = -45.0  # maximum acceleration [m/ss]

    MASS: float = 1225.887  # Vehicle mass


def draw_point(e, point, colour):
    scaled_point = 50. * point
    ret = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_point[0], scaled_point[1], 0]), ('c3B/stream', colour))
    return ret


class DrawDebug:
    def __init__(self):
        self.reference_traj_show = np.array([[0, 0]])
        self.predicted_traj_show = np.array([[0, 0]])
        self.dyn_obj_drawn = []
        self.f = 0
        self.drawn_once = False

    def draw_debug(self, e):
        # delete dynamic objects
        while len(self.dyn_obj_drawn) > 0:
            if self.dyn_obj_drawn[0] is not None:
                self.dyn_obj_drawn[0].delete()
            self.dyn_obj_drawn.pop(0)

        # spawn new objects
        for p in self.reference_traj_show:
            self.dyn_obj_drawn.append(draw_point(e, p, [255, 0, 0]))

        for p in self.predicted_traj_show:
            self.dyn_obj_drawn.append(draw_point(e, p, [0, 255, 0]))

    def draw_points_once(self, e, points, color):
        """
        :param e:
        :param points: np.array([[x, y], [x, y], ...])
        :param color: [r, g, b]
        :return:
        """
        if not self.drawn_once:
            for p in points:
                draw_point(e, p, color)
            self.drawn_once = True



def main():  # after launching this you can run visualization.py to see the results
    """
    main entry point
    """

    # Program parameters
    model_in_first_lap = 'ext_kinematic'  # options: ext_kinematic, pure_pursuit
    # currently only "custom_track" works
    map_name = 'custom_track'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape, BrandsHatch, DualLaneChange, custom_track
    use_dyn_friction = False
    gp_mpc_type = 'frenet'  # cartesian, frenet
    control_step = 100.0  # ms
    render_every = 30  # render graphics every n sim steps
    constant_speed = True
    constant_friction = 0.7

    # Creating the single-track Motion planner and Controller

    # Init Pure-Pursuit regulator
    work = {'mass': 1225.88, 'lf': 0.80597534362552312, 'tlad': 10.6461887897713965, 'vgain': 1.0}

    # Load map config file
    with open('configs/config_%s.yaml' % 'SaoPaulo') as file:  # map_name -- SaoPaulo
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    if not map_name == 'custom_track':

        if use_dyn_friction:
            tpamap_name = './maps/rounded_rectangle/rounded_rectangle_tpamap.csv'
            tpadata_name = './maps/rounded_rectangle/rounded_rectangle_tpadata.json'

            tpamap = np.loadtxt(tpamap_name, delimiter=';', skiprows=1)

            tpadata = {}
            with open(tpadata_name) as f:
                tpadata = json.load(f)

        raceline = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
        waypoints = np.array(raceline)
    else:
        centerline_descriptor = np.array([[0.0, 25 * np.pi, 25 * np.pi + 50, 2 * 25 * np.pi + 50, 2 * 25 * np.pi + 100],
                                          [0.0, 0.0, -50.0, -50.0, 0.0],
                                          [0.0, 50.0, 50.0, 0.0, 0.0],
                                          [1 / 25, 0.0, 1 / 25, 0.0, 1 / 25],
                                          [0.0, np.pi, np.pi, 0.0, 0.0]]).T

        # centerline_descriptor = np.array([[0.0, 50.0, 25 * np.pi + 50, 25 * np.pi + 100, 25 * 2 * np.pi + 100],
        #                                   [0.0, -50.0, -50.0, 0.0, 0.0],
        #                                   [0.0, 0.0, 50.0, 50.0, 0.0],
        #                                   [0.0, -1 / 25, 0.0, -1/25, 0.0],
        #                                   [np.pi, np.pi, 0.0, 0.0, np.pi]]).T

        # centerline_descriptor = np.array([[0.0, 25 * np.pi, 25 * np.pi + 25, 25 * (3.0 * np.pi / 2.0) + 25, 25 * (3.0 * np.pi / 2.0) + 50,
        #                                    25 * (2.0 * np.pi + np.pi / 2.0) + 50, 25 * (2.0 * np.pi + np.pi / 2.0) + 125, 25 * (3.0 * np.pi) + 125,
        #                                    25 * (3.0 * np.pi) + 200],
        #                                   [0.0, 0.0, -25.0, -50.0, -50.0, -100.0, -100.0, -75.0, 0.0],
        #                                   [0.0, 50.0, 50.0, 75.0, 100.0, 100.0, 25.0, 0.0, 0.0],
        #                                   [1 / 25, 0.0, -1 / 25, 0.0, 1 / 25, 0.0, 1 / 25, 0.0, 1/25],
        #                                   [0.0, np.pi, np.pi, np.pi / 2.0, np.pi / 2.0, 3.0 * np.pi / 2.0, 3.0 * np.pi / 2.0, 0.0, 0.0]]).T

        print(centerline_descriptor)
        print(centerline_descriptor.shape)

        track = Track(centerline_descriptor=centerline_descriptor, track_width=10.0, reference_speed=5.0)
        waypoints = track.get_reference_trajectory()
    # waypoints[:, 3] += 1.5707963268

    # waypoints[:, 5] *= 0.82

    if constant_speed:
        waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * 4.5

    # init controllers
    planner_pp = PurePursuitPlanner(conf, 0.805975 + 1.50876)  # 0.805975 + 1.50876
    planner_pp.waypoints = waypoints

    planner_gp_mpc = STMPCPlanner(model=GPEnsembleModel(config=MPCConfigGP()), waypoints=waypoints,
                                  config=MPCConfigGP())

    if gp_mpc_type == 'frenet':
        planner_gp_mpc_frenet = STMPCPlanner(model=GPEnsembleModelFrenet(config=MPCConfigGPFrenet(), track=track), waypoints=waypoints,
                                             config=MPCConfigGPFrenet(), track=track)
        planner_gp_mpc_frenet.trajectry_interpolation = 1

    planner_ekin_mpc = STMPCPlanner(model=ExtendedKinematicModel(config=MPCConfigEXT()), waypoints=waypoints,
                                    config=MPCConfigEXT())

    # init graphics
    draw = DrawDebug()

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 2000
        e.left = left - 2000
        e.right = right + 2000
        e.top = top + 2000
        e.bottom = bottom - 2000

        planner_pp.render_waypoints(e)
        draw.draw_debug(e)
        draw.draw_points_once(e=e, color=[255, 0, 0], points=np.concatenate((track.left_boundary, track.right_boundary)))

    # MB - reference point: center of mass
    # dynamic_ST - reference point: center of mass

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                   num_agents=1, timestep=0.001, model='MB', drive_control_mode='acc',
                   steering_control_mode='vel')

    env.add_render_callback(render_callback)
    # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    start_id = 0
    obs, step_reward, done, info = env.reset(
        np.array([[waypoints[start_id, 1], waypoints[start_id, 2], waypoints[start_id, 3], 0.0, waypoints[start_id, 5], 0.0, 0.0]]))
    env.render()

    laptime = 0.0
    start = time.time()
    last_render = 0

    # init logger
    log = {'time': [], 'x': [], 'y': [], 'lap_n': [], 'vx': [], 'v_ref': [], 'vx_mean': [], 'vx_var': [], 'vy_mean': [],
           'vy_var': [], 'theta_mean': [], 'theta_var': [], 'true_vx': [], 'true_vy': [], 'true_yaw_rate': [], 'tracking_error': []}

    log_dataset = {'X0': [], 'X1': [], 'X2': [], 'X3': [], 'X4': [], 'X5': [], 'Y0': [], 'Y1': [], 'Y2': [],
                   'X0[t-1]': [], 'X1[t-1]': [], 'X2[t-1]': [], 'X3[t-1]': [], 'X4[t-1]': [], 'X5[t-1]': [], 'Y0[t-1]': [], 'Y1[t-1]': [],
                   'Y2[t-1]': []}

    X_t_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # calc number of sim steps per one control step
    num_of_sim_steps = int(control_step / (env.timestep * 1000.0))

    gp_model_trained = 0
    last_speed = waypoints[:, 5][0]
    gather_data = 0
    logged_data = 0

    print('Model used: %s' % model_in_first_lap)

    original_vel_profile = copy.deepcopy(waypoints[:, 5])
    x0 = np.array([env.sim.agents[0].state[0],
                   env.sim.agents[0].state[1],
                   env.sim.agents[0].state[3],  # vx
                   env.sim.agents[0].state[4],  # yaw angle
                   env.sim.agents[0].state[10],  # vy
                   env.sim.agents[0].state[5],  # yaw rate
                   env.sim.agents[0].state[2],  # steering angle
                   ]) + np.random.randn(7) * 0.00001

    # xcl = []
    # ucl = []
    laps_done = 0

    while not done:

        # Regulator step MPC
        vehicle_state = np.array([env.sim.agents[0].state[0],  # x
                                  env.sim.agents[0].state[1],  # y
                                  env.sim.agents[0].state[3],  # vx
                                  env.sim.agents[0].state[4],  # yaw angle
                                  env.sim.agents[0].state[10],  # vy
                                  env.sim.agents[0].state[5],  # yaw rate
                                  env.sim.agents[0].state[2],  # steering angle
                                  ])  # + np.random.randn(7) * 0.00001

        pose_frenet = track.cartesian_to_frenet(np.array([vehicle_state[0], vehicle_state[1], vehicle_state[3]]))  # np.array([x,y,yaw])

        print(f"X: {vehicle_state[0]}  Y: {vehicle_state[1]}  S: {pose_frenet[0]}")
        # print(f"X: {vehicle_state[0]}  Y: {vehicle_state[1]}  YAW: {vehicle_state[3]}  EYAW: {pose_frenet[2]}")

        # print(env.sim.agents[0].state[10])
        # print(env.sim.agents[0].state[3])

        # if len(xcl) == 0:
        #     xcl.append(vehicle_state)  # add x0 to closed loop trajectory

        mean, lower, upper = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        u = [0.0, 0.0]
        tracking_error = 0.0
        total_var = 0.0

        if gp_model_trained <= 1:
            print("Initial model")
            # if True:
            if model_in_first_lap == 'pure_pursuit':
                # Regulator step pure pursuit
                speed, steer_angle = planner_pp.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                                                     work['tlad'],
                                                     work['vgain'])

                draw.reference_traj_show = np.array([[obs['poses_x'][0]], [obs['poses_y'][0]]]).T

                error_steer = steer_angle - env.sim.agents[0].state[2]
                u[1] = 10.0 * error_steer

                error_drive = speed - env.sim.agents[0].state[3]
                u[0] = 12.0 * error_drive

                if obs['lap_counts'][0] == 0:
                    u[0] += np.random.randn(1)[0] * 0.2
                    u[1] += np.random.randn(1)[0] * 0.01

            elif model_in_first_lap == "ext_kinematic":
                u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_ekin_mpc.plan(
                    vehicle_state)
                u[0] = u[0] / planner_gp_mpc.config.MASS  # Force to acceleration

                # draw predicted states and reference trajectory
                draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
                draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

            if obs['lap_counts'][0] == 1:
                u[0] += np.random.randn(1)[0] * 0.001
                u[1] += np.random.randn(1)[0] * 0.01

        else:
            # print("GP model")
            if gp_mpc_type == 'cartesian':
                u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_gp_mpc.plan(
                    vehicle_state)
                u[0] = u[0] / planner_gp_mpc.config.MASS  # Force to acceleration

                # if waypoints[:, 5][0] <= 5.5:
                #     u[0] += np.random.randn(1)[0] * 0.000005
                #     u[1] += np.random.randn(1)[0] * 0.0005

                # draw predicted states and reference trajectory
                draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
                draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

                _, tracking_error, _, _, _ = nearest_point_on_trajectory(np.array([mpc_pred_x[0], mpc_pred_y[0]]),
                                                                         np.array([mpc_ref_path_x[0:2], mpc_ref_path_y[0:2]]).T)
            elif gp_mpc_type == 'frenet':
                pose_frenet = track.cartesian_to_frenet(np.array([vehicle_state[0], vehicle_state[1], vehicle_state[3]]))  # np.array([x,y,yaw])

                vehicle_state_frenet = np.array([pose_frenet[0],  # s
                                                 pose_frenet[1],  # ey
                                                 env.sim.agents[0].state[3],  # vx
                                                 pose_frenet[2],  # eyaw
                                                 env.sim.agents[0].state[10],  # vy
                                                 env.sim.agents[0].state[5],  # yaw rate
                                                 env.sim.agents[0].state[2],  # steering angle
                                                 ])

                u, mpc_ref_path_s, mpc_ref_path_ey, mpc_pred_s, mpc_pred_ey, mpc_os, mpc_oey = planner_gp_mpc_frenet.plan(
                    vehicle_state_frenet)

                u[0] = u[0] / planner_gp_mpc_frenet.config.MASS  # Force to acceleration

                # if waypoints[:, 5][0] <= 5.5:
                #     u[0] += np.random.randn(1)[0] * 0.000005
                #     u[1] += np.random.randn(1)[0] * 0.0005
                mpc_ref_path_x = np.zeros(mpc_ref_path_s.shape)
                mpc_ref_path_y = np.zeros(mpc_ref_path_s.shape)
                mpc_pred_x = np.zeros(mpc_ref_path_s.shape)
                mpc_pred_y = np.zeros(mpc_ref_path_s.shape)

                for i in range(mpc_ref_path_s.shape[0]):
                    pose_cartesian = track.frenet_to_cartesian(np.array([mpc_pred_s[i], mpc_pred_ey[i], 0.0]))  # [s, ey, eyaw]
                    mpc_pred_x[i] = pose_cartesian[0]
                    mpc_pred_y[i] = pose_cartesian[1]
                    pose_cartesian = track.frenet_to_cartesian(np.array([mpc_ref_path_s[i], mpc_ref_path_ey[i], 0.0]))  # [s, ey, eyaw]
                    mpc_ref_path_x[i] = pose_cartesian[0]
                    mpc_ref_path_y[i] = pose_cartesian[1]

                # draw predicted states and reference trajectory
                draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
                draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

                # _, tracking_error, _, _, _ = nearest_point_on_trajectory(np.array([mpc_pred_x[0], mpc_pred_y[0]]),
                #                                                          np.array([mpc_ref_path_x[0:2], mpc_ref_path_y[0:2]]).T)
            else:
                print("ERROR")
        # u[0] += np.random.randn(1)[0] * 0.00001
        # u[1] += np.random.randn(1)[0] * 0.0001

        if gp_mpc_type == 'cartesian':
            if gp_model_trained:
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    mean, lower, upper = planner_gp_mpc.model.scale_and_predict_model_step(vehicle_state, [u[0] * planner_gp_mpc.config.MASS, u[1]])

        # set correct friction to the environment
        if use_dyn_friction:
            min_id = get_closest_point_vectorized(np.array([obs['poses_x'][0], obs['poses_y'][0]]), np.array(tpamap))
            env.params['tire_p_dy1'] = tpadata[str(min_id)][0] * 0.9  # mu_y
            env.params['tire_p_dx1'] = tpadata[str(min_id)][0]  # mu_x
        else:
            env.params['tire_p_dy1'] = constant_friction * 0.9  # mu_y
            env.params['tire_p_dx1'] = constant_friction  # mu_x

        # Simulation step
        step_reward = 0.0
        for i in range(num_of_sim_steps):
            obs, rew, _, info = env.step(np.array([[u[1], u[0]]]))
            step_reward += rew
            # Rendering
            last_render += 1
            if last_render >= render_every:
                last_render = 0
                env.render(mode='human_fast')
        laptime += step_reward

        vehicle_state_next = np.array([env.sim.agents[0].state[0],  # x
                                       env.sim.agents[0].state[1],  # y
                                       env.sim.agents[0].state[3],  # vx
                                       env.sim.agents[0].state[4],  # yaw angle
                                       env.sim.agents[0].state[10],  # vy
                                       env.sim.agents[0].state[5],  # yaw rate
                                       env.sim.agents[0].state[2],  # steering angle
                                       ]) #+ np.random.randn(7) * 0.00001

        if constant_speed:
            if obs['lap_counts'][0] >= 0 and waypoints[:, 5][0] < 18.7:
                waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.0003
            else:
                waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.00015
        else:
            waypoints[:, 5] += original_vel_profile * 0.000027

        # Logging
        logged_data += 1
        if logged_data > 5:
            log['time'].append(laptime)
            log['lap_n'].append(obs['lap_counts'][0])
            log['x'].append(env.sim.agents[0].state[0])
            log['y'].append(env.sim.agents[0].state[1])
            log['vx'].append(env.sim.agents[0].state[3])
            log['v_ref'].append(waypoints[:, 5][0])
            log['vx_mean'].append(float(mean[0]))
            log['vx_var'].append(float(abs(mean[0] - lower[0])))
            log['vy_mean'].append(float(mean[1]))
            log['vy_var'].append(float(abs(mean[1] - lower[1])))
            log['theta_mean'].append(float(mean[2]))
            log['theta_var'].append(float(abs(mean[2] - lower[2])))
            log['true_vx'].append(env.sim.agents[0].state[3] - vehicle_state[2])
            log['true_vy'].append(env.sim.agents[0].state[10] - vehicle_state[4])
            log['true_yaw_rate'].append(env.sim.agents[0].state[5] - vehicle_state[5])
            log['tracking_error'].append(tracking_error)
            logged_data = 0

        # learning GPs
        u[0] = u[0] * planner_gp_mpc.config.MASS  # Acceleration to force

        # xcl.append(vehicle_state_next)
        # ucl.append(u)

        if planner_gp_mpc_frenet.it > 0:
            planner_gp_mpc_frenet.add_point(vehicle_state, u)


        gather_data_every = 2

        vx_transition = env.sim.agents[0].state[3] + np.random.randn(1)[0] * 0.00001 - vehicle_state[2]
        vy_transition = env.sim.agents[0].state[10] + np.random.randn(1)[0] * 0.00001 - vehicle_state[4]
        yaw_rate_transition = env.sim.agents[0].state[5] + np.random.randn(1)[0] * 0.00001 - vehicle_state[5]

        # print(mean[0] - vx_transition)
        # print(gather_data_every)
        # print('V: %f  Vx: %f  Vy: %f ' % (waypoints[:, 5][0], env.sim.agents[0].state[3], env.sim.agents[0].state[10]))

        gather_data += 1
        if gather_data >= gather_data_every:
            Y_sample = np.array([float(vx_transition), float(vy_transition), float(yaw_rate_transition)])
            X_sample = np.array([float(vehicle_state[2]), float(vehicle_state[4]),
                                 float(vehicle_state[5]), float(vehicle_state[6]), float(u[0]), float(u[1])])

            if gp_mpc_type == 'cartesian':
                planner_gp_mpc.model.add_new_datapoint(X_sample, Y_sample)
            elif gp_mpc_type == 'frenet':
                planner_gp_mpc_frenet.model.add_new_datapoint(X_sample, Y_sample)
            gather_data = 0

            # log_dataset['X0'].append(float(vehicle_state[2]))
            # log_dataset['X1'].append(float(vehicle_state[4]))
            # log_dataset['X2'].append(float(vehicle_state[5]))
            # log_dataset['X3'].append(float(vehicle_state[6]))
            # log_dataset['X4'].append(float(u[0]))
            # log_dataset['X5'].append(float(u[1]))
            # log_dataset['Y0'].append(float(vx_transition))
            # log_dataset['Y1'].append(float(vy_transition))
            # log_dataset['Y2'].append(float(yaw_rate_transition))
            # log_dataset['X0[t-1]'].append(X_t_1[0])
            # log_dataset['X1[t-1]'].append(X_t_1[1])
            # log_dataset['X2[t-1]'].append(X_t_1[2])
            # log_dataset['X3[t-1]'].append(X_t_1[3])
            # log_dataset['X4[t-1]'].append(X_t_1[4])
            # log_dataset['X5[t-1]'].append(X_t_1[5])
            # log_dataset['Y0[t-1]'].append(X_t_1[6])
            # log_dataset['Y1[t-1]'].append(X_t_1[7])
            # log_dataset['Y2[t-1]'].append(X_t_1[8])

        # X_t_1 = [float(vehicle_state[2]), float(vehicle_state[4]), float(vehicle_state[5]), float(vehicle_state[6]), float(u[0]), float(u[1]),
        #          float(vx_transition), float(vy_transition), float(yaw_rate_transition)]

        if obs['lap_counts'][0] - 1 == gp_model_trained:
            # if waypoints[:, 5][0] >= last_speed + 0.25 and False:  # or (waypoints[:, 5][0] > 7.5 and waypoints[:, 5][0] >= last_speed + 0.05) :
            last_speed = waypoints[:, 5][0]

            gp_model_trained += 1
            print("GP training...")
            num_of_new_samples = 250

            if gp_mpc_type == 'cartesian':
                print(f"{len(planner_gp_mpc.model.x_measurements[0])}")
                planner_gp_mpc.model.train_gp_min_variance(num_of_new_samples)
            elif gp_mpc_type == 'frenet':
                print(f"{len(planner_gp_mpc_frenet.model.x_measurements[0])}")
                planner_gp_mpc_frenet.model.train_gp_min_variance(num_of_new_samples)

            print("GP training done")
            print('Model used: GP')
            print('Reference speed: %f' % waypoints[:, 5][0])

            log_dataset['X0'] = planner_gp_mpc.model.x_samples[0]
            log_dataset['X1'] = planner_gp_mpc.model.x_samples[1]
            log_dataset['X2'] = planner_gp_mpc.model.x_samples[2]
            log_dataset['X3'] = planner_gp_mpc.model.x_samples[3]
            log_dataset['X4'] = planner_gp_mpc.model.x_samples[4]
            log_dataset['X5'] = planner_gp_mpc.model.x_samples[5]
            log_dataset['Y0'] = planner_gp_mpc.model.y_samples[0]
            log_dataset['Y1'] = planner_gp_mpc.model.y_samples[1]
            log_dataset['Y2'] = planner_gp_mpc.model.y_samples[2]

            with open('log01', 'w') as f:
                json.dump(log, f)
            with open('testing_dataset', 'w') as f:
                json.dump(log_dataset, f)

        if obs['lap_counts'][0] - 1 == laps_done:
            # planner_gp_mpc_frenet.add_safe_trajectory(np.array([xcl]), np.array([ucl]))
            # xcl = []
            # ucl = []
            laps_done += 1
            print("Now")

        if obs['lap_counts'][0] == 60:
            done = 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
    with open('log01', 'w') as f:
        json.dump(log, f)


if __name__ == '__main__':
    main()

# formula zero paper
# exp2
