import time
import yaml
import gym
import sys
from argparse import Namespace
from regulators.pure_pursuit import *
from regulators.path_follow_mpc import *
from models.kinematic import KinematicModel
from models.extended_kinematic import ExtendedKinematicModel
from models.GP_model_ensembling_extended import GPEnsembleModel_v2
from helpers.closest_point import *
import torch
import gpytorch
import os
import numpy as np

from pyglet.gl import GL_POINTS
import pyglet
import json
import copy


@dataclass
class MPCConfigEXT:
    NXK: int = 7  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    NU: int = 2  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 22  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.0000001, 10.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.000000001, 10.0])
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
    NXK: int = 9  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle, ws rear, ws front]
    NU: int = 2  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 22  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.0000001, 2.1])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.00000001, 2.1])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # state error cost matrix, for the next (T) prediction time steps
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
    WHEEL_RADIUS: float = 0.344  # Wheel radius [m]
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


def main():  # after launching this you can run visualization.py to see the results
    """
    main entry point
    """

    # Choose program parameters
    model_in_first_lap = 'ext_kinematic'  # options: ext_kinematic, pure_pursuit
    map_name = 'l_shape'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape
    use_dyn_friction = False
    control_step = 100.0  # ms
    render_every = 1  # render graphics every n control steps
    constant_speed = False
    constant_friction = 0.6

    # Creating the single-track Motion planner and Controller

    # Init Pure-Pursuit regulator
    work = {'mass': 1225.88, 'lf': 0.80597534362552312, 'tlad': 10.6461887897713965, 'vgain': 1.0}

    # Load map config file
    with open('configs/config_%s.yaml' % map_name) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    if use_dyn_friction:
        tpamap_name = './maps/rounded_rectangle/rounded_rectangle_tpamap.csv'
        tpadata_name = './maps/rounded_rectangle/rounded_rectangle_tpadata.json'

        tpamap = np.loadtxt(tpamap_name, delimiter=';', skiprows=1)

        tpadata = {}
        with open(tpadata_name) as f:
            tpadata = json.load(f)

    raceline = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
    waypoints = np.array(raceline)

    waypoints[:, 3] += 1.5707963268

    if constant_speed:
        waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * 4.0
    else:
        waypoints[:, 5] *= 0.2

    # init controllers
    planner_pp = PurePursuitPlanner(conf, 0.805975 + 1.50876)  # 0.805975 + 1.50876
    planner_pp.waypoints = waypoints

    planner_gp_mpc = STMPCPlanner(model=GPEnsembleModel_v2(config=MPCConfigGP()), waypoints=waypoints,
                                  config=MPCConfigGP())

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
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner_pp.render_waypoints(e)
        draw.draw_debug(e)

    # MB - reference point: center of mass
    # dynamic_ST - reference point: center of mass

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                   num_agents=1, timestep=0.001, model='MB', drive_control_mode='acc',
                   steering_control_mode='vel')

    env.add_render_callback(render_callback)
    # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    obs, step_reward, done, info = env.reset(
        np.array([[waypoints[0, 1], waypoints[0, 2], waypoints[0, 3], 0.0, waypoints[0, 5], 0.0, 0.0]]))
    env.render()

    laptime = 0.0
    start = time.time()
    last_render = 0

    # init logger
    log = {'time': [], 'x': [], 'y': [], 'lap_n': [], 'vx': [], 'v_ref': [], 'vx_mean': [], 'vx_var': [], 'vy_mean': [],
           'wsr_mean': [], 'wsf_mean': [], 'vy_var': [], 'theta_mean': [], 'theta_var': [], 'true_vx': [], 'true_vy': [], 'true_yaw_rate': [],
           'tracking_error': [], 'true_wsr': [], 'true_wsf': []}

    log_dataset = {'X0': [], 'X1': [], 'X2': [], 'X3': [], 'X4': [], 'X5': [], 'X6': [], 'X7': [], 'Y0': [], 'Y1': [], 'Y2': [], 'Y3': [], 'Y4': [],
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

    original_velocity_profile = copy.deepcopy(waypoints[:, 5])

    while not done:

        # Regulator step MPC
        ws_rear = (env.sim.agents[0].state[25] + env.sim.agents[0].state[26]) / 2.0
        ws_front = (env.sim.agents[0].state[23] + env.sim.agents[0].state[24]) / 2.0

        vehicle_state_kin = np.array([env.sim.agents[0].state[0],
                                      env.sim.agents[0].state[1],
                                      env.sim.agents[0].state[3],  # vx
                                      env.sim.agents[0].state[4],  # yaw angle
                                      env.sim.agents[0].state[10],  # vy
                                      env.sim.agents[0].state[5],  # yaw rate
                                      env.sim.agents[0].state[2],  # steering angle
                                      ]) + np.random.randn(7) * 0.00001

        vehicle_state = np.array([env.sim.agents[0].state[0],
                                  env.sim.agents[0].state[1],
                                  env.sim.agents[0].state[3],  # vx
                                  env.sim.agents[0].state[4],  # yaw angle
                                  env.sim.agents[0].state[10],  # vy
                                  env.sim.agents[0].state[5],  # yaw rate
                                  env.sim.agents[0].state[2],  # steering angle
                                  ws_rear,  # wheel speed rear
                                  ws_front,  # wheel speed front
                                  ]) + np.random.randn(9) * 0.00001

        mean, lower, upper = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        u = [0.0, 0.0]
        tracking_error = 0.0
        total_var = 0.0
        if not gp_model_trained >= 2:
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
                    vehicle_state_kin)
                u[0] = u[0] / planner_gp_mpc.config.MASS  # Force to acceleration

                # draw predicted states and reference trajectory
                draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
                draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

            u[0] += np.random.randn(1)[0] * 0.001
            u[1] += np.random.randn(1)[0] * 0.005

        else:
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_gp_mpc.plan(
                vehicle_state)
            u[0] = u[0] / planner_gp_mpc.config.MASS  # Force to acceleration

            if waypoints[:, 5][0] <= 5.5:
                u[0] += np.random.randn(1)[0] * 0.00005
                u[1] += np.random.randn(1)[0] * 0.005
            elif waypoints[:, 5][0] < 6.8:
                u[0] += np.random.randn(1)[0] * 0.00001
                u[1] += np.random.randn(1)[0] * 0.0001

            # draw predicted states and reference trajectory
            draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
            draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

            _, tracking_error, _, _, _ = nearest_point_on_trajectory(np.array([mpc_pred_x[0], mpc_pred_y[0]]),
                                                                     np.array([mpc_ref_path_x[0:2], mpc_ref_path_y[0:2]]).T)
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
        laptime += step_reward

        if map_name == 'l_shape':
            if constant_speed:
                if waypoints[:, 5][0] < 7.5:
                    waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.0006
                else:
                    waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.0003
            else:
                if waypoints[:, 5][0] < 7.5:
                    waypoints[:, 5] += original_velocity_profile * 0.0001
                else:
                    waypoints[:, 5] += original_velocity_profile * 0.00005

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
            log['wsf_mean'].append(float(mean[4]))
            log['wsr_mean'].append(float(mean[3]))
            log['true_wsf'].append((env.sim.agents[0].state[23] + env.sim.agents[0].state[24]) / 2.0 - vehicle_state[8])
            log['true_wsr'].append((env.sim.agents[0].state[25] + env.sim.agents[0].state[26]) / 2.0 - vehicle_state[7])
            logged_data = 0

        # Rendering
        last_render += 1
        if last_render >= render_every:
            last_render = 0
            env.render(mode='human_fast')

        # learning GPs
        u[0] = u[0] * planner_gp_mpc.config.MASS  # Acceleration to force

        if gp_model_trained:
            if abs((mean[0] - lower[0]) / planner_gp_mpc.model.scaler_y[0].std) >= 2.0 or abs(
                    (mean[1] - lower[1]) / planner_gp_mpc.model.scaler_y[1].std) >= 2.0 or abs(
                (mean[2] - lower[2]) / planner_gp_mpc.model.scaler_y[2].std) >= 2.0 or abs(
                (mean[3] - lower[3]) / planner_gp_mpc.model.scaler_y[2].std) >= 2.0 or abs(
                (mean[4] - lower[4]) / planner_gp_mpc.model.scaler_y[2].std) >= 2.0:
                gather_data_every = 3
            elif 1.0 <= abs((mean[0] - lower[0]) / planner_gp_mpc.model.scaler_y[0].std) < 2.0 or 1.0 <= abs(
                    (mean[1] - lower[1]) / planner_gp_mpc.model.scaler_y[1].std) < 2.0 or 1.0 <= abs(
                (mean[2] - lower[2]) / planner_gp_mpc.model.scaler_y[2].std) < 2.0 or 1.0 <= abs(
                (mean[3] - lower[3]) / planner_gp_mpc.model.scaler_y[3].std) < 2.0 or 1.0 <= abs(
                (mean[4] - lower[4]) / planner_gp_mpc.model.scaler_y[4].std) < 2.0:
                gather_data_every = 4
            else:
                if abs(env.sim.agents[0].state[10]) > 0.6 or abs(env.sim.agents[0].state[2]) > 0.1:
                    gather_data_every = 2
                else:
                    gather_data_every = 15
        else:
            gather_data_every = 3

        vx_transition = env.sim.agents[0].state[3] + np.random.randn(1)[0] * 0.00001 - vehicle_state[2]
        vy_transition = env.sim.agents[0].state[10] + np.random.randn(1)[0] * 0.00001 - vehicle_state[4]
        yaw_rate_transition = env.sim.agents[0].state[5] + np.random.randn(1)[0] * 0.00001 - vehicle_state[5]
        wheel_speed_rear_transition = (env.sim.agents[0].state[25] + env.sim.agents[0].state[26]) / 2.0 + np.random.randn(1)[0] * 0.00001 - vehicle_state[7]
        wheel_speed_front_transition = (env.sim.agents[0].state[23] + env.sim.agents[0].state[24]) / 2.0 + np.random.randn(1)[0] * 0.00001 - vehicle_state[8]

        # print(mean[0] - vx_transition)
        # print(gather_data_every)
        print('V: %f  Vx: %f  Vy: %f ' % (waypoints[:, 5][0], env.sim.agents[0].state[3], env.sim.agents[0].state[10]))

        gather_data += 1
        if gather_data >= gather_data_every:
            Y_sample = np.array([float(vx_transition), float(vy_transition), float(yaw_rate_transition),
                                 float(wheel_speed_rear_transition), float(wheel_speed_front_transition)])
            X_sample = np.array([float(vehicle_state[2]), float(vehicle_state[4]), float(vehicle_state[5]), float(vehicle_state[6]),
                                 float(u[0]), float(u[1]), float(vehicle_state[7]), float(vehicle_state[8])])

            planner_gp_mpc.model.add_new_datapoint(X_sample, Y_sample)
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

        X_t_1 = [float(vehicle_state[2]), float(vehicle_state[4]), float(vehicle_state[5]), float(vehicle_state[6]), float(u[0]), float(u[1]),
                 float(vx_transition), float(vy_transition), float(yaw_rate_transition)]

        # if obs['lap_counts'][0] - 1 == gp_model_trained:
        if (waypoints[:, 5][0] >= last_speed + 0.25 and obs['lap_counts'][0] > 0) or \
                (waypoints[:, 5][0] >= last_speed + 0.05 and waypoints[:, 5][0] > 7.5) or (
                obs['lap_counts'][0] - 1 == gp_model_trained and waypoints[:, 5][0] >= 7.5):
            last_speed = waypoints[:, 5][0]
            print(len(planner_gp_mpc.model.x_measurements[0]))
            gp_model_trained += 1
            print("GP training...")
            # scaled_x1, scaled_y1 = planner_gp_mpc.model.init_gp()
            # planner_gp_mpc.model.train_gp(scaled_x1, scaled_y1)

            if waypoints[:, 5][0] > 7.5:
                num_of_new_samples = 8
            else:
                num_of_new_samples = 50

            planner_gp_mpc.model.train_gp_min_variance(num_of_new_samples)
            print("GP training done")
            # obs, step_reward, done, info = env.reset(np.array([[conf.sx * 10, conf.sy * 10, conf.stheta, 0.0, 10.0,
            # 0.0, 0.0]]))
            print('Model used: GP')
            print('Reference speed: %f' % waypoints[:, 5][0])

            log_dataset['X0'] = planner_gp_mpc.model.x_samples[0]
            log_dataset['X1'] = planner_gp_mpc.model.x_samples[1]
            log_dataset['X2'] = planner_gp_mpc.model.x_samples[2]
            log_dataset['X3'] = planner_gp_mpc.model.x_samples[3]
            log_dataset['X4'] = planner_gp_mpc.model.x_samples[4]
            log_dataset['X5'] = planner_gp_mpc.model.x_samples[5]
            log_dataset['X6'] = planner_gp_mpc.model.x_samples[6]
            log_dataset['X7'] = planner_gp_mpc.model.x_samples[7]
            log_dataset['Y0'] = planner_gp_mpc.model.y_samples[0]
            log_dataset['Y1'] = planner_gp_mpc.model.y_samples[1]
            log_dataset['Y2'] = planner_gp_mpc.model.y_samples[2]
            log_dataset['Y3'] = planner_gp_mpc.model.y_samples[3]
            log_dataset['Y4'] = planner_gp_mpc.model.y_samples[4]

            with open('log01', 'w') as f:
                json.dump(log, f)
            with open('dataset_driving_final_0_6_100ms_ext_min_curv', 'w') as f:
                json.dump(log_dataset, f)

        if obs['lap_counts'][0] == 40:
            done = 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
    with open('log01', 'w') as f:
        json.dump(log, f)


if __name__ == '__main__':
    main()

# formula zero paper
# exp2
