import time
import yaml
import gym
import sys
from argparse import Namespace
from regulators.pure_pursuit import *
from regulators.path_follow_mpc import *
from models.kinematic import KinematicModel
from models.extended_kinematic import ExtendedKinematicModel
from models.GP_model_ensembleing import GPEnsembleModel
from helpers.closest_point import *
import torch
import gpytorch
import os
import numpy as np

from pyglet.gl import GL_POINTS
import pyglet
import json


@dataclass
class MPCConfigEXT:
    NXK: int = 7  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    NU: int = 2  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 10  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.0000007, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.0000001, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # state error cost matrix, for the next (T) prediction time steps
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0])
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
        default_factory=lambda: np.diag([0.000005, 20.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.0000005, 20.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # state error cost matrix, for the next (T) prediction time steps
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0])
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
    map_name = 'rounded_rectangle'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape
    use_dyn_friction = False
    control_step = 100.0  # ms
    render_every = 1  # render graphics every n control steps
    constant_speed = True

    # Creating the single-track Motion planner and Controller

    # Init Pure-Pursuit regulator
    work = {'mass': 1225.88, 'lf': 0.80597534362552312, 'tlad': 10.6461887897713965, 'vgain': 1.0}

    # Load map config file
    with open('config_%s.yaml' % map_name) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    if use_dyn_friction:
        tpamap_name = './maps/rounded_rectangle/rounded_rectangle_tpamap.csv'
        tpadata_name = './maps/rounded_rectangle/rounded_rectangle_tpadata.json'

        tpamap = np.loadtxt(tpamap_name, delimiter=';', skiprows=1)
        tpamap *= 1.5  # map is 1.5 times larger than normal

        tpadata = {}
        with open(tpadata_name) as f:
            tpadata = json.load(f)

    raceline = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
    waypoints = np.array(raceline)

    if constant_speed:
        # waypoints[:, 5] = waypoints[:, 5] * 2.0
        waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * 11.2

    # init controllers
    planner_pp = PurePursuitPlanner(conf, 0.805975 + 1.50876)  # 0.805975 + 1.50876
    planner_pp.waypoints = waypoints

    planner_gp_mpc = STMPCPlanner(model=GPEnsembleModel(config=MPCConfigGP()), waypoints=waypoints,
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
           'vy_var': [], 'theta_mean': [], 'theta_var': [], 'true_vx': [], 'true_vy': [], 'true_yaw_rate': [], 'tracking_error': []}

    log_dataset = {'X0': [], 'X1': [], 'X2': [], 'X3': [], 'X4': [], 'X5': [], 'Y0': [], 'Y1': [], 'Y2': []}

    # calc number of sim steps per one control step
    num_of_sim_steps = int(control_step / (env.timestep * 1000.0))

    gp_model_trained = 0
    gather_data = 0

    print('Model used: %s' % model_in_first_lap)

    while not done:

        # Regulator step MPC
        vehicle_state = np.array([env.sim.agents[0].state[0],
                                  env.sim.agents[0].state[1],
                                  env.sim.agents[0].state[3],  # vx
                                  env.sim.agents[0].state[4],  # yaw angle
                                  env.sim.agents[0].state[10],  # vy
                                  env.sim.agents[0].state[5],  # yaw rate
                                  env.sim.agents[0].state[2],  # steering angle
                                  ]) + np.random.randn(7) * 0.0000001

        u = [0.0, 0.0]
        if not gp_model_trained:
            if model_in_first_lap == 'pure_pursuit':
                # Regulator step pure pursuit
                speed, steer_angle = planner_pp.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                                                     work['tlad'],
                                                     work['vgain'])

                error_steer = steer_angle - env.sim.agents[0].state[2]
                u[1] = 10.0 * error_steer

                error_drive = 12.0 - env.sim.agents[0].state[3]
                u[0] = 12.0 * error_drive

            elif model_in_first_lap == "ext_kinematic":
                u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_ekin_mpc.plan(
                    vehicle_state)
                u[0] = u[0] / planner_gp_mpc.config.MASS  # Force to acceleration

                # draw predicted states and reference trajectory
                draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
                draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

            if obs['lap_counts'][0] == 0:
                u[0] += np.random.randn(1)[0] * 0.1
                u[1] += np.random.randn(1)[0] * 0.04


        else:
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_gp_mpc.plan(
                vehicle_state)
            u[0] = u[0] / planner_gp_mpc.config.MASS  # Force to acceleration

            if waypoints[:, 5][0] <= 11.6:
                u[0] += np.random.randn(1)[0] * 0.2
                u[1] += np.random.randn(1)[0] * 0.00
            elif waypoints[:, 5][0] < 15.0:
                u[0] += np.random.randn(1)[0] * 0.1
                u[1] += np.random.randn(1)[0] * 0.0

            # draw predicted states and reference trajectory
            draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
            draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

        # set correct friction to the environment
        if use_dyn_friction:
            min_id = get_closest_point_vectorized(np.array([obs['poses_x'][0], obs['poses_y'][0]]), np.array(tpamap))
            env.params['tire_p_dy1'] = tpadata[str(min_id)][0]  # mu_y
            env.params['tire_p_dx1'] = tpadata[str(min_id)][0] * 1.1  # mu_x
        else:
            env.params['tire_p_dy1'] = 0.7  # mu_y
            env.params['tire_p_dx1'] = 0.8  # mu_x

        # Simulation step
        step_reward = 0.0
        for i in range(num_of_sim_steps):
            obs, rew, _, info = env.step(np.array([[u[1], u[0]]]))
            step_reward += rew
        laptime += step_reward

        if obs['lap_counts'][0] > 0 and waypoints[:, 5][0] < 15.0:
            waypoints[:, 5] += np.ones((waypoints[:, 5].shape[0],)) * 0.0004

        # Logging
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

        # Rendering
        last_render += 1
        if last_render >= render_every:
            last_render = 0
            env.render(mode='human_fast')

        # learning GPs
        u[0] = u[0] * planner_gp_mpc.config.MASS  # Acceleration to force

        if gp_model_trained:
            if abs((mean[0] - lower[0]) / planner_gp_mpc.model.scaler_y[0].std) >= 2.0 or abs((mean[1] - lower[1]) / planner_gp_mpc.model.scaler_y[1].std) >= 2.0 or abs((mean[2] - lower[2]) / planner_gp_mpc.model.scaler_y[2].std) >= 2.0:
                gather_data_every = 3
            elif 0.95 <= abs((mean[0] - lower[0]) / planner_gp_mpc.model.scaler_y[0].std) < 2.0 or 0.95 <= abs((mean[1] - lower[1]) / planner_gp_mpc.model.scaler_y[1].std) < 2.0 or 0.95 <= abs((mean[2] - lower[2]) / planner_gp_mpc.model.scaler_y[2].std) < 2.0:
                gather_data_every = 6
            else:
                if abs(env.sim.agents[0].state[2]) > 0.09:
                    gather_data_every = 1
                else:
                    gather_data_every = 20
        else:
            gather_data_every = 5

        print(gather_data_every)

        gather_data += 1
        if gather_data >= gather_data_every:
            vx_transition = env.sim.agents[0].state[3] + np.random.randn(1)[0] * 0.001 - vehicle_state[2]
            vy_transition = env.sim.agents[0].state[10] + np.random.randn(1)[0] * 0.001 - vehicle_state[4]
            yaw_rate_transition = env.sim.agents[0].state[5] + np.random.randn(1)[0] * 0.001 - vehicle_state[5]

            Y_sample = np.array([float(vx_transition), float(vy_transition), float(yaw_rate_transition)])
            X_sample = np.array([float(vehicle_state[2]), float(vehicle_state[4]),
                                 float(vehicle_state[5]), float(vehicle_state[6]), float(u[0]), float(u[1])])

            planner_gp_mpc.model.add_new_datapoint(X_sample, Y_sample)
            gather_data = 0

        # if obs['lap_counts'][0] == 2 and gp_model_trained == 0 or obs['lap_counts'][0] == 3 and gp_model_trained == 1:
        if obs['lap_counts'][0] - 1 == gp_model_trained:
            print(len(planner_gp_mpc.model.vx))
            gp_model_trained += 1
            print("GP training...")
            planner_gp_mpc.model.train_gp()
            print("GP training done")
            # obs, step_reward, done, info = env.reset(np.array([[conf.sx * 10, conf.sy * 10, conf.stheta, 0.0, 10.0,
            # 0.0, 0.0]]))
            print('Model used: GP')
            print('Reference speed: %f' % waypoints[:, 5][0])

            log_dataset['X0'] = planner_gp_mpc.model.x_measurements[0]
            log_dataset['X1'] = planner_gp_mpc.model.x_measurements[1]
            log_dataset['X2'] = planner_gp_mpc.model.x_measurements[2]
            log_dataset['X3'] = planner_gp_mpc.model.x_measurements[3]
            log_dataset['X4'] = planner_gp_mpc.model.x_measurements[4]
            log_dataset['X5'] = planner_gp_mpc.model.x_measurements[5]
            log_dataset['Y0'] = planner_gp_mpc.model.y_measurements[0]
            log_dataset['Y1'] = planner_gp_mpc.model.y_measurements[1]
            log_dataset['Y2'] = planner_gp_mpc.model.y_measurements[2]

            with open('log01', 'w') as f:
                json.dump(log, f)
            # with open('dataset_1_1', 'w') as f:
            #     json.dump(log_dataset, f)

        if obs['lap_counts'][0] == 20:
            done = 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
    with open('log01', 'w') as f:
        json.dump(log, f)


if __name__ == '__main__':
    main()

# formula zero paper
# exp2
