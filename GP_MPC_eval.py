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
        default_factory=lambda: np.diag([0.0000005, 100.0])
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
    TK: int = 15  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.00001, 10.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.000001, 10.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([25.5, 25.5, 5.5, 0.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # state error cost matrix, for the next (T) prediction time steps
    Qfk: list = field(
        default_factory=lambda: np.diag([15.5, 15.5, 5.5, 0.0, 0.0, 0.0, 0.0])
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
    map_name = 'l_shape'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape
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
        if map_name == 'rounded_rectangle':
            waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * 10.0
        if map_name == 'l_shape':
            waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * 8.0

    # init controllers
    planner_pp = PurePursuitPlanner(conf, 0.805975 + 1.50876)  # 0.805975 + 1.50876
    planner_pp.waypoints = waypoints

    planner_gp_mpc = STMPCPlanner(model=GPEnsembleModel(config=MPCConfigGP()), waypoints=waypoints,
                                  config=MPCConfigGP())

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

    # learn model from stored measurements

    with open('dataset_1_1', 'r') as f:
        data = json.load(f)

    planner_gp_mpc.model.x_measurements[0] = data['X0']
    planner_gp_mpc.model.x_measurements[1] = data['X1']
    planner_gp_mpc.model.x_measurements[2] = data['X2']
    planner_gp_mpc.model.x_measurements[3] = data['X3']
    planner_gp_mpc.model.x_measurements[4] = data['X4']
    planner_gp_mpc.model.x_measurements[5] = data['X5']
    planner_gp_mpc.model.y_measurements[0] = data['Y0']
    planner_gp_mpc.model.y_measurements[1] = data['Y1']
    planner_gp_mpc.model.y_measurements[2] = data['Y2']

    print(len(planner_gp_mpc.model.x_measurements[0]))
    print("GP training...")
    planner_gp_mpc.model.train_gp()
    print("GP training done")
    gp_model_trained = 1
    print('Model used: GP')
    print('Reference speed: %f' % waypoints[:, 5][0])


    gather_data = 0

    while not done:

        # Regulator step MPC
        vehicle_state = np.array([env.sim.agents[0].state[0],
                                  env.sim.agents[0].state[1],
                                  env.sim.agents[0].state[3],  # vx
                                  env.sim.agents[0].state[4],  # yaw angle
                                  env.sim.agents[0].state[10],  # vy
                                  env.sim.agents[0].state[5],  # yaw rate
                                  env.sim.agents[0].state[2],  # steering angle
                                  ]) + np.random.randn(7) * 0.001

        mean, lower, upper = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        u = [0.0, 0.0]
        tracking_error = 0.0
        total_var = 0.0
        # if not gp_model_trained:
        if gp_model_trained:
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_gp_mpc.plan(
                vehicle_state)
            u[0] = u[0] / planner_gp_mpc.config.MASS  # Force to acceleration

            if waypoints[:, 5][0] <= 11.6:
                u[0] += np.random.randn(1)[0] * 0.01
                u[1] += np.random.randn(1)[0] * 0.001
            elif waypoints[:, 5][0] < 12.4:
                u[0] += np.random.randn(1)[0] * 0.005
                u[1] += np.random.randn(1)[0] * 0.001

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
            env.params['tire_p_dy1'] = tpadata[str(min_id)][0]  # mu_y
            env.params['tire_p_dx1'] = tpadata[str(min_id)][0] * 1.1  # mu_x
        else:
            env.params['tire_p_dy1'] = 0.4  # mu_y
            env.params['tire_p_dx1'] = 0.5  # mu_x

        # Simulation step
        step_reward = 0.0
        for i in range(num_of_sim_steps):
            obs, rew, _, info = env.step(np.array([[u[1], u[0]]]))
            step_reward += rew
        laptime += step_reward

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

        if obs['lap_counts'][0] == gp_model_trained:
            gp_model_trained += 1
            with open('log01_eval', 'w') as f:
                json.dump(log, f)
            print('Log saved...')

        if obs['lap_counts'][0] == 30:
            done = 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
    with open('log01_eval', 'w') as f:
        json.dump(log, f)


if __name__ == '__main__':
    main()

# formula zero paper
# exp2
