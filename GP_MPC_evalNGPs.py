import time
import yaml
import gym
import sys
from argparse import Namespace
from regulators.pure_pursuit import *
from regulators.path_follow_mpc import *
from models.kinematic import KinematicModel
from models.extended_kinematic import ExtendedKinematicModel
from models.GP_model_ensembling_NGPs import GPEnsembleModelsNGPs
from helpers.closest_point import *
import torch
import gpytorch
import os
import numpy as np

from datetime import datetime
from pyglet.gl import GL_POINTS
import pyglet
import json
import time
#
# SAVE_MODELS = False
# PRETRAINED = False
# PRETRAINED_NAMES = {'model1': ['gp107-10-2022_18:02:54', 'gp1_likelihood07-10-2022_18:02:54'],
#                     'model2': ['gp207-10-2022_18:02:54', 'gp2_likelihood07-10-2022_18:02:54']}


@dataclass
class MPCConfigGP:
    NXK: int = 7  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    NU: int = 2  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 20  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.000000002, 2.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.000000003, 2.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([26.5, 26.5, 20.0, 20.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # state error cost matrix, for the next (T) prediction time steps
    Qfk: list = field(
        default_factory=lambda: np.diag([26.5, 26.5, 20.0, 20.0, 0.0, 0.0, 0.0])
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
    map_name = 'DualLaneChange'  # SaoPaulo, rounded_rectangle, l_shape, DualLaneChange
    use_dyn_friction = False
    control_step = 100.0  # ms
    render_every = 1  # render graphics every n control steps
    constant_friction = 1.1
    constant_speed = True
    # datasets = ['dataset_lShape_0_5_100ms_v2', 'dataset_lShape_1_1_100ms_v2']
    datasets = ['dataset_DualLaneChange_0_5_100ms_v3',
                'dataset_DualLaneChange_0_7_100ms_v2',
                'dataset_DualLaneChange_1_1_100ms_v2']
    N_HIST = 10
    EPS = 0.0
    NUM_MODELS = len(datasets)

    # Load map config file
    with open('configs/config_%s.yaml' % map_name) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    if use_dyn_friction:
        tpamap_name = './maps/DualLaneChange/friction_data/DualLaneChange_track_tpamap.csv'
        # tpamap_name = './maps/l_shape/friction_data/l_shape_l720_track_tpamap.csv'
        # tpamap_name = './maps/SaoPaulo/friction_data/SaoPaulo_track_tpamap.csv'
        # tpamap_name = './maps/l_shape/friction_data/l_shape_friction_gen_input_tpamap.csv'
        tpadata_name = './maps/DualLaneChange/friction_data/DualLaneChange_track_tpadata.json'
        # tpadata_name = './maps/l_shape/friction_data/l_shape_l720_track_tpadata.json'
        # tpadata_name = './maps/SaoPaulo/friction_data/SaoPaulo_track_tpadata.json'
        # tpadata_name = './maps/l_shape/friction_data/l_shape_friction_gen_input_tpadata.json'

        tpamap = np.loadtxt(tpamap_name, delimiter=';', skiprows=1)

        tpadata = {}
        with open(tpadata_name) as f:
            tpadata = json.load(f)

    raceline = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
    waypoints = np.array(raceline)

    waypoints[:, 3] += 1.5707963268

    if constant_speed:
        waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * 15.0
    else:
        waypoints[:, 5] *= 0.985
        waypoints[waypoints[:, 5] > 19.5, 5] = 19.5

    # init controllers
    planner_pp = PurePursuitPlanner(conf, 0.805975 + 1.50876)  # 0.805975 + 1.50876
    planner_pp.waypoints = waypoints

    planner_gp_mpc = STMPCPlanner(model=GPEnsembleModelsNGPs(config=MPCConfigGP(), n_models=NUM_MODELS), waypoints=waypoints,
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
        e.left = left - 1200
        e.right = right + 1200
        e.top = top + 1200
        e.bottom = bottom - 1200

        planner_pp.render_waypoints(e)
        draw.draw_debug(e)

    # MB - reference point: center of mass
    # dynamic_ST - reference point: center of mass

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                   num_agents=1, timestep=0.001, model='MB', drive_control_mode='acc',
                   steering_control_mode='vel')

    env.add_render_callback(render_callback)
    # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    spawn_point = 0
    obs, step_reward, done, info = env.reset(
        np.array([[waypoints[spawn_point, 1], waypoints[spawn_point, 2], waypoints[spawn_point, 3], 0.0, waypoints[spawn_point, 5], 0.0, 0.0]]))
    env.render()

    laptime = 0.0
    start = time.time()
    last_render = 0

    # init logger
    log = {'time': [], 'x': [], 'y': [], 'x_gt': [], 'y_gt': [], 'lap_n': [], 'vx': [], 'v_ref': [], 'vx_mean': [], 'vx_var': [], 'vy_mean': [],
           'vy_var': [], 'theta_mean': [], 'theta_var': [], 'true_vx': [], 'true_mu': [], 'true_vy': [], 'true_yaw_rate': [], 'tracking_error': [], 'w': []}

    log_dataset = {'X0': [], 'X1': [], 'X2': [], 'X3': [], 'X4': [], 'X5': [], 'Y0': [], 'Y1': [], 'Y2': []}

    # calc number of sim steps per one control step
    num_of_sim_steps = int(control_step / (env.timestep * 1000.0))

    for i in range(NUM_MODELS):
        with open(datasets[i], 'r') as f:
            data = json.load(f)

        planner_gp_mpc.model.gp_models[i].x_measurements[0] = data['X0']
        planner_gp_mpc.model.gp_models[i].x_measurements[1] = data['X1']
        planner_gp_mpc.model.gp_models[i].x_measurements[2] = data['X2']
        planner_gp_mpc.model.gp_models[i].x_measurements[3] = data['X3']
        planner_gp_mpc.model.gp_models[i].x_measurements[4] = data['X4']
        planner_gp_mpc.model.gp_models[i].x_measurements[5] = data['X5']
        planner_gp_mpc.model.gp_models[i].y_measurements[0] = data['Y0']
        planner_gp_mpc.model.gp_models[i].y_measurements[1] = data['Y1']
        planner_gp_mpc.model.gp_models[i].y_measurements[2] = data['Y2']
        print(len(planner_gp_mpc.model.gp_models[i].x_measurements[0]))
        scaled_x, scaled_y = planner_gp_mpc.model.gp_models[i].init_gp()
        print(f'train model {i}')
        print("GP training...")
        planner_gp_mpc.model.gp_models[i].train_gp(scaled_x, scaled_y, method=0)
        print("GP training done")

    # done training
    gp_models_trained = 1
    print('Model used: GP')
    print('Reference speed: %f' % waypoints[:, 5][0])

    gather_data = 0

    prev_means = np.zeros((N_HIST, 3, NUM_MODELS))
    prev_observations = np.zeros((N_HIST, 3))
    prev_w = np.ones((NUM_MODELS, 1)) / NUM_MODELS
    while not done:
        # Regulator step MPC
        vehicle_state = np.array([env.sim.agents[0].state[0],
                                  env.sim.agents[0].state[1],
                                  env.sim.agents[0].state[3],  # vx
                                  env.sim.agents[0].state[4],  # yaw angle
                                  env.sim.agents[0].state[10],  # vy
                                  env.sim.agents[0].state[5],  # yaw rate
                                  env.sim.agents[0].state[2],  # steering angle
                                  ]) + np.random.randn(7) * 0.0001

        mean, lower, upper = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        u = [0.0, 0.0]
        tracking_error = 0.0
        total_var = 0.0
        n_point = 0

        # if not gp_models_trained:
        if gp_models_trained:
            start = time.time()
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_gp_mpc.plan(vehicle_state)
            end = time.time()
            print(end-start)
            u[0] = u[0] / planner_gp_mpc.config.MASS  # Force to acceleration

            # draw predicted states and reference trajectory
            draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
            draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

            _, tracking_error, _, _, n_point = nearest_point_on_trajectory(np.array([env.sim.agents[0].state[0], env.sim.agents[0].state[1]]),
                                                                     np.array([waypoints[:, 1], waypoints[:, 2]]).T)
        if gp_models_trained:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # scaled_means = np.zeros((self.n_models, 3, 1))

                mean, lower, upper, prev_mean = planner_gp_mpc.model.scale_and_predict_model_step(vehicle_state, [u[0] * planner_gp_mpc.config.MASS, u[1]])

        # print('ax = %f  delta-v = %f' % (u[0], u[1]))

        # set correct friction to the environment
        if use_dyn_friction:
            min_id = get_closest_point_vectorized(np.array([obs['poses_x'][0], obs['poses_y'][0]]), np.array(tpamap))
            # env.params['tire_p_dy1'] = tpadata[str(min_id)][0] - 0.1  # mu_y
            env.params['tire_p_dy1'] = tpadata[str(min_id)][0] * 0.9  # mu_y
            env.params['tire_p_dx1'] = tpadata[str(min_id)][0]  # mu_x
        else:
            env.params['tire_p_dy1'] = constant_friction * 0.9  # mu_y
            # env.params['tire_p_dy1'] = constant_friction - 0.1  # mu_y
            env.params['tire_p_dx1'] = constant_friction  # mu_x

        # print(env.params['tire_p_dx1'])

        # Simulation step
        step_reward = 0.0
        for i in range(num_of_sim_steps):
            obs, rew, _, info = env.step(np.array([[u[1], u[0]]]))
            step_reward += rew
        laptime += step_reward

        vx_transition = env.sim.agents[0].state[3] + np.random.randn(1)[0] * 0.0001 - vehicle_state[2]
        vy_transition = env.sim.agents[0].state[10] + np.random.randn(1)[0] * 0.0001 - vehicle_state[4]
        yaw_rate_transition = env.sim.agents[0].state[5] + np.random.randn(1)[0] * 0.0001 - vehicle_state[5]

        Y_real = np.array([float(vx_transition), float(vy_transition), float(yaw_rate_transition)])

        # Roll previous means and observations
        prev_means = np.roll(prev_means, shift=1, axis=0)
        prev_observations = np.roll(prev_observations, shift=1, axis=0)

        # replace with new means
        for i in range(NUM_MODELS):
            prev_means[-1, :, i] = prev_mean[i].flatten()
        # prev_means[-1, :, 1] = prev_mean2.flatten()

        # replace with new observations
        prev_observations[-1, :] = Y_real.flatten()

        planner_gp_mpc.model.compute_w(prev_observations.reshape(-1, 1), prev_means.reshape(-1, NUM_MODELS),
                                       prev_w, EPS,
                                       np.array([u[0] * planner_gp_mpc.config.MASS, u[1]]))
        prev_w = planner_gp_mpc.model.w_var.value

        s_temp = 'tire_p_dx1'
        print(f'W: {planner_gp_mpc.model.w}   Ref speed: {waypoints[:, 5][n_point]}  Speed: {env.sim.agents[0].state[3]}   Friction: {env.params[s_temp]}')

        # Logging
        log['time'].append(laptime)
        log['lap_n'].append(obs['lap_counts'][0])
        log['x'].append(env.sim.agents[0].state[0])
        log['x_gt'].append(mpc_ref_path_x[0])
        log['y'].append(env.sim.agents[0].state[1])
        log['y_gt'].append(mpc_ref_path_y[0])
        log['true_mu'].append(env.params['tire_p_dx1'])
        log['vx'].append(env.sim.agents[0].state[3])
        log['v_ref'].append(waypoints[:, 5][n_point])
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
        log['w'].append(planner_gp_mpc.model.w.tolist())
        # log['w1'].append(float(planner_gp_mpc.model.w1))
        # log['w2'].append(float(planner_gp_mpc.model.w2))

        # Rendering
        last_render += 1
        if last_render >= render_every:
            last_render = 0
            env.render(mode='human_fast')

        if obs['lap_counts'][0] == gp_models_trained:
            gp_models_trained += 1
            with open('log01_eval', 'w') as f:
                json.dump(log, f)
            print('Log saved...')

        if obs['lap_counts'][0] == 30 or tracking_error > 10.0 or env.sim.agents[0].state[0] > 505:
            done = 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
    with open('log01_eval', 'w') as f:
        json.dump(log, f)


if __name__ == '__main__':
    main()

# formula zero paper
# exp2
