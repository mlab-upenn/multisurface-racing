import time
import yaml
import gym
import sys
from argparse import Namespace
from regulators.pure_pursuit import *
from regulators.path_follow_mpc import *
from models.kinematic import KinematicModel
from models.extended_kinematic import ExtendedKinematicModel
from models.GP_model_ensembling_2GPs_extended import GPEnsembleModels2GPsExtended
from helpers.closest_point import *
import torch
import gpytorch
import os
import numpy as np

from datetime import datetime
from pyglet.gl import GL_POINTS
import pyglet
import json

SAVE_MODELS = False
PRETRAINED = False
PRETRAINED_NAMES = {'model1': ['gp116-10-2022_02:07:05', 'gp1_likelihood16-10-2022_02:07:05'],
                    'model2': ['gp216-10-2022_02:07:05', 'gp2_likelihood16-10-2022_02:07:05']}


@dataclass
class MPCConfigGP:
    NXK: int = 9  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle, ws rear, ws front]
    NU: int = 2  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 40  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.000001, 2.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.0000001, 1.5])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([20.5, 20.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
    map_name = 'l_shape'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape
    use_dyn_friction = True
    control_step = 100.0  # ms
    render_every = 1  # render graphics every n control steps
    constant_friction = 0.4
    constant_speed = False
    NUM_MODELS = 2
    N_HIST = 30
    EPS = 0.0000000001

    # Load map config file
    with open('configs/config_%s.yaml' % map_name) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    if use_dyn_friction:
        tpamap_name = './maps/l_shape/friction_data/l_shape_friction_gen_input_tpamap.csv'
        tpadata_name = './maps/l_shape/friction_data/l_shape_friction_gen_input_tpadata.json'

        tpamap = np.loadtxt(tpamap_name, delimiter=';', skiprows=1)
        tpamap *= 1.0  # map is 1.5 times larger than normal

        tpadata = {}
        with open(tpadata_name) as f:
            tpadata = json.load(f)

    raceline = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
    waypoints = np.array(raceline)

    waypoints[:, 3] += 1.5707963268

    if constant_speed:
        waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * 6.8
    else:
        waypoints[:, 5] *= 0.3
    # init controllers
    planner_pp = PurePursuitPlanner(conf, 0.805975 + 1.50876)  # 0.805975 + 1.50876
    planner_pp.waypoints = waypoints

    planner_gp_mpc = STMPCPlanner(model=GPEnsembleModels2GPsExtended(config=MPCConfigGP()), waypoints=waypoints,
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
    obs, step_reward, done, info = env.reset(
        np.array([[waypoints[0, 1], waypoints[0, 2], waypoints[0, 3], 0.0, waypoints[0, 5], 0.0, 0.0]]))
    env.render()

    laptime = 0.0
    start = time.time()
    last_render = 0

    # init logger
    log = {'time': [], 'x': [], 'y': [], 'lap_n': [], 'vx': [], 'v_ref': [], 'vx_mean': [], 'vx_var': [], 'vy_mean': [],
           'vy_var': [], 'theta_mean': [], 'theta_var': [], 'true_vx': [], 'true_vy': [], 'true_yaw_rate': [], 'tracking_error': [], 'w1': [], 'w2': []}

    log_dataset = {'X0': [], 'X1': [], 'X2': [], 'X3': [], 'X4': [], 'X5': [], 'Y0': [], 'Y1': [], 'Y2': []}

    # calc number of sim steps per one control step
    num_of_sim_steps = int(control_step / (env.timestep * 1000.0))

    with open('dataset_driving_final_0_1_100ms_ext_min_curv', 'r') as f:
        data = json.load(f)

    planner_gp_mpc.model.gp_model1.x_measurements[0] = data['X0']
    planner_gp_mpc.model.gp_model1.x_measurements[1] = data['X1']
    planner_gp_mpc.model.gp_model1.x_measurements[2] = data['X2']
    planner_gp_mpc.model.gp_model1.x_measurements[3] = data['X3']
    planner_gp_mpc.model.gp_model1.x_measurements[4] = data['X4']
    planner_gp_mpc.model.gp_model1.x_measurements[5] = data['X5']
    planner_gp_mpc.model.gp_model1.x_measurements[6] = data['X6']
    planner_gp_mpc.model.gp_model1.x_measurements[7] = data['X7']
    planner_gp_mpc.model.gp_model1.y_measurements[0] = data['Y0']
    planner_gp_mpc.model.gp_model1.y_measurements[1] = data['Y1']
    planner_gp_mpc.model.gp_model1.y_measurements[2] = data['Y2']
    planner_gp_mpc.model.gp_model1.y_measurements[3] = data['Y3']
    planner_gp_mpc.model.gp_model1.y_measurements[4] = data['Y4']
    print(len(planner_gp_mpc.model.gp_model1.x_measurements[0]))
    scaled_x1, scaled_y1 = planner_gp_mpc.model.gp_model1.init_gp()

    with open('dataset_driving_final_0_6_100ms_ext_min_curv', 'r') as f:
        data = json.load(f)

    planner_gp_mpc.model.gp_model2.x_measurements[0] = data['X0']
    planner_gp_mpc.model.gp_model2.x_measurements[1] = data['X1']
    planner_gp_mpc.model.gp_model2.x_measurements[2] = data['X2']
    planner_gp_mpc.model.gp_model2.x_measurements[3] = data['X3']
    planner_gp_mpc.model.gp_model2.x_measurements[4] = data['X4']
    planner_gp_mpc.model.gp_model2.x_measurements[5] = data['X5']
    planner_gp_mpc.model.gp_model2.x_measurements[6] = data['X6']
    planner_gp_mpc.model.gp_model2.x_measurements[7] = data['X7']
    planner_gp_mpc.model.gp_model2.y_measurements[0] = data['Y0']
    planner_gp_mpc.model.gp_model2.y_measurements[1] = data['Y1']
    planner_gp_mpc.model.gp_model2.y_measurements[2] = data['Y2']
    planner_gp_mpc.model.gp_model2.y_measurements[3] = data['Y3']
    planner_gp_mpc.model.gp_model2.y_measurements[4] = data['Y4']
    print(len(planner_gp_mpc.model.gp_model2.x_measurements[0]))
    scaled_x2, scaled_y2 = planner_gp_mpc.model.gp_model2.init_gp()

    gp_models_trained = 0
    # learn model from stored measurements
    if not PRETRAINED:
        print('train model 1')
        print("GP training...")
        planner_gp_mpc.model.gp_model1.train_gp(scaled_x1, scaled_y1, method=1)
        print("GP training done")
        print('Model used: GP')
        print('Reference speed: %f' % waypoints[:, 5][0])

        print('train model 2')
        print("GP training...")
        planner_gp_mpc.model.gp_model2.train_gp(scaled_x2, scaled_y2, method=1)
        print("GP training done")
        print('Model used: GP')
        print('Reference speed: %f' % waypoints[:, 5][0])

        if SAVE_MODELS:
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

            torch.save(planner_gp_mpc.model.gp_model1.gp_model.state_dict(), 'gp1' + dt_string + '.pth')
            torch.save(planner_gp_mpc.model.gp_model1.gp_likelihood.state_dict(), 'gp1_likelihood' + dt_string + '.pth')
            torch.save(planner_gp_mpc.model.gp_model2.gp_model.state_dict(), 'gp2' + dt_string + '.pth')
            torch.save(planner_gp_mpc.model.gp_model2.gp_likelihood.state_dict(), 'gp2_likelihood' + dt_string + '.pth')

        # done training
        gp_models_trained = 1

    else:
        # If you have pretrained models, load them
        model1_names = PRETRAINED_NAMES['model1']
        model2_names = PRETRAINED_NAMES['model2']

        # Load model1
        state_dict_gp1 = torch.load('trained_models/' + model1_names[0] + '.pth').copy()
        planner_gp_mpc.model.gp_model1.gp_model.load_state_dict(state_dict_gp1)

        state_dict_likelihood1 = torch.load('trained_models/' + model1_names[1] + '.pth').copy()
        planner_gp_mpc.model.gp_model1.gp_likelihood.load_state_dict(state_dict_likelihood1)

        planner_gp_mpc.model.gp_model1.eval()
        planner_gp_mpc.model.gp_model1.cuda()

        # Load model2
        state_dict_gp2 = torch.load('trained_models/' + model2_names[0] + '.pth').copy()
        planner_gp_mpc.model.gp_model2.gp_model.load_state_dict(state_dict_gp2)

        state_dict_likelihood2 = torch.load('trained_models/' + model2_names[1] + '.pth').copy()
        planner_gp_mpc.model.gp_model2.gp_likelihood.load_state_dict(state_dict_likelihood2)

        planner_gp_mpc.model.gp_model2.eval()
        planner_gp_mpc.model.gp_model2.cuda()
        # Pretrained so yes
        gp_models_trained = 1

    gather_data = 0

    prev_means = np.zeros((N_HIST, 5, NUM_MODELS))
    prev_observations = np.zeros((N_HIST, 5))
    prev_w = np.ones((NUM_MODELS, 1)) / NUM_MODELS
    while not done:
        # Regulator step MPC
        ws_rear = (env.sim.agents[0].state[25] + env.sim.agents[0].state[26]) / 2.0
        ws_front = (env.sim.agents[0].state[23] + env.sim.agents[0].state[24]) / 2.0

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

        mean, lower, upper = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        u = [0.0, 0.0]
        tracking_error = 0.0
        total_var = 0.0
        # if not gp_models_trained:
        if gp_models_trained:
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_gp_mpc.plan(vehicle_state)
            u[0] = u[0] / planner_gp_mpc.config.MASS  # Force to acceleration

            # draw predicted states and reference trajectory
            draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
            draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

            _, tracking_error, _, _, _ = nearest_point_on_trajectory(np.array([mpc_pred_x[0], mpc_pred_y[0]]),
                                                                     np.array([mpc_ref_path_x[0:2], mpc_ref_path_y[0:2]]).T)
        if gp_models_trained:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mean, lower, upper, prev_mean1, prev_mean2 = planner_gp_mpc.model.scale_and_predict_model_step(vehicle_state,
                                                                                                               [u[0] * planner_gp_mpc.config.MASS, u[1]])

        # print('ax = %f  delta-v = %f' % (u[0], u[1]))

        # set correct friction to the environment
        if use_dyn_friction:
            min_id = get_closest_point_vectorized(np.array([obs['poses_x'][0], obs['poses_y'][0]]), np.array(tpamap))
            if tpadata[str(min_id)][0] == 0.9:
                env.params['tire_p_dy1'] = 0.6 * 0.9
                env.params['tire_p_dx1'] = 0.6
            else:
                env.params['tire_p_dy1'] = 0.1 * 0.9
                env.params['tire_p_dx1'] = 0.1
            # env.params['tire_p_dy1'] = tpadata[str(min_id)][0] * 0.9  # mu_y
            # env.params['tire_p_dx1'] = tpadata[str(min_id)][0]  # mu_x
        else:
            env.params['tire_p_dy1'] = constant_friction * 0.9  # mu_y
            env.params['tire_p_dx1'] = constant_friction  # mu_x


        # Simulation step
        step_reward = 0.0
        for i in range(num_of_sim_steps):
            obs, rew, _, info = env.step(np.array([[u[1], u[0]]]))
            step_reward += rew
        laptime += step_reward

        vx_transition = env.sim.agents[0].state[3] + np.random.randn(1)[0] * 0.00001 - vehicle_state[2]
        vy_transition = env.sim.agents[0].state[10] + np.random.randn(1)[0] * 0.00001 - vehicle_state[4]
        yaw_rate_transition = env.sim.agents[0].state[5] + np.random.randn(1)[0] * 0.00001 - vehicle_state[5]
        wheel_speed_rear_transition = (env.sim.agents[0].state[25] + env.sim.agents[0].state[26]) / 2.0 + np.random.randn(1)[0] * 0.00001 - \
                                          vehicle_state[7]
        wheel_speed_front_transition = (env.sim.agents[0].state[23] + env.sim.agents[0].state[24]) / 2.0 + np.random.randn(1)[0] * 0.00001 - \
                                       vehicle_state[8]

        Y_real = np.array([float(vx_transition), float(vy_transition), float(yaw_rate_transition),
                           float(wheel_speed_rear_transition), float(wheel_speed_front_transition)])

        # Roll previous means and observations
        prev_means = np.roll(prev_means, shift=1, axis=0)
        prev_observations = np.roll(prev_observations, shift=1, axis=0)

        # replace with new means
        prev_means[-1, :, 0] = prev_mean1.flatten()
        prev_means[-1, :, 1] = prev_mean2.flatten()

        # replace with new observations
        prev_observations[-1, :] = Y_real.flatten()

        planner_gp_mpc.model.compute_w(prev_observations.reshape(-1, 1), prev_means.reshape(-1, NUM_MODELS),
                                       prev_w, EPS,
                                       np.array([u[0] * planner_gp_mpc.config.MASS, u[1]]))
        prev_w = planner_gp_mpc.model.w_var.value
        print('W1: %f    W2: %f    Speed: %f   Friction: %f' % (
            planner_gp_mpc.model.w1, planner_gp_mpc.model.w2, env.sim.agents[0].state[3], env.params['tire_p_dx1']))

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
        log['w1'].append(float(planner_gp_mpc.model.w1))
        log['w2'].append(float(planner_gp_mpc.model.w2))

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

        if obs['lap_counts'][0] == 30:
            done = 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
    with open('log01_eval', 'w') as f:
        json.dump(log, f)


if __name__ == '__main__':
    main()

# formula zero paper
# exp2
