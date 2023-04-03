import time
import yaml
import gym
from argparse import Namespace
from regulators.pure_pursuit import *
from regulators.path_follow_mpc import *
from models.configs import MPCConfigGP, MPCConfigGPFrenet
from models.GP_model_ensembling import GPEnsembleModel
from models.GP_model_ensembling_frenet import GPEnsembleModelFrenet
from helpers.closest_point import *
import torch
import gpytorch
import numpy as np
from helpers.track import Track
import datetime

from helper.utils import save_GP_enemble_model
from helpers.draw_debug import DrawDebug
import json

def main():  # after launching this you can run visualization.py to see the results
    """
    main entry point
    """

    # Choose program parameters
    # currently only "custom_track" works for frenet
    map_name = 'custom_track'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape, BrandsHatch, DualLaneChange, custom_track
    use_dyn_friction = False
    gp_mpc_type = 'frenet'  # cartesian, frenet
    constant_friction = 0.7
    control_step = 100.0  # ms
    render_every = 30  # render graphics every n sim steps
    constant_speed = True

    SAVE_MODEL = False
    PRETRAINED = False
    PRETRAINED_MODEL = {'model': ['gp107-10-2022_18:02:54', 'gp1_likelihood07-10-2022_18:02:54']}


    # Creating the single-track Motion planner and Controller

    # Load map config file
    if not map_name == 'custom_track':
        with open('configs/config_%s.yaml' % map_name) as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    else:
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

        waypoints[:, 3] += 1.5707963268


    else:
        # Chose the same track used in GP_MPC.py
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

    if constant_speed:
        waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * 4.5
    else:
        waypoints[:, 5] *= 0.5


    # init controllers
    planner_pp = PurePursuitPlanner(conf, 0.805975 + 1.50876)  # 0.805975 + 1.50876
    planner_pp.waypoints = waypoints

    planner_gp_mpc = STMPCPlanner(model=GPEnsembleModel(config=MPCConfigGP()), waypoints=waypoints,
                                  config=MPCConfigGP())

    if gp_mpc_type == 'frenet':
        planner_gp_mpc_frenet = STMPCPlanner(model=GPEnsembleModelFrenet(config=MPCConfigGPFrenet(), track=track), waypoints=waypoints,
                                             config=MPCConfigGPFrenet(), track=track)
        planner_gp_mpc_frenet.trajectry_interpolation = 1

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
    
    # calc number of sim steps per one control step
    num_of_sim_steps = int(control_step / (env.timestep * 1000.0))

    # learn model from stored measurements

    with open('log_dataset', 'r') as f:
        data = json.load(f)

    if gp_mpc_type == 'cartesian':
        planner_gp_mpc.model.x_measurements[0] = data['X0']
        planner_gp_mpc.model.x_measurements[1] = data['X1']
        planner_gp_mpc.model.x_measurements[2] = data['X2']
        planner_gp_mpc.model.x_measurements[3] = data['X3']
        planner_gp_mpc.model.x_measurements[4] = data['X4']
        planner_gp_mpc.model.x_measurements[5] = data['X5']
        planner_gp_mpc.model.y_measurements[0] = data['Y0']
        planner_gp_mpc.model.y_measurements[1] = data['Y1']
        planner_gp_mpc.model.y_measurements[2] = data['Y2']
    elif gp_mpc_type == 'frenet':
        planner_gp_mpc_frenet.model.x_measurements[0] = data['X0']
        planner_gp_mpc_frenet.model.x_measurements[1] = data['X1']
        planner_gp_mpc_frenet.model.x_measurements[2] = data['X2']
        planner_gp_mpc_frenet.model.x_measurements[3] = data['X3']
        planner_gp_mpc_frenet.model.x_measurements[4] = data['X4']
        planner_gp_mpc_frenet.model.x_measurements[5] = data['X5']
        planner_gp_mpc_frenet.model.y_measurements[0] = data['Y0']
        planner_gp_mpc_frenet.model.y_measurements[1] = data['Y1']
        planner_gp_mpc_frenet.model.y_measurements[2] = data['Y2']

    print(len(planner_gp_mpc.model.x_measurements[0]))
    print("GP training...")
    train_x_scaled, train_y_scaled = planner_gp_mpc.model.init_gp()
    # learn model from stored measurements
    if not PRETRAINED:
        if gp_mpc_type == 'cartesian':
            print(f"{len(planner_gp_mpc.model.x_measurements[0])}")
            planner_gp_mpc.model.train_gp_min_variance(len(planner_gp_mpc.model.x_measurements[0]))

            if SAVE_MODEL:
                now = datetime.now()
                # dd/mm/YY H:M:S
                dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

                torch.save(planner_gp_mpc.model.gp_model.state_dict(), 'gp' + dt_string + '.pth')
                torch.save(planner_gp_mpc.model.gp_likelihood.state_dict(), 'gp_likelihood' + dt_string + '.pth')

        elif gp_mpc_type == 'frenet':
            print(f"{len(planner_gp_mpc_frenet.model.x_measurements[0])}")
            planner_gp_mpc_frenet.model.train_gp_min_variance(len(planner_gp_mpc_frenet.model.x_measurements[0]))

            if SAVE_MODEL:
                now = datetime.now()
                # dd/mm/YY H:M:S
                dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

                torch.save(planner_gp_mpc_frenet.model.gp_model.state_dict(), 'gp' + dt_string + '.pth')
                torch.save(planner_gp_mpc_frenet.model.gp_likelihood.state_dict(), 'gp_likelihood' + dt_string + '.pth')
    else:
        # If you have pretrained models, load them
        model_path = PRETRAINED_MODEL['model']

        if gp_mpc_type == 'cartesian':
            # Load model
            state_dict_gp1 = torch.load('trained_models/' + model_path[0] + '.pth').copy()
            planner_gp_mpc.model.gp_model.load_state_dict(state_dict_gp1)

            state_dict_likelihood1 = torch.load('trained_models/' + model_path[1] + '.pth').copy()
            planner_gp_mpc.model.gp_likelihood.load_state_dict(state_dict_likelihood1)
        elif gp_mpc_type == 'frenet':
            # Load model
            state_dict_gp1 = torch.load('trained_models/' + model_path[0] + '.pth').copy()
            planner_gp_mpc_frenet.model.gp_model.load_state_dict(state_dict_gp1)

            state_dict_likelihood1 = torch.load('trained_models/' + model_path[1] + '.pth').copy()
            planner_gp_mpc_frenet.model.gp_likelihood.load_state_dict(state_dict_likelihood1)

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
        if gp_mpc_type == 'frenet':
            pose_frenet = track.cartesian_to_frenet(np.array([vehicle_state[0], vehicle_state[1], vehicle_state[3]]))  # np.array([x,y,yaw])


        # print(env.sim.agents[0].state[3])

        mean, lower, upper = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        idx = 0
        u = [0.0, 0.0]
        tracking_error = 0.0
        total_var = 0.0
        # if not gp_model_trained:
        if gp_model_trained:
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

                # _, tracking_error, _, _, _ = nearest_point_on_trajectory(np.array([mpc_pred_x[0], mpc_pred_y[0]]),
                #                                                          np.array([mpc_ref_path_x[0:2], mpc_ref_path_y[0:2]]).T)
            else:
                print("ERROR")
                
            # draw predicted states and reference trajectory
            draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
            draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

            _, tracking_error, _, _, idx = nearest_point_on_trajectory(np.array([mpc_pred_x[0], mpc_pred_y[0]]),
                                                                     np.array([mpc_ref_path_x[0:2], mpc_ref_path_y[0:2]]).T)
        
        if gp_mpc_type == 'cartesian':
            if gp_model_trained:
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    if gp_mpc_type == 'cartesian':
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

        vy_error = abs(mean[1] - (env.sim.agents[0].state[10] - vehicle_state[4]))
        yaw_rate_error = abs(mean[2] - (env.sim.agents[0].state[5] - vehicle_state[5]))
        vx_error = abs(mean[0] - (env.sim.agents[0].state[3] - vehicle_state[2]))

        print('V: %5.5f  Vx: %5.5f  Vy: %+5.5f  %sVy_e: %5.5f\033[0m, %sYaw_rate_e: %5.5f\033[0m, %sVx_e: %5.5f\033[0m' % (
            waypoints[:, 5][idx],
            env.sim.agents[0].state[3],
            env.sim.agents[0].state[10],
            '\033[91m' if vy_error > 0.08 else '\033[32m',
            vy_error,
            '\033[91m' if yaw_rate_error > 0.08 else '\033[32m',
            yaw_rate_error,
            '\033[91m' if vx_error > 0.08 else '\033[32m',
            vx_error,
        ))

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

    if SAVE_MODEL:
        if gp_mpc_type == 'cartesian':
            save_GP_enemble_model(planner_gp_mpc.model)
        elif gp_mpc_type == 'frenet':
            save_GP_enemble_model(planner_gp_mpc_frenet.model)

if __name__ == '__main__':
    main()

# formula zero paper
# exp2
