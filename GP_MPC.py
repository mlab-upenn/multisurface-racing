import time
import yaml
import gym
from argparse import Namespace
from regulators.pure_pursuit import *
from regulators.path_follow_mpc import *
from models.configs import *
from models.extended_kinematic import ExtendedKinematicModel
from models.GP_model_ensembling import GPEnsembleModel
from models.GP_model_ensembling_frenet import GPEnsembleModelFrenet
from helpers.closest_point import *
from helpers.track import Track
import torch
import gpytorch
import os
import numpy as np
from datetime import datetime

from helpers.draw_debug import DrawDebug
from helpers.utils import save_GP_enemble_model
import copy
import json
import logging
from helpers.logging import create_logger

def main():  # after launching this you can run visualization.py to see the results
    """
    main entry point
    """
    main_logger = create_logger('main', logging.DEBUG)
    main_logger.info('Starting main')

    # Program parameters
    model_in_first_lap = 'ext_kinematic'  # options: ext_kinematic, pure_pursuit
    # currently only "custom_track" works for frenet
    map_name = 'custom_track'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape, BrandsHatch, DualLaneChange, custom_track
    use_dyn_friction = False
    gp_mpc_type = 'frenet'  # cartesian, frenet
    control_step = 100.0  # ms
    render_every = 30  # render graphics every n sim steps
    constant_speed = True
    constant_friction = 0.7
    number_of_laps = 20
    SAVE_MODEL = True


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

        centerline_descriptor = np.array([[0.0, 25 * np.pi, 25 * np.pi + 25, 25 * (3.0 * np.pi / 2.0) + 25, 25 * (3.0 * np.pi / 2.0) + 50,
                                           25 * (2.0 * np.pi + np.pi / 2.0) + 50, 25 * (2.0 * np.pi + np.pi / 2.0) + 125, 25 * (3.0 * np.pi) + 125,
                                           25 * (3.0 * np.pi) + 200],
                                          [0.0, 0.0, -25.0, -50.0, -50.0, -100.0, -100.0, -75.0, 0.0],
                                          [0.0, 50.0, 50.0, 75.0, 100.0, 100.0, 25.0, 0.0, 0.0],
                                          [1 / 25, 0.0, -1 / 25, 0.0, 1 / 25, 0.0, 1 / 25, 0.0, 1/25],
                                          [0.0, np.pi, np.pi, np.pi / 2.0, np.pi / 2.0, 3.0 * np.pi / 2.0, 3.0 * np.pi / 2.0, 0.0, 0.0]]).T

        main_logger.debug(centerline_descriptor)
        main_logger.debug(centerline_descriptor.shape)

        track = Track(centerline_descriptor=centerline_descriptor, track_width=10.0, reference_speed=5.0, log_level=0)
        waypoints = track.get_reference_trajectory()
    # waypoints[:, 3] += 1.5707963268

    # waypoints[:, 5] *= 0.82

    if constant_speed:
        waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * 4.5

    # init controllers
    planner_pp = PurePursuitPlanner(conf, 0.805975 + 1.50876)  # 0.805975 + 1.50876
    planner_pp.waypoints = waypoints

    planner_gp_mpc = None

    if gp_mpc_type == 'frenet':
        planner_gp_mpc = STMPCPlanner(model=GPEnsembleModelFrenet(config=MPCConfigGPFrenet(), track=track), waypoints=waypoints,
                                             config=MPCConfigGPFrenet(), track=track)
        planner_gp_mpc.trajectry_interpolation = 1
    elif gp_mpc_type == 'cartesian':
        planner_gp_mpc = STMPCPlanner(model=GPEnsembleModel(config=MPCConfigGP()), waypoints=waypoints,
                                  config=MPCConfigGP())
    else:
        raise ValueError('Unknown gp_mpc_type')


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

    main_logger.info('Model used: %s' % model_in_first_lap)

    original_vel_profile = copy.deepcopy(waypoints[:, 5])
    
    xcl = np.empty((0,7))
    ucl = np.empty((0,2))
    cov_cl = np.empty((0,3))

    laps_done = 0

    laps_before_LMPC = 5
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
        
        # add x0 to closed loop trajectory
        xcl = np.vstack((xcl, vehicle_state))

        mean, lower, upper = [np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0]])]
        u = [0.0, 0.0]
        tracking_error = 0.0
        total_var = 0.0

        if gp_model_trained <= 1:
            main_logger.debug("Initial model")
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
            # main_logger.debug("GP model")
            if gp_mpc_type == 'cartesian':

                print(f"X: {vehicle_state[0]}  Y: {vehicle_state[1]}  YAW: {vehicle_state[3]}")

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
            
                print(f"X: {vehicle_state[0]}  Y: {vehicle_state[1]}  S: {pose_frenet[0]}")

                vehicle_state_frenet = np.array([pose_frenet[0],  # s
                                                 pose_frenet[1],  # ey
                                                 env.sim.agents[0].state[3],  # vx
                                                 pose_frenet[2],  # eyaw
                                                 env.sim.agents[0].state[10],  # vy
                                                 env.sim.agents[0].state[5],  # yaw rate
                                                 env.sim.agents[0].state[2],  # steering angle
                                                 ])

                u, mpc_ref_path_s, mpc_ref_path_ey, mpc_pred_s, mpc_pred_ey, mpc_os, mpc_oey = planner_gp_mpc.plan(
                    vehicle_state_frenet)

                u[0] = u[0] / planner_gp_mpc.config.MASS  # Force to acceleration

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

                _, tracking_error, _, _, _ = nearest_point_on_trajectory(np.array([mpc_pred_x[0], mpc_pred_y[0]]),
                                                                         np.array([mpc_ref_path_x[0:2], mpc_ref_path_y[0:2]]).T)
            else:
                main_logger.error("GP MPC type not supported!")
        # u[0] += np.random.randn(1)[0] * 0.00001
        # u[1] += np.random.randn(1)[0] * 0.0001

        if gp_model_trained:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mean, lower, upper, covariance = planner_gp_mpc.model.scale_and_predict_model_step(vehicle_state, [u[0] * planner_gp_mpc.config.MASS, u[1]])
                cov_cl = np.vstack((cov_cl, covariance))
        else:
            covariance = None


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
            log['vx_mean'].append(float(mean[0,0]))
            log['vx_var'].append(float(abs(mean[0,0] - lower[0,0])))
            log['vy_mean'].append(float(mean[0,1]))
            log['vy_var'].append(float(abs(mean[0,1] - lower[0,1])))
            log['theta_mean'].append(float(mean[0,2]))
            log['theta_var'].append(float(abs(mean[0,2] - lower[0,2])))
            log['true_vx'].append(env.sim.agents[0].state[3] - vehicle_state[2])
            log['true_vy'].append(env.sim.agents[0].state[10] - vehicle_state[4])
            log['true_yaw_rate'].append(env.sim.agents[0].state[5] - vehicle_state[5])
            log['tracking_error'].append(tracking_error)
            logged_data = 0

        # learning GPs
        u[0] = u[0] * planner_gp_mpc.config.MASS  # Acceleration to force

        # Add control input to ucl
        ucl = np.vstack((ucl, u))        

        gather_data_every = 2

        vx_transition = env.sim.agents[0].state[3] + np.random.randn(1)[0] * 0.00001 - vehicle_state[2]
        vy_transition = env.sim.agents[0].state[10] + np.random.randn(1)[0] * 0.00001 - vehicle_state[4]
        yaw_rate_transition = env.sim.agents[0].state[5] + np.random.randn(1)[0] * 0.00001 - vehicle_state[5]

        gather_data += 1
        if gather_data >= gather_data_every:
            Y_sample = np.array([float(vx_transition), float(vy_transition), float(yaw_rate_transition)])
            X_sample = np.array([float(vehicle_state[2]), float(vehicle_state[4]),
                                 float(vehicle_state[5]), float(vehicle_state[6]), float(u[0]), float(u[1])])

            planner_gp_mpc.model.add_new_datapoint(X_sample, Y_sample)
            gather_data = 0

        if obs['lap_counts'][0] - 1 == gp_model_trained:
            # if waypoints[:, 5][0] >= last_speed + 0.25 and False:  # or (waypoints[:, 5][0] > 7.5 and waypoints[:, 5][0] >= last_speed + 0.05) :
            last_speed = waypoints[:, 5][0]

            gp_model_trained += 1
            main_logger.debug("GP training...")
            num_of_new_samples = 250

            print(f"{len(planner_gp_mpc.model.x_measurements[0])}")
            planner_gp_mpc.model.train_gp_min_variance(num_of_new_samples)

            main_logger.debug("GP training done")
            main_logger.debug('Model used: GP')
            main_logger.debug('Reference speed: %f' % waypoints[:, 5][0])

            if gp_mpc_type == 'cartesian':
                log_dataset['X0'] = planner_gp_mpc.model.x_samples[0]
                log_dataset['X1'] = planner_gp_mpc.model.x_samples[1]
                log_dataset['X2'] = planner_gp_mpc.model.x_samples[2]
                log_dataset['X3'] = planner_gp_mpc.model.x_samples[3]
                log_dataset['X4'] = planner_gp_mpc.model.x_samples[4]
                log_dataset['X5'] = planner_gp_mpc.model.x_samples[5]
                log_dataset['Y0'] = planner_gp_mpc.model.y_samples[0]
                log_dataset['Y1'] = planner_gp_mpc.model.y_samples[1]
                log_dataset['Y2'] = planner_gp_mpc.model.y_samples[2]

            # Recompute safe set covariance
            planner_gp_mpc.recompute_covariance(planner_gp_mpc.model.get_covariance)

            with open('log01', 'w') as f:
                json.dump(log, f)
            with open('testing_dataset', 'w') as f:
                json.dump(log_dataset, f)

        if obs['lap_counts'][0] - 1 == laps_done:
            if gp_model_trained:
                planner_gp_mpc.add_safe_trajectory(xcl, ucl, cov_cl, planner_gp_mpc.model.get_covariance)
            else:
                planner_gp_mpc.add_safe_trajectory(xcl, ucl, cov_cl)
            xcl = np.empty((0,7))
            ucl = np.empty((0,2))
            cov_cl = np.empty((0,3))
            laps_done += 1
            main_logger.debug('%d Laps Done' % laps_done)

        if obs['lap_counts'][0] == laps_before_LMPC:
            planner_gp_mpc.config.LMPC_FLAG = True

            # Re-initialize the MPC problem with the new LMPC flag
            planner_gp_mpc.mpc_prob_init()
            
        if obs['lap_counts'][0] == number_of_laps:
            done = 1

    main_logger.info('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
    with open('log01', 'w') as f:
        json.dump(log, f)
    with open('log_dataset', 'w') as f:
        json.dump(log_dataset, f)

    if SAVE_MODEL:
        save_GP_enemble_model(planner_gp_mpc.model)


if __name__ == '__main__':
    main()