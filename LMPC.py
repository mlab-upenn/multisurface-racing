import time
import yaml
import gym
from argparse import Namespace
from regulators.pure_pursuit import *
from helpers.closest_point import *
import numpy as np
from pyglet.gl import GL_POINTS
import json

# import sys
# sys.path.append('fnc')
import matplotlib.pyplot as plt
from fnc.plot import plotTrajectory, plotClosedLoopLMPC, plot_predicted_trajectory, animation_states, saveGif_xyResults
from initControllerParameters import initMPCParams, initLMPCParams
from fnc.PredictiveControllers import MPC, LMPC, MPCParams
from fnc.PredictiveModel import PredictiveModel
from fnc.Utilities import Regression, PID
from fnc.Track import Map
import pickle
import pdb
import sys, os
# sys.path.append(sys.path[0]+'/../../utils/')

import timeit


@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    dist_from_segment_start = np.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])
    return projections[min_dist_segment], dist_from_segment_start, dists[min_dist_segment], t[
        min_dist_segment], min_dist_segment


def determine_side(a, b, p):
    ''' this function determine, if car is on right side of trajectory or on left side
    Arguments:
         a - point of trajectory, which is nearest to the car
         b - next trajectory point
         p - actual position of car
    Returns:
         -1 if car is on left side of trajectory
         1 if car is on right side of trajectory
         0 if car is on trajectory
    '''
    side = (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    if side > 0:
        return 1.0
    elif side < 0:
        return -1.0
    else:
        return 0.0


def cartesian_to_curv(position, orientation, waypoints):
    waypoints_distances = np.linalg.norm(waypoints[1:, (1, 2)] - waypoints[:-1, (1, 2)], axis=1)

    # Find nearest index/setpoint from where the trajectories are calculated
    _, dist_to_segment_start, dist_to_traj, t, ind = nearest_point(np.array([position[0], position[1]]),
                                                                   waypoints[:, (1, 2)])

    orientation_diff = (waypoints[np.mod(ind + 1, waypoints.shape[0] - 1)][3] - waypoints[ind][3])

    orientation_ref = waypoints[ind][3] + (t * orientation_diff)
    orientation_ref = (orientation_ref + np.pi) % (2 * np.pi) - np.pi

    s = waypoints[ind][0] + dist_to_segment_start
    side = determine_side([waypoints[ind][1], waypoints[ind][2]],
                          [waypoints[np.mod(ind + 1, waypoints.shape[0] - 1)][1],
                           waypoints[np.mod(ind + 1, waypoints.shape[0] - 1)][2]],
                          position,
                          )
    e_y = dist_to_traj * side
    e_psi = orientation - orientation_ref
    e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi

    return e_psi, s, e_y


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


def main():
    """
    main entry point
    """

    # ================================================================================================================
    # ========================================== Initialize parameters  ==============================================
    # ================================================================================================================
    N = 14  # Horizon length
    n = 6
    d = 2  # State and Input dimension
    x0 = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # Initial condition [vx, vy, yaw_rate, ...]

    vt = 6.5  # target vevlocity

    # Initialize controller parameters
    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(3.0, N)

    # load friction map
    tpamap_name = './maps/rounded_rectangle/rounded_rectangle_tpamap.csv'
    tpadata_name = './maps/rounded_rectangle/rounded_rectangle_tpadata.json'

    tpamap = np.loadtxt(tpamap_name, delimiter=';', skiprows=1)
    tpamap *= 1.5  # map is 1.5 times larger than normal

    tpadata = {}
    with open(tpadata_name) as f:
        tpadata = json.load(f)

    # Init regulator
    work = {'mass': 1225.88, 'lf': 0.80597534362552312, 'tlad': 2.6461887897713965, 'vgain': 0.950338203837889}

    # Load config file
    with open('config_rounded_rectangle.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    raceline = np.loadtxt('./maps/rounded_rectangle/rounded_rectangle_waypoints.csv', delimiter=";", skiprows=3)
    waypoints = np.array(raceline)

    planner_pp = PurePursuitPlanner(conf, 0.17145 + 0.15875)

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
        # draw.draw_debug(e)

    # MB - reference point: center of mass
    # ST - reference point: center of mass
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                   num_agents=1, timestep=0.001, model='MB', drive_control_mode='acc',
                   steering_control_mode='angle')

    env.add_render_callback(render_callback)
    # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    obs, step_reward, done, info = env.reset(
        np.array([[conf.sx * 10, conf.sy * 10, conf.stheta, 0.0, x0[0], 0.0, 0.0]]))
    env.render()

    laptime = 0.0
    start = time.time()
    render_every = 2
    last_render = 0

    log = {}
    log['time'] = []

    control_step = 100.0  # ms
    num_of_sim_steps = int(control_step / (env.timestep * 1000.0))

    acceleration = 0.0
    steer_angle = 0.0

    print("Starting Pure Pursuit...")
    xcl_pid = [x0]
    xcl_pid_glob = [x0]
    ucl_pid = []

    lmpc_init = 0

    num_of_init_laps = 1

    num_of_samples = 1000
    current_num_of_samples = 0

    while not done:
        lap_num = obs['lap_counts'][0]
        # LMPC before sim step
        if current_num_of_samples < num_of_samples:
            # Regulator step pure pursuit
            speed, steer_angle = planner_pp.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                                                 work['tlad'],
                                                 work['vgain'])

            acceleration = (vt - env.sim.agents[0].state[3]) * 0.1
            ut = [steer_angle, acceleration]
            ucl_pid.append(ut)
        else:
            if not lmpc_init:
                print("Starting LMPC")
                # Initialize Predictive Model for lmpc
                predictiveModel = PredictiveModel(n, d, waypoints, 4)
                for i in range(0, 4):  # add trajectories used for model learning
                    predictiveModel.addTrajectory(np.array(xcl_pid)[:-1, :], np.array(ucl_pid))

                # Initialize Controller
                lmpcParameters.timeVarying = True
                lmpc = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, predictiveModel)
                for i in range(0, numSS_it):  # add trajectories for safe set
                    lmpc.addTrajectory(np.array(xcl_pid)[:-1, :], np.array(ucl_pid), np.array(xcl_pid_glob)[:-1, :])
                lmpc_init = 1
            break
        # print(env.sim.agents[0].state[3])

        # set correct friction to the environment
        min_id = get_closest_point_vectorized(np.array([obs['poses_x'][0], obs['poses_y'][0]]), np.array(tpamap))
        env.params['tire_p_dy1'] = 1.0  # tpadata[str(min_id)][0]  # mu_y
        env.params['tire_p_dx1'] = 1.1  # tpadata[str(min_id)][0] * 1.1  # mu_x

        # Simulation step
        step_reward = 0.0
        for i in range(num_of_sim_steps):
            obs, rew, _, info = env.step(np.array([[steer_angle, acceleration]]))
            step_reward += rew
        laptime += step_reward

        state_global = [env.sim.agents[0].state[3],  # vx
                        env.sim.agents[0].state[10],  # vy
                        env.sim.agents[0].state[5],  # yaw rate
                        env.sim.agents[0].state[4],  # yaw angle
                        env.sim.agents[0].state[0],  # x
                        env.sim.agents[0].state[1],  # y
                        ] + np.random.randn(6) * 0.1

        e_psi, s, e_y = cartesian_to_curv([state_global[4], state_global[5]], state_global[3], waypoints)

        state = [state_global[0],  # vx
                 state_global[1],  # vy
                 state_global[2],  # yaw rate
                 e_psi,
                 s,
                 e_y,
                 ]

        # Rendering
        last_render += 1
        if last_render == render_every:
            last_render = 0
        env.render(mode='human_fast')

        # Logging
        log['time'].append(laptime)

        if obs['lap_counts'][0] == 40:
            done = 1

        # LMPC after sim step
        if current_num_of_samples < num_of_samples:
            xcl_pid.append(state)
            xcl_pid_glob.append(state_global)
            current_num_of_samples += 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
    with open('log01', 'w') as f:
        json.dump(log, f)


if __name__ == '__main__':
    main()

# formula zero paper
# exp2
