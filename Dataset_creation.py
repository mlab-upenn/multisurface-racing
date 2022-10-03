import yaml
import gym
from argparse import Namespace
import numpy as np
import json


def main():  # after launching this you can run visualization.py to see the results
    """
    main entry point
    """

    # Choose program parameters
    model_in_first_lap = 'ext_kinematic'  # options: ext_kinematic, pure_pursuit
    map_name = 'rounded_rectangle'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape
    use_dyn_friction = False
    render_every = 1  # render graphics every n control steps

    # Creating the single-track Motion planner and Controller

    # Load map config file
    with open('config_%s.yaml' % map_name) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # MB - reference point: center of mass
    # dynamic_ST - reference point: center of mass

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                   num_agents=1, timestep=0.001, model='MB', drive_control_mode='acc',
                   steering_control_mode='vel')

    # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    obs, step_reward, done, info = env.reset(
        np.array([[0.0, 0.0, 0.0, 0.8, 10.0, 0.0, 0.0]]))
    env.render()

    log_dataset = {'X0': [], 'X1': [], 'X2': [], 'X3': [], 'X4': [], 'X5': [], 'Y0': [], 'Y1': [], 'Y2': [],
                   'X0[t-1]': [], 'X1[t-1]': [], 'X2[t-1]': [], 'X3[t-1]': [], 'X4[t-1]': [], 'X5[t-1]': [], 'Y0[t-1]': [], 'Y1[t-1]': [],
                   'Y2[t-1]': []}

    num_of_sim_steps = 50
    laptime = 0.0
    last_render = 0
    gather_data_every = 2  # must be > 1

    env.params['tire_p_dy1'] = 1.2  # mu_y
    env.params['tire_p_dx1'] = 1.3  # mu_x

    # u0_arr = np.array([-6000.0, -5000.0, -4000.0, 4000.0, 5000.0, 6000.0])
    u0_arr = np.array([-10000.0, 10000.0])
    u1_arr = np.array([-3.2, -2.2, -1.0, 0.0, 1.0, 2.2, 3.2])

    steering_angle_arr = np.array([0.0])
    velocity_arr = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.5, 12.8])
    yaw_rate_arr = np.array([-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0])
    beta_arr = np.array([0.0])

    np.random.seed(42)  # Not so random - solves everything

    for iter in range(800):
        if (iter + 1) % 100 == 0:
            print('Iteration: %d' % iter)

        u0_choice = np.random.choice(u0_arr)
        u1_choice = np.random.choice(u1_arr)

        steering_angle_choice = np.random.choice(steering_angle_arr)
        velocity_choice = np.random.choice(velocity_arr)
        yaw_rate_choice = np.random.choice(yaw_rate_arr)
        beta_choice = np.random.choice(beta_arr)

        u = [u0_choice, u1_choice]
        # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
        obs, step_reward, done, info = env.reset(
            np.array([[0.0, 0.0, 0.0, steering_angle_choice, velocity_choice, yaw_rate_choice, beta_choice]]))
        env.render()
        gather_data = 0

        for i in range(2):

            vehicle_state = np.array([env.sim.agents[0].state[0],
                                      env.sim.agents[0].state[1],
                                      env.sim.agents[0].state[3],  # vx
                                      env.sim.agents[0].state[4],  # yaw angle
                                      env.sim.agents[0].state[10],  # vy
                                      env.sim.agents[0].state[5],  # yaw rate
                                      env.sim.agents[0].state[2],  # steering angle
                                      ]) + np.random.randn(7) * 0.00001

            # Simulation step
            step_reward = 0.0
            for i in range(num_of_sim_steps):
                obs, rew, _, info = env.step(np.array([[u[1], u[0]]]))
                step_reward += rew
            laptime += step_reward

            # Rendering
            last_render += 1
            if last_render >= render_every:
                last_render = 0
                env.render(mode='human_fast')

            vx_transition = env.sim.agents[0].state[3] + np.random.randn(1)[0] * 0.00001 - vehicle_state[2]
            vy_transition = env.sim.agents[0].state[10] + np.random.randn(1)[0] * 0.00001 - vehicle_state[4]
            yaw_rate_transition = env.sim.agents[0].state[5] + np.random.randn(1)[0] * 0.00001 - vehicle_state[5]

            gather_data += 1
            if gather_data >= gather_data_every:
                gather_data = 0
                log_dataset['X0'].append(float(vehicle_state[2]))
                log_dataset['X1'].append(float(vehicle_state[4]))
                log_dataset['X2'].append(float(vehicle_state[5]))
                log_dataset['X3'].append(float(vehicle_state[6]))
                log_dataset['X4'].append(float(u[0]))
                log_dataset['X5'].append(float(u[1]))
                log_dataset['Y0'].append(float(vx_transition))
                log_dataset['Y1'].append(float(vy_transition))
                log_dataset['Y2'].append(float(yaw_rate_transition))
                log_dataset['X0[t-1]'].append(X_t_1[0])
                log_dataset['X1[t-1]'].append(X_t_1[1])
                log_dataset['X2[t-1]'].append(X_t_1[2])
                log_dataset['X3[t-1]'].append(X_t_1[3])
                log_dataset['X4[t-1]'].append(X_t_1[4])
                log_dataset['X5[t-1]'].append(X_t_1[5])
                log_dataset['Y0[t-1]'].append(X_t_1[6])
                log_dataset['Y1[t-1]'].append(X_t_1[7])
                log_dataset['Y2[t-1]'].append(X_t_1[8])

            X_t_1 = [float(vehicle_state[2]), float(vehicle_state[4]), float(vehicle_state[5]), float(vehicle_state[6]), float(u[0]), float(u[1]),
                     float(vx_transition), float(vy_transition), float(yaw_rate_transition)]

    print('Data saved...')
    with open('dataset_1_3_50ms_ex4', 'w') as f:
        json.dump(log_dataset, f)


if __name__ == '__main__':
    main()

# formula zero paper
# exp2
