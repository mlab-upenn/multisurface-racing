import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import json

import numpy as np

colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

# input parameters
name = 'log01_eval'
# name = './ignore_stored_data/all_data_bad_comparison'

from_lap = 0
to_lap = 20

# visualization
with open('%s' % name, 'r') as f:
    data = json.load(f)

print('Number of datapoints: %s' % len(data['time']))

keys = list(data.keys())
for key in keys:
    data[key] = np.array(data[key])

data_lap = {}
keys = list(data.keys())
for key in keys:
    data_lap[key] = np.array(data[key][np.bitwise_and(from_lap <= data['lap_n'], data['lap_n'] <= to_lap)])

data_lap['w'] = data_lap['w'].squeeze().T


plt.plot(data_lap['time'], data_lap['w'][0], label='0.5')
plt.plot(data_lap['time'], data_lap['w'][1], label='0.6')
plt.plot(data_lap['time'], data_lap['w'][2], label='0.9')
plt.plot(data_lap['time'], data_lap['w'][3], label='1.1')
# plt.plot(data_lap['time'], data_lap['w'][4], label='1.1')
plt.legend()
plt.show()


#  np.array(data['time'])[np.array(data['lap_n']) == 1.0]
plt.plot(data_lap['time'], data_lap['vx'])
plt.plot(data_lap['time'], data_lap['v_ref'])
plt.ylabel('speed [m/s]')
plt.xlabel('time [s]')
plt.show()


plt.plot(data_lap['time'], data_lap['tracking_error'])
plt.ylabel('tracking error')
plt.xlabel('time')
# ax.set_zlabel('vx [m/s]')
# ax.set_aspect('equalxy', adjustable='datalim')
plt.show()


# np.array(data['time'])[np.array(data['lap_n']) == 1.0]
ax = plt.figure().add_subplot()
color_arr = []
for i in range(len(data_lap['x'])):
    # color_arr.append(data_lap['true_mu'][i])
    color_arr.append(data_lap['w2'][i])

# scat = ax.scatter(data_lap['x'], data_lap['y'], cmap='plasma', marker='o', c=color_arr)  # , cmap='coolwarm', vmin=0, vmax=1.0
scat = ax.scatter(data_lap['x'], data_lap['y'], cmap='coolwarm', marker='o', c=color_arr, vmin=0, vmax=1.0)  # , cmap='coolwarm', vmin=0, vmax=1.0
plt.colorbar(scat)
plt.ylabel('y [m]')
plt.xlabel('x [m]')
plt.show()

ax = plt.figure().add_subplot()
color_arr = []
for i in range(len(data_lap['x'])):
    color_arr.append(data_lap['true_mu'][i])
    # color_arr.append(data_lap['w2'][i])

# scat = ax.scatter(data_lap['x'], data_lap['y'], cmap='plasma', marker='o', c=color_arr)  # , cmap='coolwarm', vmin=0, vmax=1.0
scat = ax.scatter(data_lap['x'], data_lap['y'], cmap='coolwarm', marker='o', c=color_arr, vmin=0, vmax=1.0)  # , cmap='coolwarm', vmin=0, vmax=1.0
plt.colorbar(scat)
plt.ylabel('y [m]')
plt.xlabel('x [m]')
plt.show()

# exit()

plt.plot(data_lap['x'], data_lap['y'])
plt.plot(data_lap['x_gt'], data_lap['y_gt'])
plt.ylabel('y [m]')
plt.xlabel('x [m]')
plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(data_lap['x'], data_lap['y'], data_lap['vx_var'], marker='o')
plt.ylabel('vx var')
plt.xlabel('time')
ax.set_zlabel('ax variance [m/s/s]')
ax.set_aspect('equalxy', adjustable='datalim')
plt.show()

# ax = plt.figure().add_subplot(projection='3d')
plt.scatter(data_lap['time'], data_lap['vx_var'], marker='o')
plt.ylabel('vx var')
plt.xlabel('time')
# plt.set_zlabel('ax variance [m/s/s]')
# plt.set_aspect('equalxy', adjustable='datalim')
plt.show()

# # ax = plt.figure().add_subplot(projection='3d')
# plt.scatter(data_lap['time'], data_lap['wsf_s'], marker='o')
# plt.ylabel('slip')
# plt.xlabel('time')
# # ax.set_zlabel('slip ratio')
# # ax.set_aspect('equalxy', adjustable='datalim')
# plt.show()
#
# plt.scatter(data_lap['time'], data_lap['wsf'], marker='o')
# plt.scatter(data_lap['time'], data_lap['wsf2'], marker='o')
# plt.scatter(data_lap['time'], data_lap['wsf3'], marker='o')
# plt.scatter(data_lap['time'], data_lap['wsf4'], marker='o')
# plt.ylabel('wheel speeds [rad/s]')
# plt.xlabel('time')
# # ax.set_zlabel('slip ratio')
# # ax.set_aspect('equalxy', adjustable='datalim')
# plt.show()

# ax = plt.figure().add_subplot(projection='3d')
plt.scatter(data_lap['time'], data_lap['vx'], marker='o')
plt.ylabel('vx')
plt.xlabel('time')
# ax.set_zlabel('vx [m/s]')
# ax.set_aspect('equalxy', adjustable='datalim')
plt.show()



#  np.array(data['time'])[np.array(data['lap_n']) == 1.0]
# ax = plt.figure().add_subplot(projection='3d')
plt.scatter(data_lap['time'], data_lap['vy_var'], marker='o')
plt.ylabel('vy var')
plt.xlabel('time')
# ax.set_zlabel('ay variance [m/s/s]')
# ax.set_aspect('equalxy', adjustable='datalim')
plt.show()

# exit()

#  np.array(data['time'])[np.array(data['lap_n']) == 1.0]
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(data_lap['x'], data_lap['y'], data_lap['theta_var'], marker='o')
plt.ylabel('y [m]')
plt.xlabel('x [m]')
ax.set_zlabel('yaw rate change variance [rad/s/s]')
ax.set_aspect('equalxy', adjustable='datalim')
plt.show()

#  np.array(data['time'])[np.array(data['lap_n']) == 1.0]
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(data_lap['x'], data_lap['y'], data_lap['tracking_error'], marker='o')
plt.ylabel('y [m]')
plt.xlabel('x [m]')
ax.set_zlabel('tracking error [m]')
ax.set_aspect('equalxy', adjustable='datalim')
plt.show()

plt.plot(data_lap['time'], data_lap['vx_mean'], label='prediction')
plt.plot(data_lap['time'], data_lap['true_vx'], label='ground truth')
plt.legend(['prediction', 'ground truth'])
plt.fill_between(data_lap['time'], data_lap['vx_mean'] - data_lap['vx_var'], data_lap['vx_mean'] + data_lap['vx_var'], alpha=0.5)
plt.ylabel('ax')
plt.xlabel('time [s]')
plt.show()

plt.plot(data_lap['time'], data_lap['vy_mean'], label='prediction')
plt.plot(data_lap['time'], data_lap['true_vy'], label='ground truth')
plt.legend(['prediction', 'ground truth'])
plt.fill_between(data_lap['time'], data_lap['vy_mean'] - data_lap['vy_var'], data_lap['vy_mean'] + data_lap['vy_var'], alpha=0.5)
plt.ylabel('ay')
plt.xlabel('time [s]')
plt.show()

plt.plot(data_lap['time'], data_lap['theta_mean'], label='prediction')
plt.plot(data_lap['time'], data_lap['true_yaw_rate'], label='ground truth')
plt.legend(['prediction', 'ground truth'])
plt.fill_between(data_lap['time'], data_lap['theta_mean'] - data_lap['theta_var'], data_lap['theta_mean'] + data_lap['theta_var'], alpha=0.5)
plt.ylabel('yaw rate')
plt.xlabel('time [s]')
plt.show()

plt.plot(data_lap['time'], data_lap['wsf_mean'], label='prediction')
plt.plot(data_lap['time'], data_lap['true_wsf'], label='ground truth')
plt.legend(['prediction', 'ground truth'])
# plt.fill_between(data_lap['time'], data_lap['theta_mean'] - data_lap['theta_var'], data_lap['theta_mean'] + data_lap['theta_var'], alpha=0.5)
plt.ylabel('wsf')
plt.xlabel('time [s]')
plt.show()

plt.plot(data_lap['time'], data_lap['wsr_mean'], label='prediction')
plt.plot(data_lap['time'], data_lap['true_wsr'], label='ground truth')
plt.legend(['prediction', 'ground truth'])
# plt.fill_between(data_lap['time'], data_lap['theta_mean'] - data_lap['theta_var'], data_lap['theta_mean'] + data_lap['theta_var'], alpha=0.5)
plt.ylabel('wsr')
plt.xlabel('time [s]')
plt.show()

plt.plot(data_lap['time'], data_lap['vx_var'])
plt.ylabel('ax var')
plt.xlabel('time [s]')
plt.show()

plt.plot(data_lap['time'], data_lap['vy_var'])
plt.ylabel('ay var')
plt.xlabel('time [s]')
plt.show()

plt.plot(data_lap['time'], data_lap['theta_var'])
plt.ylabel('yaw rate var')
plt.xlabel('time [s]')
plt.show()
