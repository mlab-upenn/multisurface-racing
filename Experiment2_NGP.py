import json
import numpy as np
from models.GP_model_ensembling import GPEnsembleModel
from models.GP_model_ensembling_NGPs import GPEnsembleModelsNGPs
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
import gpytorch


@dataclass
class Config:
    DTK: float = 0.05  # time step [s] kinematic
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


# inputs
datasets_name = ['dataset_0_3_20ms_final',
                 'dataset_1_0_20ms_final',
                 ]  # dataset names
test_percentage = 30.0  # what percentage of the dataset should be used for testing

np.random.seed(42)

# code
data_names = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'Y0', 'Y1', 'Y2',  # x(t) and y(t)
              'X0[t-1]', 'X1[t-1]', 'X2[t-1]', 'X3[t-1]', 'X4[t-1]', 'X5[t-1]', 'Y0[t-1]', 'Y1[t-1]', 'Y2[t-1]',  # x(t-1) and y(t-1)
              ]

datasets = {}

for name in datasets_name:
    with open(name, 'r') as f:
        # load data
        data = json.load(f)
        # from dict to array to np.array
        datasets[name] = []
        for data_name in data_names:
            datasets[name].append(data[data_name])
        datasets[name] = np.array(datasets[name])
        # shuffle data inside each dataset
        datasets[name] = datasets[name].T
        np.random.shuffle(datasets[name])
        datasets[name] = datasets[name].T

# split into training and testing dataset
datasets_train = {}
datasets_test = {}
for name in datasets_name:
    num_of_data_in_test = int(datasets[name].shape[1] / 100.0 * test_percentage)
    datasets_test[name] = datasets[name][:, :num_of_data_in_test]
    datasets_train[name] = datasets[name][:, num_of_data_in_test:]

print('Dateset info start')
for name in datasets_name:
    print('-----------')
    print('Dataset name: %s' % name)
    print('Training data size: %f' % datasets_train[datasets_name[0]].shape[1])
    print('Testing data size: %f' % datasets_test[datasets_name[0]].shape[1])
    print('Min speed: %f' % np.min(datasets[name][0, :]))
    print('Max speed: %f' % np.max(datasets[name][0, :]))
print('Dateset info end')
print('\n\n')

# Method 1 - Training one GP on all the data - should have the worst results
arr_data_train_1 = []
arr_data_test_1 = []
for name in datasets_name:
    arr_data_train_1.append(datasets_train[name])
    arr_data_test_1.append(datasets_test[name])

data_method_1_train = np.concatenate(arr_data_train_1, axis=1, dtype='float32')
data_method_1_test = np.concatenate(arr_data_test_1, axis=1, dtype='float32')

model_exp_1 = GPEnsembleModel(Config())

model_exp_1.x_measurements[0] = np.concatenate([data_method_1_train[0], data_method_1_train[9]], axis=0, dtype='float32')
model_exp_1.x_measurements[1] = np.concatenate([data_method_1_train[1], data_method_1_train[10]], axis=0, dtype='float32')
model_exp_1.x_measurements[2] = np.concatenate([data_method_1_train[2], data_method_1_train[11]], axis=0, dtype='float32')
model_exp_1.x_measurements[3] = np.concatenate([data_method_1_train[3], data_method_1_train[12]], axis=0, dtype='float32')
model_exp_1.x_measurements[4] = np.concatenate([data_method_1_train[4], data_method_1_train[13]], axis=0, dtype='float32')
model_exp_1.x_measurements[5] = np.concatenate([data_method_1_train[5], data_method_1_train[14]], axis=0, dtype='float32')

model_exp_1.y_measurements[0] = np.concatenate([data_method_1_train[6], data_method_1_train[15]], axis=0, dtype='float32')
model_exp_1.y_measurements[1] = np.concatenate([data_method_1_train[7], data_method_1_train[16]], axis=0, dtype='float32')
model_exp_1.y_measurements[2] = np.concatenate([data_method_1_train[8], data_method_1_train[17]], axis=0, dtype='float32')



print('Method 1, training:')
scaled_x1, scaled_y1 = model_exp_1.init_gp()
model_exp_1.train_gp(scaled_x1, scaled_y1, method=0)
means_exp_1 = []
uppers_exp_1 = []
lowers_exp_1 = []

print('Method 1, testing:')
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for i in range(data_method_1_test.shape[1]):
        mean, lower, upper = model_exp_1.scale_and_predict_model_step([0.0, 0.0, data_method_1_test[0][i], 0.0, data_method_1_test[1][i],
                                                                       data_method_1_test[2][i], data_method_1_test[3][i]],
                                                                      [data_method_1_test[4][i], data_method_1_test[5][i]])
        if (i + 1) % 200 == 0:
            print('Method 1, prediction %s/%s' % (i, data_method_1_test.shape[1]))
        means_exp_1.append(mean)
        uppers_exp_1.append(upper)
        lowers_exp_1.append(lower)

means_exp_1 = np.array(means_exp_1).squeeze()  # n x 3
uppers_exp_1 = np.array(uppers_exp_1).squeeze()  # n x 3
lowers_exp_1 = np.array(lowers_exp_1).squeeze()  # n x 3

errors_exp_1 = np.absolute(means_exp_1.T - np.array([data_method_1_test[6], data_method_1_test[7], data_method_1_test[8]]))

print('--------Method 1 results--------')
print('ax prediction average error: %f' % (np.sum(errors_exp_1[0, :]) / errors_exp_1[0, :].size))
print('ax prediction RMS error: %f' % (np.sqrt(np.sum(np.square(errors_exp_1[0, :]))/errors_exp_1[0, :].size)))
print('ax prediction maximum error: %f' % np.max(errors_exp_1[0, :]))
print('ay prediction average error: %f' % (np.sum(errors_exp_1[1, :]) / errors_exp_1[1, :].size))
print('ay prediction RMS error: %f' % (np.sqrt(np.sum(np.square(errors_exp_1[1, :]))/errors_exp_1[1, :].size)))
print('ay prediction maximum error: %f' % np.max(errors_exp_1[1, :]))
print('yaw rate prediction average error: %f' % (np.sum(errors_exp_1[2, :]) / errors_exp_1[2, :].size))
print('yaw rate prediction RMS error: %f' % (np.sqrt(np.sum(np.square(errors_exp_1[2, :]))/errors_exp_1[2, :].size)))
print('yaw rate prediction maximum error: %f' % np.max(errors_exp_1[2, :]))

print('\n\n')

# Method 2
arr_data_train_2 = []
for name in datasets_name:
    arr_data_train_2.append(np.array(datasets_train[name], dtype='float32'))
# data_method_2_train_m1 = np.array(datasets_train[datasets_name[0]], dtype='float32')
# data_method_2_train_m2 = np.array(datasets_train[datasets_name[1]], dtype='float32')

# arr_data_test_2__ = [datasets_test[datasets_name[0]], datasets_test[datasets_name[1]]]
# data_method_2_test__ = np.concatenate(arr_data_test_2__, axis=1, dtype='float32')

arr_data_test_2 = []
for name in datasets_name:
    arr_data_test_2.append([datasets_test[name]])
data_method_2_test = np.concatenate(arr_data_test_2, axis=2, dtype='float32').squeeze()

model_exp_2 = GPEnsembleModelsNGPs(config=Config(), n_models=len(datasets_name))

for i in range(len(datasets_name)):
    model_exp_2.gp_models[i].x_measurements[0] = np.concatenate([arr_data_train_2[i][0], arr_data_train_2[i][9]], axis=0, dtype='float32')
    model_exp_2.gp_models[i].x_measurements[1] = np.concatenate([arr_data_train_2[i][1], arr_data_train_2[i][10]], axis=0, dtype='float32')
    model_exp_2.gp_models[i].x_measurements[2] = np.concatenate([arr_data_train_2[i][2], arr_data_train_2[i][11]], axis=0, dtype='float32')
    model_exp_2.gp_models[i].x_measurements[3] = np.concatenate([arr_data_train_2[i][3], arr_data_train_2[i][12]], axis=0, dtype='float32')
    model_exp_2.gp_models[i].x_measurements[4] = np.concatenate([arr_data_train_2[i][4], arr_data_train_2[i][13]], axis=0, dtype='float32')
    model_exp_2.gp_models[i].x_measurements[5] = np.concatenate([arr_data_train_2[i][5], arr_data_train_2[i][14]], axis=0, dtype='float32')

    model_exp_2.gp_models[i].y_measurements[0] = np.concatenate([arr_data_train_2[i][6], arr_data_train_2[i][15]], axis=0, dtype='float32')
    model_exp_2.gp_models[i].y_measurements[1] = np.concatenate([arr_data_train_2[i][7], arr_data_train_2[i][16]], axis=0, dtype='float32')
    model_exp_2.gp_models[i].y_measurements[2] = np.concatenate([arr_data_train_2[i][8], arr_data_train_2[i][17]], axis=0, dtype='float32')

# model_exp_2.gp_model2.x_measurements[0] = np.concatenate([data_method_2_train_m2[0], data_method_2_train_m2[9]], axis=0, dtype='float32')
# model_exp_2.gp_model2.x_measurements[1] = np.concatenate([data_method_2_train_m2[1], data_method_2_train_m2[10]], axis=0, dtype='float32')
# model_exp_2.gp_model2.x_measurements[2] = np.concatenate([data_method_2_train_m2[2], data_method_2_train_m2[11]], axis=0, dtype='float32')
# model_exp_2.gp_model2.x_measurements[3] = np.concatenate([data_method_2_train_m2[3], data_method_2_train_m2[12]], axis=0, dtype='float32')
# model_exp_2.gp_model2.x_measurements[4] = np.concatenate([data_method_2_train_m2[4], data_method_2_train_m2[13]], axis=0, dtype='float32')
# model_exp_2.gp_model2.x_measurements[5] = np.concatenate([data_method_2_train_m2[5], data_method_2_train_m2[14]], axis=0, dtype='float32')
#
# model_exp_2.gp_model2.y_measurements[0] = np.concatenate([data_method_2_train_m2[6], data_method_2_train_m2[15]], axis=0, dtype='float32')
# model_exp_2.gp_model2.y_measurements[1] = np.concatenate([data_method_2_train_m2[7], data_method_2_train_m2[16]], axis=0, dtype='float32')
# model_exp_2.gp_model2.y_measurements[2] = np.concatenate([data_method_2_train_m2[8], data_method_2_train_m2[17]], axis=0, dtype='float32')

print('Method 2, training:')
for i in range(len(datasets_name)):
    scaled_x, scaled_y = model_exp_2.gp_models[i].init_gp()
    model_exp_2.gp_models[i].train_gp(scaled_x, scaled_y)

# scaled_x2, scaled_y2 = model_exp_2.gp_model2.init_gp()
# model_exp_2.gp_model2.train_gp(scaled_x2, scaled_y2)

means_exp_2 = []
uppers_exp_2 = []
lowers_exp_2 = []

print('Method 2, testing:')
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for i in range(data_method_2_test.shape[1]):
        # compute W based on x(t-1) and Y(t-1)
        vehicle_state = np.array([0.0,
                                  0.0,
                                  data_method_2_test[9][i].astype('float32'),  # vx
                                  0.0,
                                  data_method_2_test[10][i].astype('float32'),  # vy
                                  data_method_2_test[11][i].astype('float32'),  # yaw rate
                                  data_method_2_test[12][i].astype('float32'),  # steering angle
                                  ])
        u = np.array([
            data_method_2_test[13][i].astype('float32'),
            data_method_2_test[14][i].astype('float32'),
        ])
        Y_real = np.array([data_method_2_test[15][i].astype('float32'), data_method_2_test[16][i].astype('float32'),
                           data_method_2_test[17][i].astype('float32')]).reshape((3, 1))

        # predict Y(t) based on x(t) and W computed in previous step
        _, _, _, means = model_exp_2.scale_and_predict_model_step(vehicle_state, u)

        model_exp_2.compute_w(Y_real, means.squeeze().T,
                              prev_w=np.ones(len(datasets_name)).reshape((len(datasets_name), 1)) * 1.0 / len(datasets_name), eps=0.0, u=u)

        # predict Y(t) based on x(t) and W computed in previous step
        mean, lower, upper, _ = model_exp_2.scale_and_predict_model_step([0.0, 0.0, data_method_2_test[0][i], 0.0, data_method_2_test[1][i],
                                                                             data_method_2_test[2][i], data_method_2_test[3][i]],
                                                                            [data_method_2_test[4][i], data_method_2_test[5][i]])
        if (i + 1) % 200 == 0:
            print('Method 2, prediction %s/%s' % (i, data_method_2_test.shape[1]))
        means_exp_2.append(mean)
        uppers_exp_2.append(upper)
        lowers_exp_2.append(lower)

means_exp_2 = np.array(means_exp_2).squeeze()  # n x 3
uppers_exp_2 = np.array(uppers_exp_2).squeeze()  # n x 3
lowers_exp_2 = np.array(lowers_exp_2).squeeze()  # n x 3

errors_exp_2 = np.absolute(means_exp_2.T - np.array([data_method_2_test[6], data_method_2_test[7], data_method_2_test[8]]))

print('--------Method 2 results--------')
print('ax prediction average error: %f' % (np.sum(errors_exp_2[0, :]) / errors_exp_2[0, :].size))
print('ax prediction RMS error: %f' % (np.sqrt(np.sum(np.square(errors_exp_2[0, :]))/errors_exp_2[0, :].size)))
print('ax prediction maximum error: %f' % np.max(errors_exp_2[0, :]))
print('ay prediction average error: %f' % (np.sum(errors_exp_2[1, :]) / errors_exp_2[1, :].size))
print('ay prediction RMS error: %f' % (np.sqrt(np.sum(np.square(errors_exp_2[1, :]))/errors_exp_2[1, :].size)))
print('ay prediction maximum error: %f' % np.max(errors_exp_2[1, :]))
print('yaw rate prediction average error: %f' % (np.sum(errors_exp_2[2, :]) / errors_exp_2[2, :].size))
print('yaw rate prediction RMS error: %f' % (np.sqrt(np.sum(np.square(errors_exp_2[2, :]))/errors_exp_2[2, :].size)))
print('yaw rate prediction maximum error: %f' % np.max(errors_exp_2[2, :]))

plt.plot(errors_exp_1[0, :], label='Single GP')
plt.plot(errors_exp_2[0, :], label='Multi GP')
plt.legend()
plt.ylabel('error')
plt.xlabel('item')
plt.show()
