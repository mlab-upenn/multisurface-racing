import numpy as np


def dynamic_bicycle_model(state, u, Frx, Ffx, Ffy, Fry, mass=1212.6160, Iz=1560.3729, lf=0.88392, lr=1.50876):
    """ from: AMZ Driverless: The Full Autonomous Racing System
    model reference point: CoG
    longitudinal drive-train forces act on the center of gravity
    :param state: [x, y, yaw, vx, vy, yaw rate, steering angle]
    :param u: [drive force, steering speed]
    :param mass: vehicle mass
    :param Iz: Moment of inertia
    :param lf:
    :param lr:
    :return: f: dx/dt = f(x, u)
    """

    # currently no input/output saturation

    x, y, yaw, vx, vy, yaw_rate, steer_angle = state
    drive_force, steer_speed = u

    f = np.zeros(state.shape[0])
    f[0] = vx * np.cos(yaw) - vy * np.sin(yaw)
    f[1] = vx * np.sin(yaw) + vy * np.cos(yaw)
    f[2] = yaw_rate
    f[3] = 1.0 / mass * (Frx - Ffy * np.sin(steer_angle) + Ffx * np.cos(steer_angle) + vy * yaw_rate * mass)
    f[4] = 1.0 / mass * (Fry + Ffy * np.cos(steer_angle) + Ffx * np.sin(steer_angle) - vx * yaw_rate * mass)
    f[5] = 1.0 / Iz * (Ffy * lf * np.cos(steer_angle) - Fry * lr)
    f[6] = steer_speed

    return f


def calc_tire_forces(state, u, lf=0.88392, lr=1.50876, Df=3714.8218, Cf=5.9139, Bf=9.4246, Dr=3702.9280, Cr=1.3754,
                     Br=24.9504, Cm=0.9459, Cr0=2.3451, Cr2=-0.0095, torque_split=0):
    """from: AMZ Driverless: The Full Autonomous Racing System
    A simplified Pacejka tire model
    load transfer can be neglected, combined slip can be neglected
    :param state: [x, y, yaw, vx, vy, yaw rate, steering angle]
    :param u: [drive force, steering speed]
    :param lf:
    :param lr:
    :param Df:
    :param Cf:
    :param Bf:
    :param Dr:
    :param Cr:
    :param Br:
    :param Cm: Motor constant (1.0)
    :param Cr0: Rolling resistance
    :param Cr2: Drag
    :param torque_split: <0, 1> Split of torque between front and rear tire. (F1/10 ... 0.5)
    :return:
    """
    x, y, yaw, vx, vy, yaw_rate, steer_angle = state
    drive_force, steer_speed = u

    alfa_f = steer_angle - np.arctan((yaw_rate * lf + vy) / vx)
    alfa_r = np.arctan((yaw_rate * lr - vy) / vx)

    Ffy = Df * np.sin(Cf * np.arctan(Bf * alfa_f))
    Fry = Dr * np.sin(Cr * np.arctan(Br * alfa_r))

    Fx = Cm * drive_force - Cr0 - Cr2 * vx**2.0
    Frx = Fx * (1.0 - torque_split)
    Ffx = Fx * torque_split

    return Ffx, Frx, Ffy, Fry
