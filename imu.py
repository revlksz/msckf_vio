import numpy as np
from utils import skew, quaternion_to_rotation_matrix, Omega, normalize_quaternion, quaternion_multiply

class IMUState:
    def __init__(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.b_g = np.zeros(3)
        self.b_a = np.zeros(3)
        self.P = np.eye(15) * 1e-4

def propagate(state, P, imu_meas, dt, config):
    w_m, a_m = imu_meas
    w_true = w_m - state.b_g
    a_true = a_m - state.b_a
    R_WB = quaternion_to_rotation_matrix(state.q)
    p_dot = state.v
    v_dot = R_WB @ a_true + config.g
    w_quat = np.array([0.0, w_true[0], w_true[1], w_true[2]])
    q_dot = 0.5 * quaternion_multiply(state.q, w_quat)
    state.p = state.p + p_dot * dt
    state.v = state.v + v_dot * dt
    state.q = state.q + q_dot * dt
    state.q = normalize_quaternion(state.q)
    F = np.zeros((15, 15))
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 6:9] = -R_WB @ skew(a_true)
    F[3:6, 12:15] = -R_WB
    F[6:9, 6:9] = -skew(w_true)
    F[6:9, 9:12] = -np.eye(3)
    G = np.zeros((15, 12))
    G[3:6, 3:6] = -R_WB
    G[6:9, 0:3] = -np.eye(3)
    G[9:12, 6:9] = np.eye(3)
    G[12:15, 9:12] = np.eye(3)
    Q_c = np.zeros((12, 12))
    Q_c[0:3, 0:3] = np.eye(3) * config.gyro_noise_density**2
    Q_c[3:6, 3:6] = np.eye(3) * config.accel_noise_density**2
    Q_c[6:9, 6:9] = np.eye(3) * config.gyro_random_walk**2
    Q_c[9:12, 9:12] = np.eye(3) * config.accel_random_walk**2
    Fdt = F * dt
    Phi = np.eye(15) + Fdt + 0.5 * (Fdt @ Fdt)
    Q_d = G @ Q_c @ G.T * dt
    P_new = np.copy(P)
    P_II = P[0:15, 0:15]
    P_new_II = Phi @ P_II @ Phi.T + Q_d
    P_new[0:15, 0:15] = P_new_II
    if P.shape[0] > 15:
        P_IC = P[0:15, 15:]
        P_new_IC = Phi @ P_IC
        P_new[0:15, 15:] = P_new_IC
        P_new[15:, 0:15] = P_new_IC.T
    P_new = 0.5 * (P_new + P_new.T)
    return state, P_new
