import numpy as np
import scipy.linalg
from imu import IMUState, propagate
from utils import skew, quaternion_to_rotation_matrix, quaternion_multiply, rotation_matrix_to_quaternion

class CameraState:
    def __init__(self, p, q):
        self.p = p # Position in world
        self.q = q # Orientation (quaternion, World to Camera or Camera to World? Let's say Body to World for consistency with IMU, R_WC)

class MSCKF:
    def __init__(self, config):
        self.config = config
        self.state = IMUState()
        self.cam_states = [] # List of CameraState
        self.P = self.state.P # 15x15 initially
        
    def imu_callback(self, imu_meas, dt):
        """
        imu_meas: (w_m, a_m)
        """
        self.state, self.P = propagate(self.state, self.P, imu_meas, dt, self.config)
        
    def augment_state(self):
        """
        Add a new camera state to the sliding window.
        """
        # Calculate new camera pose
        R_WI = quaternion_to_rotation_matrix(self.state.q)
        
        # p_C = p_W + R_WI * p_IC
        p_C = self.state.p + R_WI @ self.config.p_CI
        
        # q_C = q_I * q_IC. Or R_WC = R_WI * R_IC
        R_WC = R_WI @ self.config.R_CI
        q_C = rotation_matrix_to_quaternion(R_WC)
        
        self.cam_states.append(CameraState(p_C, q_C))
        
        # Augment covariance matrix
        # Error state for camera: [delta p_C, delta theta_C] (6 elements)
        # J_I is 6x15 Jacobian w.r.t IMU state
        J_I = np.zeros((6, 15))
        J_I[0:3, 0:3] = np.eye(3)
        J_I[0:3, 6:9] = -R_WI @ skew(self.config.p_CI)
        J_I[3:6, 6:9] = self.config.R_CI.T # delta theta_C = R_IC^T * delta theta_I
        
        # The new row/col to add to P
        dim = self.P.shape[0]
        P_new = np.zeros((dim + 6, dim + 6))
        P_new[:dim, :dim] = self.P
        
        # Cross covariance
        # P_C_old = J_I * P_I_old
        # But we need full cross covariance with all existing states
        # [P_II, P_IC1...]
        # P_C_any = J_I * P_I_any
        J_full = np.zeros((6, dim))
        J_full[:, :15] = J_I
        
        P_cross = J_full @ self.P
        P_new[dim:, :dim] = P_cross
        P_new[:dim, dim:] = P_cross.T
        P_new[dim:, dim:] = J_full @ self.P @ J_full.T
        
        # Ensure symmetry
        self.P = 0.5 * (P_new + P_new.T)
        
    def triangulate(self, track):
        """
        Linear triangulation of a feature track.
        """
        A = []
        for obs in track.observations:
            cam_idx = obs[0]
            # Since pruning removes from left, cam_idx might be absolute or relative.
            # We need to map obs[0] to the current index in self.cam_states.
            # Assume we prune by removing from the beginning, so index is cam_idx - offset
            offset = self.cam_state_offset
            if cam_idx - offset < 0 or cam_idx - offset >= len(self.cam_states):
                continue
                
            cam = self.cam_states[cam_idx - offset]
            R_WC = quaternion_to_rotation_matrix(cam.q)
            R_CW = R_WC.T
            p_C = cam.p
            
            u, v = obs[1], obs[2]
            
            # Projection matrix P = K[R | t], but we use normalized coordinates
            # so P = [R_CW | -R_CW * p_C]
            P_mat = np.zeros((3, 4))
            P_mat[:, :3] = R_CW
            P_mat[:, 3] = -R_CW @ p_C
            
            A.append(u * P_mat[2, :] - P_mat[0, :])
            A.append(v * P_mat[2, :] - P_mat[1, :])
            
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        return X[:3]

    def update(self, tracks):
        if len(tracks) == 0:
            return
            
        H_o_all = []
        r_o_all = []
        
        for track in tracks:
            # Triangulate
            p_f_W = self.triangulate(track)
            
            H_x_j_list = []
            H_f_j_list = []
            r_j_list = []
            
            track_valid = True
            
            for obs in track.observations:
                cam_idx = obs[0]
                offset = self.cam_state_offset
                local_idx = cam_idx - offset
                
                if local_idx < 0 or local_idx >= len(self.cam_states):
                    continue
                    
                cam = self.cam_states[local_idx]
                R_WC = quaternion_to_rotation_matrix(cam.q)
                R_CW = R_WC.T
                
                # f in camera frame
                f_c = R_CW @ (p_f_W - cam.p)
                z = f_c[2]
                
                if z < 0.1: # behind camera or too close
                    continue
                    
                # Residual
                z_meas = np.array([obs[1], obs[2]])
                z_pred = np.array([f_c[0]/z, f_c[1]/z])
                r_j_curr = z_meas - z_pred
                
                # Outlier rejection based on reprojection error
                # 0.02 normalized units ~ 9-10 pixels error
                if np.linalg.norm(r_j_curr) > 0.02:
                    track_valid = False
                    break
                
                # Jacobians
                dz_dfc = np.array([
                    [1/z, 0, -f_c[0]/(z**2)],
                    [0, 1/z, -f_c[1]/(z**2)]
                ])
                
                dfc_d_pC = -R_CW
                dfc_d_thetaC = skew(f_c)
                
                H_c = np.zeros((2, 6))
                H_c[:, :3] = dz_dfc @ dfc_d_pC
                H_c[:, 3:] = dz_dfc @ dfc_d_thetaC
                
                H_f = dz_dfc @ R_CW
                
                H_x_curr = np.zeros((2, self.P.shape[0]))
                col_start = 15 + local_idx * 6
                H_x_curr[:, col_start:col_start+6] = H_c
                
                H_x_j_list.append(H_x_curr)
                H_f_j_list.append(H_f)
                r_j_list.append(r_j_curr)
                
            if not track_valid or len(H_x_j_list) < 3:
                continue
                
            H_x_j = np.vstack(H_x_j_list)
            H_f_j = np.vstack(H_f_j_list)
            r_j = np.concatenate(r_j_list)
            
            # Left Nullspace Projection
            Q, R_qr = np.linalg.qr(H_f_j, mode='complete')
            V = Q[:, 3:] # (2M, 2M-3)
            
            r_o = V.T @ r_j
            H_o = V.T @ H_x_j
            
            H_o_all.append(H_o)
            r_o_all.append(r_o)
            
        if len(H_o_all) == 0:
            return
            
        H_o_all = np.vstack(H_o_all)
        r_o_all = np.concatenate(r_o_all)
        
        # MSCKF Optimization: Reduce dimensionality if measurements exceed state dimension
        if H_o_all.shape[0] > self.P.shape[0]:
            Q_H, R_H = np.linalg.qr(H_o_all, mode='reduced')
            H_o_all = R_H
            r_o_all = Q_H.T @ r_o_all
        
        # EKF Update
        R_mat = np.eye(H_o_all.shape[0]) * self.config.observation_noise_variance
        
        S = H_o_all @ self.P @ H_o_all.T + R_mat
        
        try:
            # K = P * H_o^T * S^-1 => K * S = P * H_o^T
            # Solve S.T * K.T = (P * H_o^T).T => K.T = solve(S.T, H_o * P)
            K = scipy.linalg.solve(S, H_o_all @ self.P, assume_a='pos').T
            
            delta_x = K @ r_o_all
            
            # Update State
            self.apply_state_update(delta_x)
            
            # Update Covariance
            I_KH = np.eye(self.P.shape[0]) - K @ H_o_all
            self.P = I_KH @ self.P @ I_KH.T + K @ R_mat @ K.T
            self.P = 0.5 * (self.P + self.P.T)
        except np.linalg.LinAlgError:
            print("MSCKF Update SVD/Inverse failed. Skipping update.")
            
    def apply_state_update(self, delta_x):
        # IMU
        self.state.p += delta_x[0:3]
        self.state.v += delta_x[3:6]
        
        dq = np.array([1.0, 0.5*delta_x[6], 0.5*delta_x[7], 0.5*delta_x[8]])
        self.state.q = quaternion_multiply(self.state.q, dq)
        self.state.q /= np.linalg.norm(self.state.q)
        
        self.state.b_g += delta_x[9:12]
        self.state.b_a += delta_x[12:15]
        
        # Cameras
        for i in range(len(self.cam_states)):
            idx = 15 + i * 6
            self.cam_states[i].p += delta_x[idx:idx+3]
            
            dq_c = np.array([1.0, 0.5*delta_x[idx+3], 0.5*delta_x[idx+4], 0.5*delta_x[idx+5]])
            self.cam_states[i].q = quaternion_multiply(self.cam_states[i].q, dq_c)
            self.cam_states[i].q /= np.linalg.norm(self.cam_states[i].q)

    def prune_cam_states(self):
        """
        Removes oldest camera states if we exceed max_cam_states.
        """
        while len(self.cam_states) > self.config.max_cam_states:
            # Remove oldest (first)
            self.cam_states.pop(0)
            self.cam_state_offset += 1
            
            # Remove from covariance P
            # It's at index 15 to 20
            idx_remove = np.arange(15, 21)
            self.P = np.delete(self.P, idx_remove, axis=0)
            self.P = np.delete(self.P, idx_remove, axis=1)

    # Need an offset counter to align global cam_state indices in tracks with local array indices
    cam_state_offset = 0
