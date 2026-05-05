import numpy as np
import cv2
import scipy.optimize
import scipy.sparse
from utils import skew, quaternion_multiply, quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from camera import Camera

class IMUPreintegrator:
    def __init__(self, config, b_g=np.zeros(3), b_a=np.zeros(3)):
        self.config = config
        self.b_g = b_g.copy()
        self.b_a = b_a.copy()
        self.dp = np.zeros(3)
        self.dv = np.zeros(3)
        self.dq = np.array([1.0, 0.0, 0.0, 0.0])
        self.dq_dbg = np.zeros((3, 3))
        self.dt_sum = 0.0
        self.last_w = None
        self.last_a = None
        self.imu_buffer = []
        
    def integrate(self, w, a, dt):
        self.imu_buffer.append((w, a, dt))
        self._integrate_step(w, a, dt)
        
    def _integrate_step(self, w, a, dt):
        if self.last_w is None:
            self.last_w = w
            self.last_a = a
        un_w = 0.5 * (self.last_w + w) - self.b_g
        un_a_0 = self.last_a - self.b_a
        un_a_1 = a - self.b_a
        dtheta = un_w * dt
        norm_w = np.linalg.norm(dtheta)
        if norm_w > 1e-5:
            dq_delta = np.array([np.cos(norm_w/2), 
                                 dtheta[0]/norm_w*np.sin(norm_w/2),
                                 dtheta[1]/norm_w*np.sin(norm_w/2),
                                 dtheta[2]/norm_w*np.sin(norm_w/2)])
        else:
            dq_delta = np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]])
            dq_delta /= np.linalg.norm(dq_delta)
        dq_new = quaternion_multiply(self.dq, dq_delta)
        dq_new /= np.linalg.norm(dq_new)
        R_k = quaternion_to_rotation_matrix(self.dq)
        R_k_1 = quaternion_to_rotation_matrix(dq_new)
        un_a = 0.5 * (R_k @ un_a_0 + R_k_1 @ un_a_1)
        self.dp += self.dv * dt + 0.5 * un_a * dt**2
        self.dv += un_a * dt
        R_delta = quaternion_to_rotation_matrix(dq_delta)
        self.dq_dbg = R_delta.T @ self.dq_dbg - np.eye(3) * dt
        self.dq = dq_new
        self.dt_sum += dt
        self.last_w = w
        self.last_a = a
        
    def repropagate(self, new_bg, new_ba):
        self.b_g = new_bg.copy()
        self.b_a = new_ba.copy()
        self.dp = np.zeros(3)
        self.dv = np.zeros(3)
        self.dq = np.array([1.0, 0.0, 0.0, 0.0])
        self.dq_dbg = np.zeros((3, 3))
        self.dt_sum = 0.0
        self.last_w = None
        self.last_a = None
        buffer = self.imu_buffer
        self.imu_buffer = []
        for w, a, dt in buffer:
            self.integrate(w, a, dt)

class Initializer:
    def __init__(self, config):
        self.config = config
        self.camera = Camera(config)
        self.window_size = 15
        self.frames = []
        self.b_g = np.zeros(3)
        self.b_a = np.zeros(3)
        self.is_initialized = False
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, fastThreshold=20)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_imu_time = -1.0
        
    def add_frame(self, img, imu_buffer):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        kps, des = self.orb.detectAndCompute(gray, None)
        if kps is None or len(kps) < 50:
            return False
        pts_2d = np.array([kp.pt for kp in kps])
        norm_pts = self.camera.unproject(pts_2d)
        pre_int = IMUPreintegrator(self.config, self.b_g, self.b_a)
        for ts, (w, a) in imu_buffer:
            if self.last_imu_time > 0:
                dt = ts - self.last_imu_time
                if dt > 0:
                    pre_int.integrate(w, a, dt)
            self.last_imu_time = ts
        frame_data = {
            'kps': kps, 'des': des, 'norm_pts': norm_pts,
            'pre_int': pre_int, 'p_c': np.zeros(3),
            'q_c': np.array([1.0, 0.0, 0.0, 0.0])
        }
        self.frames.append(frame_data)
        if len(self.frames) > self.window_size:
            self.frames.pop(0)
        if len(self.frames) == self.window_size:
            return self.initialize()
        return False
        
    def initialize(self):
        print("Starting VINS Initialization...")
        if not self._vision_only_sfm():
            print("SfM Failed.")
            return False
        self._calibrate_gyro_bias()
        success, g_v, s, velocities = self._linear_alignment()
        if not success:
            print("Linear Alignment Failed.")
            return False
        g_v = self._refine_gravity(g_v, velocities)
        if np.linalg.norm(g_v) < 9.0 or np.linalg.norm(g_v) > 11.0:
            print(f"Gravity refinement failed. g norm = {np.linalg.norm(g_v)}")
            return False
        # Align to World Frame
        g_w = np.array([0, 0, 9.81])
        g_v_norm = g_v / np.linalg.norm(g_v)
        g_w_norm = g_w / np.linalg.norm(g_w)
        v = np.cross(g_v_norm, g_w_norm)
        c = np.dot(g_v_norm, g_w_norm)
        if c > 0.9999:
            R_v_w = np.eye(3)
        elif c < -0.9999:
            R_v_w = -np.eye(3)
        else:
            s_val = np.linalg.norm(v)
            v_skew = skew(v)
            R_v_w = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1 - c) / (s_val**2))
        self.final_states = []
        for i in range(self.window_size):
            p_b_v = self.frames[i]['p_b'] * s
            q_b_v = self.frames[i]['q_b']
            p_w = R_v_w @ p_b_v
            R_b_v = quaternion_to_rotation_matrix(q_b_v)
            R_b_w = R_v_w @ R_b_v
            q_w = rotation_matrix_to_quaternion(R_b_w)
            v_w = R_v_w @ velocities[i]
            self.final_states.append({'p': p_w, 'q': q_w, 'v': v_w})
        self.is_initialized = True
        print("Initialization Successful!")
        return True
        
    def _vision_only_sfm(self):
        max_parallax = 0
        best_pair = None
        best_matches = None
        best_R = None
        best_t = None
        for i in range(self.window_size - 1):
            matches = self.matcher.match(self.frames[i]['des'], self.frames[-1]['des'])
            if len(matches) < 30:
                continue
            src_pts = np.float32([self.frames[i]['norm_pts'][m.queryIdx] for m in matches])
            dst_pts = np.float32([self.frames[-1]['norm_pts'][m.trainIdx] for m in matches])
            E, mask = cv2.findEssentialMat(src_pts, dst_pts, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=0.01)
            if mask is None or np.sum(mask) < 20:
                continue
            parallax = np.mean(np.linalg.norm(src_pts - dst_pts, axis=1))
            if parallax > max_parallax:
                _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, np.eye(3), mask=mask)
                if np.sum(mask_pose) > 20:
                    max_parallax = parallax
                    best_pair = (i, self.window_size - 1)
                    best_matches = [m for j, m in enumerate(matches) if mask_pose[j][0] > 0]
                    best_R = R
                    best_t = t
        if best_pair is None or max_parallax < 0.05:
            return False
        idx1, idx2 = best_pair
        self.frames[idx1]['p_c'] = np.zeros(3)
        self.frames[idx1]['q_c'] = np.array([1.0, 0.0, 0.0, 0.0])
        self.frames[idx2]['p_c'] = best_t.flatten()
        self.frames[idx2]['q_c'] = rotation_matrix_to_quaternion(best_R)
        pts_1 = np.float32([self.frames[idx1]['norm_pts'][m.queryIdx] for m in best_matches]).T
        pts_2 = np.float32([self.frames[idx2]['norm_pts'][m.trainIdx] for m in best_matches]).T
        P1 = np.hstack((np.eye(3), np.zeros((3,1)))).astype(np.float64)
        P2 = np.hstack((best_R, best_t)).astype(np.float64)
        pts_1 = pts_1.astype(np.float64)
        pts_2 = pts_2.astype(np.float64)
        print(f"P1: {P1.shape}, {P1.dtype}")
        print(f"P2: {P2.shape}, {P2.dtype}")
        print(f"pts_1: {pts_1.shape}, {pts_1.dtype}")
        points_4d = cv2.triangulatePoints(P1, P2, pts_1, pts_2)
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T
        for i in range(self.window_size):
            if i == idx1 or i == idx2:
                continue
            matches = self.matcher.match(self.frames[idx1]['des'], self.frames[i]['des'])
            valid_2d = []
            valid_3d = []
            for m in matches:
                for j, bm in enumerate(best_matches):
                    if m.queryIdx == bm.queryIdx:
                        valid_2d.append(self.frames[i]['norm_pts'][m.trainIdx])
                        valid_3d.append(points_3d[j])
                        break
            if len(valid_2d) >= 10:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(valid_3d), np.array(valid_2d), np.eye(3), None)
                if success:
                    R_c, _ = cv2.Rodrigues(rvec)
                    self.frames[i]['q_c'] = rotation_matrix_to_quaternion(R_c)
                    self.frames[i]['p_c'] = tvec.flatten()
                else:
                    return False
            else:
                return False
        # Bundle Adjustment
        n_cams = self.window_size
        n_points = len(points_3d)
        camera_params = np.zeros((n_cams, 6))
        for i in range(n_cams):
            if i == idx1:
                continue
            R_c = quaternion_to_rotation_matrix(self.frames[i]['q_c'])
            rvec, _ = cv2.Rodrigues(R_c)
            camera_params[i, :3] = rvec.flatten()
            camera_params[i, 3:] = self.frames[i]['p_c'].flatten()
        points_params = points_3d.copy()
        observations = []
        for j, bm in enumerate(best_matches):
            query_idx = bm.queryIdx
            for i in range(n_cams):
                matches_i = self.matcher.match(self.frames[idx1]['des'], self.frames[i]['des'])
                for m_i in matches_i:
                    if m_i.queryIdx == query_idx:
                        u, v = self.frames[i]['norm_pts'][m_i.trainIdx]
                        observations.append((i, j, u, v))
                        break
        observations = np.array(observations)
        def fun(params, n_cams, n_points, obs, fixed_cam_idx):
            cam_p = np.zeros((n_cams, 6))
            mask = np.ones(n_cams, dtype=bool)
            mask[fixed_cam_idx] = False
            cam_p[mask] = params[:(n_cams-1)*6].reshape((n_cams-1, 6))
            pts_3d = params[(n_cams-1)*6:].reshape((n_points, 3))
            
            cam_indices = obs[:, 0].astype(int)
            pt_indices = obs[:, 1].astype(int)
            
            pts = pts_3d[pt_indices]
            res = np.zeros(len(obs) * 2)
            
            for i in range(n_cams):
                c_mask = (cam_indices == i)
                if not np.any(c_mask):
                    continue
                rvec = cam_p[i, :3]
                tvec = cam_p[i, 3:]
                pts_i = pts[c_mask]
                
                proj, _ = cv2.projectPoints(pts_i, rvec, tvec, np.eye(3), np.zeros(4))
                proj = proj.reshape(-1, 2)
                
                res_u = proj[:, 0] - obs[c_mask, 2]
                res_v = proj[:, 1] - obs[c_mask, 3]
                
                indices = np.where(c_mask)[0]
                res[indices * 2] = res_u
                res[indices * 2 + 1] = res_v
                
            return res

        x0 = np.hstack((
            np.delete(camera_params, idx1, axis=0).flatten(),
            points_params.flatten()
        ))
        print("Running Bundle Adjustment...")
        
        # Build jacobian sparsity matrix
        n_params = (n_cams - 1) * 6 + n_points * 3
        A = scipy.sparse.lil_matrix((len(observations) * 2, n_params), dtype=int)
        
        cam_param_idx = np.zeros(n_cams, dtype=int)
        p_idx = 0
        for i in range(n_cams):
            if i == idx1:
                cam_param_idx[i] = -1
            else:
                cam_param_idx[i] = p_idx
                p_idx += 1

        for i, (cam_idx, pt_idx, _, _) in enumerate(observations):
            cam_idx = int(cam_idx)
            pt_idx = int(pt_idx)
            row_u = i * 2
            row_v = i * 2 + 1
            
            c_p_idx = cam_param_idx[cam_idx]
            if c_p_idx != -1:
                cam_col = c_p_idx * 6
                A[row_u, cam_col:cam_col+6] = 1
                A[row_v, cam_col:cam_col+6] = 1
                
            pt_col = (n_cams - 1) * 6 + pt_idx * 3
            A[row_u, pt_col:pt_col+3] = 1
            A[row_v, pt_col:pt_col+3] = 1

        res = scipy.optimize.least_squares(
            fun, x0, verbose=0, jac_sparsity=A, x_scale='jac', ftol=1e-3, method='trf', 
            args=(n_cams, n_points, observations, idx1)
        )
        opt_params = res.x
        opt_cams = np.zeros((n_cams, 6))
        mask = np.ones(n_cams, dtype=bool)
        mask[idx1] = False
        opt_cams[mask] = opt_params[:(n_cams-1)*6].reshape((n_cams-1, 6))
        
        for i in range(n_cams):
            if i == idx1:
                continue
            rvec = opt_cams[i, :3]
            tvec = opt_cams[i, 3:]
            R_c, _ = cv2.Rodrigues(rvec)
            self.frames[i]['q_c'] = rotation_matrix_to_quaternion(R_c)
            self.frames[i]['p_c'] = tvec
        # Transform Camera poses to Body poses
        R_IC = self.config.R_CI
        p_IC = self.config.p_CI
        for i in range(self.window_size):
            q_c_v = self.frames[i]['q_c']
            p_c_v = self.frames[i]['p_c']
            R_c_v = quaternion_to_rotation_matrix(q_c_v)
            R_b_v = R_c_v @ R_IC.T
            self.frames[i]['q_b'] = rotation_matrix_to_quaternion(R_b_v)
            self.frames[i]['p_b'] = p_c_v - R_b_v @ p_IC
        return True
        
    def _calibrate_gyro_bias(self):
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for i in range(self.window_size - 1):
            q_b_k = self.frames[i]['q_b']
            q_b_k1 = self.frames[i+1]['q_b']
            dq = self.frames[i+1]['pre_int'].dq
            q_k_inv = q_b_k.copy()
            q_k_inv[1:] *= -1
            q_rel = quaternion_multiply(q_k_inv, q_b_k1)
            dq_inv = dq.copy()
            dq_inv[1:] *= -1
            err_q = quaternion_multiply(dq_inv, q_rel)
            err_theta = 2.0 * err_q[1:]
            J = self.frames[i+1]['pre_int'].dq_dbg
            A += J.T @ J
            b += J.T @ err_theta
        delta_bg = np.linalg.solve(A + np.eye(3)*1e-6, b)
        self.b_g += delta_bg
        for i in range(1, self.window_size):
            self.frames[i]['pre_int'].repropagate(self.b_g, self.b_a)
            
    def _linear_alignment(self):
        n = self.window_size
        dim = 3 * n + 4
        A = np.zeros((6 * (n - 1), dim))
        b = np.zeros(6 * (n - 1))
        for i in range(n - 1):
            dt = self.frames[i+1]['pre_int'].dt_sum
            dp = self.frames[i+1]['pre_int'].dp
            dv = self.frames[i+1]['pre_int'].dv
            R_b_k = quaternion_to_rotation_matrix(self.frames[i]['q_b'])
            p_b_k = self.frames[i]['p_b']
            p_b_k1 = self.frames[i+1]['p_b']
            row = i * 6
            A[row:row+3, i*3:i*3+3] = -R_b_k.T * dt
            A[row:row+3, dim-4:dim-1] = 0.5 * R_b_k.T * dt**2
            A[row:row+3, dim-1] = R_b_k.T @ (p_b_k1 - p_b_k)
            b[row:row+3] = dp
            A[row+3:row+6, i*3:i*3+3] = -R_b_k.T
            A[row+3:row+6, (i+1)*3:(i+1)*3+3] = R_b_k.T
            A[row+3:row+6, dim-4:dim-1] = R_b_k.T * dt
            b[row+3:row+6] = dv
        X, residuals, rank, s_vals = np.linalg.lstsq(A, b, rcond=None)
        velocities = [X[i*3:i*3+3] for i in range(n)]
        g_v = X[dim-4:dim-1]
        scale = X[dim-1]
        if scale < 0:
            print("Negative scale!")
            return False, g_v, scale, velocities
        return True, g_v, scale, velocities
        
    def _refine_gravity(self, g_v, velocities):
        G_MAG = 9.81
        n = self.window_size
        dim = 3 * n + 3
        for iter in range(4):
            g_norm = g_v / np.linalg.norm(g_v)
            b1 = np.cross(g_norm, np.array([1,0,0]))
            if np.linalg.norm(b1) < 1e-4:
                b1 = np.cross(g_norm, np.array([0,1,0]))
            b1 = b1 / np.linalg.norm(b1)
            b2 = np.cross(g_norm, b1)
            A = np.zeros((6 * (n - 1), dim))
            b = np.zeros(6 * (n - 1))
            for i in range(n - 1):
                dt = self.frames[i+1]['pre_int'].dt_sum
                dp = self.frames[i+1]['pre_int'].dp
                dv = self.frames[i+1]['pre_int'].dv
                R_b_k = quaternion_to_rotation_matrix(self.frames[i]['q_b'])
                p_b_k = self.frames[i]['p_b']
                p_b_k1 = self.frames[i+1]['p_b']
                row = i * 6
                A[row:row+3, i*3:i*3+3] = -R_b_k.T * dt
                A[row:row+3, dim-3] = 0.5 * R_b_k.T @ b1 * dt**2
                A[row:row+3, dim-2] = 0.5 * R_b_k.T @ b2 * dt**2
                A[row:row+3, dim-1] = R_b_k.T @ (p_b_k1 - p_b_k)
                b[row:row+3] = dp - 0.5 * R_b_k.T @ g_norm * G_MAG * dt**2
                A[row+3:row+6, i*3:i*3+3] = -R_b_k.T
                A[row+3:row+6, (i+1)*3:(i+1)*3+3] = R_b_k.T
                A[row+3:row+6, dim-3] = R_b_k.T @ b1 * dt
                A[row+3:row+6, dim-2] = R_b_k.T @ b2 * dt
                b[row+3:row+6] = dv - R_b_k.T @ g_norm * G_MAG * dt
            X, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            w1 = X[dim-3]
            w2 = X[dim-2]
            dg = w1 * b1 + w2 * b2
            g_v = g_norm * G_MAG + dg
            if np.linalg.norm(dg) < 1e-3:
                break
        return g_v
