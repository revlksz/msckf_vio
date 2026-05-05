import numpy as np

class Config:
    def __init__(self):
        # IMU Parameters
        self.imu_freq = 200.0  # Hz
        self.cam_freq = 20.0   # Hz
        self.g = np.array([0, 0, -9.81])  # Gravity vector magnitude (we'll handle direction in IMU)
        
        # Noise parameters (continuous time)
        # Sample values, based on EuRoC dataset typically
        self.gyro_noise_density = 1.6968e-04
        self.accel_noise_density = 2.0000e-3
        self.gyro_random_walk = 1.9393e-05
        self.accel_random_walk = 3.0000e-03
        
        # Camera to IMU Extrinsics (T_IMU_CAM or R_CI, p_CI)
        # EuRoC cam0 to IMU extrinsics (T_BS)
        T_BS = np.array([
            [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
            [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
            [-0.0257744366974, 0.00375618835797, 0.999660727108, 0.00981073058949],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.R_CI = T_BS[:3, :3]
        self.p_CI = T_BS[:3, 3]
        
        # Camera Intrinsics
        # Default placeholder (EuRoC cam0)
        self.K = np.array([
            [458.654, 0, 367.215],
            [0, 457.296, 248.375],
            [0, 0, 1]
        ])
        self.D = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]) # Distortion
        
        # MSCKF Parameters
        self.max_cam_states = 20
        self.min_parallax = 0.005 # Lower parallax threshold to prevent starvation when stationary
        self.max_track_horizon = 10 # Frames a feature can be unseen
        self.min_track_length = 3 # Minimum observations to be processed
        self.observation_noise = 1.5 # Pixels
        self.observation_noise_variance = (self.observation_noise / self.K[0,0])**2 # Approx normalized var
        
        # ORB Parameters
        self.orb_nfeatures = 150
        self.orb_scaleFactor = 1.2
        self.orb_nlevels = 4
        self.orb_fastThreshold = 20
