import numpy as np
import cv2
import os
import csv

class SensorData:
    def __init__(self, type, timestamp, data):
        self.type = type # 'imu' or 'image'
        self.timestamp = timestamp
        self.data = data # tuple (w, a) for imu, np.ndarray for image

class EurocDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        
        self.imu_data = []
        self.cam_data = []
        self.gt_data = []
        
        self._load_imu()
        self._load_cam()
        self._load_gt()
        
        self.imu_idx = 0
        self.cam_idx = 0
        self.gt_idx = 0
        
    def _load_imu(self):
        imu_csv = os.path.join(self.data_path, 'mav0', 'imu0', 'data.csv')
        with open(imu_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                ts = float(row[0]) * 1e-9
                w = np.array([float(row[1]), float(row[2]), float(row[3])])
                a = np.array([float(row[4]), float(row[5]), float(row[6])])
                self.imu_data.append((ts, w, a))
                
    def _load_cam(self):
        cam_csv = os.path.join(self.data_path, 'mav0', 'cam0', 'data.csv')
        with open(cam_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                ts = float(row[0]) * 1e-9
                filename = row[1]
                self.cam_data.append((ts, filename))
                
    def _load_gt(self):
        gt_csv = os.path.join(self.data_path, 'mav0', 'state_groundtruth_estimate0', 'data.csv')
        if not os.path.exists(gt_csv):
            return
            
        with open(gt_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                ts = float(row[0]) * 1e-9
                p_W = np.array([float(row[1]), float(row[2]), float(row[3])])
                q_W = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
                v_W = np.array([float(row[8]), float(row[9]), float(row[10])])
                b_g = np.array([float(row[11]), float(row[12]), float(row[13])])
                b_a = np.array([float(row[14]), float(row[15]), float(row[16])])
                
                self.gt_data.append({
                    'timestamp': ts,
                    'p': p_W,
                    'q': q_W,
                    'v': v_W,
                    'b_g': b_g,
                    'b_a': b_a
                })
                
    def get_gt_pose(self, timestamp):
        if not self.gt_data:
            return None
        
        while self.gt_idx < len(self.gt_data) - 1:
            if abs(self.gt_data[self.gt_idx]['timestamp'] - timestamp) < abs(self.gt_data[self.gt_idx + 1]['timestamp'] - timestamp):
                break
            self.gt_idx += 1
            
        if self.gt_idx < len(self.gt_data):
            return self.gt_data[self.gt_idx]['p']
        return None
        
    def get_gt_state(self, timestamp):
        if not self.gt_data:
            return None
        # simple search for closest
        idx = 0
        while idx < len(self.gt_data) - 1:
            if abs(self.gt_data[idx]['timestamp'] - timestamp) < abs(self.gt_data[idx + 1]['timestamp'] - timestamp):
                break
            idx += 1
        return self.gt_data[idx]
                
    def __iter__(self):
        return self
        
    def __next__(self):
        # Determine whether the next data is IMU or Camera based on timestamp
        if self.imu_idx >= len(self.imu_data) and self.cam_idx >= len(self.cam_data):
            raise StopIteration
            
        imu_ts = self.imu_data[self.imu_idx][0] if self.imu_idx < len(self.imu_data) else float('inf')
        cam_ts = self.cam_data[self.cam_idx][0] if self.cam_idx < len(self.cam_data) else float('inf')
        
        if imu_ts <= cam_ts:
            # Yield IMU data
            ts, w, a = self.imu_data[self.imu_idx]
            self.imu_idx += 1
            return SensorData('imu', ts, (w, a))
        else:
            # Yield Camera data
            ts, filename = self.cam_data[self.cam_idx]
            self.cam_idx += 1
            img_path = os.path.join(self.data_path, 'mav0', 'cam0', 'data', filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
            return SensorData('image', ts, img)

class DummyDataLoader:
    """
    A simple loader that generates dummy IMU and blank images, 
    useful to test the pipeline without a real dataset.
    """
    def __init__(self, duration_sec=5.0, imu_freq=200.0, cam_freq=20.0):
        self.duration = duration_sec
        self.imu_freq = imu_freq
        self.cam_freq = cam_freq
        
        self.imu_dt = 1.0 / self.imu_freq
        self.cam_dt = 1.0 / self.cam_freq
        
        self.current_time = 0.0
        self.next_cam_time = 0.0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.current_time > self.duration:
            raise StopIteration
            
        if self.current_time >= self.next_cam_time:
            # Generate a dummy image (e.g., random noise to detect some features)
            img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            data = SensorData('image', self.next_cam_time, img)
            self.next_cam_time += self.cam_dt
        else:
            # Generate dummy IMU (mostly stationary with some gravity)
            # w = 0, a = [0, 0, 9.81]
            w = np.array([0.0, 0.0, 0.0])
            # To make it interesting, let's add a small rotation rate
            w[2] = 0.1 * np.sin(self.current_time)
            
            a = np.array([0.0, 0.0, 9.81])
            data = SensorData('imu', self.current_time, (w, a))
            self.current_time += self.imu_dt
            
        return data

    def get_gt_pose(self, timestamp):
        return None

    def get_gt_state(self, timestamp):
        return {
            'timestamp': timestamp,
            'p': np.zeros(3),
            'q': np.array([1.,0.,0.,0.]),
            'v': np.zeros(3),
            'b_g': np.zeros(3),
            'b_a': np.zeros(3)
        }
