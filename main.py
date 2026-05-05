import time
import numpy as np
import cv2
import os
from utils import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion

from config import Config
from camera import Camera
from feature_tracker import FeatureTracker
from msckf_filter import MSCKF
from visualizer3d import Visualizer3D
from data_loader import EurocDataLoader, DummyDataLoader
from initializer import Initializer

def main():
    print("Initializing MSCKF VIO System...")
    config = Config()
    camera = Camera(config)
    feature_tracker = FeatureTracker(config, camera)
    msckf = MSCKF(config)
    viz = Visualizer3D(live_update_every=5)
    
    dataset_path = os.path.join(os.path.dirname(__file__), 'MH_02_easy')
    if os.path.exists(dataset_path):
        print(f"Loading EuRoC dataset from: {dataset_path}")
        loader = EurocDataLoader(dataset_path)
    else:
        print("Dataset not found, falling back to dummy data.")
        loader = DummyDataLoader(duration_sec=10.0)
    
    last_imu_time = -1.0
    cam_state_idx = -1
    
    start_time = time.time()
    frames_processed = 0
    
    initializer = Initializer(config)
    imu_buffer = []
    initialized = False
    # Visualization-only alignment: maps VIO frame → GT frame at t=0
    viz_R_align = np.eye(3)
    viz_t_align = np.zeros(3)
    
    print("Starting VIO Loop...")
    
    try:
        for sensor_data in loader:
            if not initialized:
                if sensor_data.type == 'imu':
                    imu_buffer.append((sensor_data.timestamp, sensor_data.data))
                elif sensor_data.type == 'image':
                    img = sensor_data.data
                    # viz.update_camera(img, [], fps=0, initializing=True)
                    
                    success = initializer.add_frame(img, imu_buffer)
                    imu_buffer = [] # Clear buffer
                    
                    if success:
                        final_state = initializer.final_states[-1]
                        
                        msckf.state.p = final_state['p'].copy()
                        msckf.state.q = final_state['q'].copy()
                        msckf.state.v = final_state['v'].copy()
                        msckf.state.b_g = initializer.b_g.copy()
                        msckf.state.b_a = np.zeros(3)
                        
                        last_imu_time = sensor_data.timestamp
                        initialized = True
                        
                        # ── Visualization-only alignment ──────────────────────────
                        # Compute a rigid SE(3) transform so that the VIO's starting
                        # position and orientation visually coincide with GT.
                        # The MSCKF filter state is NEVER modified here.
                        gt_state = loader.get_gt_state(sensor_data.timestamp)
                        if gt_state is not None:
                            R_vio0 = quaternion_to_rotation_matrix(final_state['q'])
                            R_gt0  = quaternion_to_rotation_matrix(gt_state['q'])
                            viz_R_align = R_gt0 @ R_vio0.T          # rotate VIO → GT frame
                            viz_t_align = gt_state['p'] - viz_R_align @ final_state['p']
                            print(f"[Viz] Initial alignment: t={viz_t_align}")
                        
                        print("MSCKF Initialized from VINS-Mono method!")
                continue
                
            if sensor_data.type == 'imu':
                if last_imu_time < 0:
                    last_imu_time = sensor_data.timestamp
                    continue
                    
                dt = sensor_data.timestamp - last_imu_time
                if dt > 0:
                    msckf.imu_callback(sensor_data.data, dt)
                last_imu_time = sensor_data.timestamp
                
            elif sensor_data.type == 'image':
                cam_state_idx += 1
                
                # 1. Augment MSCKF state
                msckf.augment_state()
                
                # 2. Track features
                img = sensor_data.data
                feature_tracker.track(img, cam_state_idx)
                
                # 3. MSCKF Update with mature tracks
                mature_tracks = feature_tracker.get_mature_tracks()
                if len(mature_tracks) > 0:
                    msckf.update(mature_tracks)
                    
                # 4. Prune old camera states
                msckf.prune_cam_states()
                
                # Visualization
                frames_processed += 1
                elapsed = time.time() - start_time
                fps = frames_processed / elapsed if elapsed > 0 else 0
                
                if frames_processed % 30 == 0:
                    gt_p = loader.get_gt_pose(sensor_data.timestamp)
                    
                    # Apply visualization alignment (VIO state is untouched)
                    p_viz = viz_R_align @ msckf.state.p + viz_t_align
                    R_viz = viz_R_align @ quaternion_to_rotation_matrix(msckf.state.q)
                    q_viz = rotation_matrix_to_quaternion(R_viz)
                    
                    viz.update_trajectory(p_viz, q_viz, gt_p_W=gt_p)
                    num_features = len(feature_tracker.active_tracks)
                    print(f"Frame {frames_processed} | FPS: {fps:.1f} | Features: {num_features} | Pos: {p_viz} | GT: {gt_p}")
                
                # if frames_processed % 2 == 0:
                #     viz.update_camera(img, feature_tracker.active_tracks, fps=fps)
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
        
    print("VIO Loop Finished.")
    viz.show_interactive()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
