# import time
# import numpy as np
# import cv2
# import os
# from utils import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion

# from config import Config
# from camera import Camera
# from feature_tracker import FeatureTracker
# from msckf_filter import MSCKF
# from visualizer3d import Visualizer3D
# from data_loader import EurocDataLoader, DummyDataLoader
# from initializer import Initializer

# def main():
#     print("Initializing MSCKF VIO System...")
#     config = Config()
#     camera = Camera(config)
#     feature_tracker = FeatureTracker(config, camera)
#     msckf = MSCKF(config)
#     viz = Visualizer3D(live_update_every=5)
    
#     dataset_path = os.path.join(os.path.dirname(__file__), 'MH_02_easy')
#     if os.path.exists(dataset_path):
#         print(f"Loading EuRoC dataset from: {dataset_path}")
#         loader = EurocDataLoader(dataset_path)
#     else:
#         print("Dataset not found, falling back to dummy data.")
#         loader = DummyDataLoader(duration_sec=10.0)
    
#     last_imu_time = -1.0
#     cam_state_idx = -1
    
#     start_time = time.time()
#     frames_processed = 0
    
#     initializer = Initializer(config)
#     imu_buffer = []
#     initialized = False  
#     imu_window = []          
#     IMU_WINDOW_SIZE = 40
#     stationary_count = 0     # consecutive stationary camera frames (debounce)
#     ZUPT_DEBOUNCE = 5        # require N consecutive still frames before ZUPT fires
#     # Visualization-only alignment: maps VIO frame → GT frame at t=0
#     viz_R_align = np.eye(3)
#     viz_t_align = np.zeros(3)
    
#     print("Starting VIO Loop...")
    
#     try:
#         for sensor_data in loader:
#             if not initialized:
#                 if sensor_data.type == 'imu':
#                     imu_buffer.append((sensor_data.timestamp, sensor_data.data))
#                 elif sensor_data.type == 'image':
#                     img = sensor_data.data
#                     viz.update_camera(img, [], fps=0, initializing=True)
                    
#                     success = initializer.add_frame(img, imu_buffer)
#                     imu_buffer_snapshot = list(imu_buffer)  # keep for b_a estimate
#                     imu_buffer = [] # Clear buffer
                    
#                     if success:
#                         final_state = initializer.final_states[-1]
                        
#                         msckf.state.p = final_state['p'].copy()
#                         msckf.state.q = final_state['q'].copy()
#                         msckf.state.v = final_state['v'].copy()
#                         msckf.state.b_g = initializer.b_g.copy()
#                         # Estimate b_a from static period: mean(a) - g_body
#                         if len(imu_buffer_snapshot) > 10:
#                             acc_vecs = np.array([a for _, (_, a) in imu_buffer_snapshot])
#                             mean_a = np.mean(acc_vecs, axis=0)
#                             R0 = quaternion_to_rotation_matrix(final_state['q'])
#                             g_body = R0.T @ np.array([0, 0, 9.81])
#                             msckf.state.b_a = mean_a - g_body
#                             print(f"[Init] b_a estimate: {msckf.state.b_a}")
#                         else:
#                             msckf.state.b_a = np.zeros(3)
                        
#                         last_imu_time = sensor_data.timestamp
#                         initialized = True
                        
#                         # ── Visualization-only alignment ──────────────────────────
#                         # Compute a rigid SE(3) transform so that the VIO's starting
#                         # position and orientation visually coincide with GT.
#                         # The MSCKF filter state is NEVER modified here.
#                         gt_state = loader.get_gt_state(sensor_data.timestamp)
#                         if gt_state is not None:
#                             R_vio0 = quaternion_to_rotation_matrix(final_state['q'])
#                             R_gt0  = quaternion_to_rotation_matrix(gt_state['q'])
#                             viz_R_align = R_gt0 @ R_vio0.T          # rotate VIO → GT frame
#                             viz_t_align = gt_state['p'] - viz_R_align @ final_state['p']
#                             print(f"[Viz] Initial alignment: t={viz_t_align}")
                        
#                         print("MSCKF Initialized from VINS-Mono method!")
#                 continue
                
#             if sensor_data.type == 'imu':
#                 if last_imu_time < 0:
#                     last_imu_time = sensor_data.timestamp
#                     continue
                    
#                 dt = sensor_data.timestamp - last_imu_time
#                 if dt > 0:
#                     msckf.imu_callback(sensor_data.data, dt)
#                 last_imu_time = sensor_data.timestamp

#                 # Maintain rolling IMU window for ZUPT
#                 imu_window.append(sensor_data.data)
#                 if len(imu_window) > IMU_WINDOW_SIZE:
#                     imu_window.pop(0)
                
#             elif sensor_data.type == 'image':
#                 img = sensor_data.data
#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                
#                 # -------------------------------------------------------------
#                 # ORB TABANLI PARALAKS (KEYFRAME) KONTROLÜ
#                 # -------------------------------------------------------------
#                 is_keyframe = True
#                 MIN_PARALLAX_PIXELS = 3.0
                
#                 # Gelen karedeki ORB özelliklerini çıkar
#                 kps, des = feature_tracker.orb.detectAndCompute(gray, None)
                
#                 # Eğer daha önceden kaydedilmiş bir özellik havuzu varsa karşılaştır
#                 if feature_tracker.prev_des is not None and des is not None and len(des) > 0:
#                     # Mevcut kareyi, kabul edilen son ANAHTAR KARE (prev_des) ile eşleştir
#                     matches = feature_tracker.matcher.match(feature_tracker.prev_des, des)
                    
#                     if len(matches) > 8:
#                         src_pts = np.float32([feature_tracker.prev_kps[m.queryIdx].pt for m in matches])
#                         dst_pts = np.float32([kps[m.trainIdx].pt for m in matches])
                        
#                         # Eşleşen noktalar arasındaki ortalama piksel mesafesini (parallax) hesapla
#                         avg_parallax = np.mean(np.linalg.norm(src_pts - dst_pts, axis=1))
                        
#                         # Dron duruyorsa veya çok yavaşsa, bu kareyi filtreye sokma (marginalize et)
#                         if avg_parallax < MIN_PARALLAX_PIXELS:
#                             is_keyframe = False
#                 # -------------------------------------------------------------
                
#                 # Sadece yeterli hareket varsa sistemi güncelle
#                 if is_keyframe:
#                     cam_state_idx += 1
                    
#                     # 1. Augment MSCKF state (Yeni bir kamera pozisyonu ekle)
#                     msckf.augment_state()
                    
#                     # 2. Track features (Mevcut ORB track algoritman olduğu gibi çalışır)
#                     feature_tracker.track(img, cam_state_idx)
                    
#                     # 3. MSCKF Update with mature tracks
#                     mature_tracks = feature_tracker.get_mature_tracks()
#                     if len(mature_tracks) > 0:
#                         msckf.update(mature_tracks)
                        
#                     # 4. Prune old camera states
#                     msckf.prune_cam_states()



#                 # 5. Zero-Velocity Update (ZUPT)
#                 # Dronun tamamen durduğu anları IMU ile yakala (Bir önceki ZUPT mantığımız)
#                 few_visual = (is_keyframe == False) or (len(feature_tracker.get_mature_tracks()) == 0)
#                 is_still = MSCKF.is_stationary(imu_window, acc_thresh=0.12, gyro_thresh=0.008)
                
#                 if is_still and few_visual:
#                     stationary_count += 1
#                 else:
#                     stationary_count = 0

#                 if stationary_count >= ZUPT_DEBOUNCE:
#                     msckf.zero_velocity_update()
#                     zupt_active = "[ZUPT] "
#                 else:
#                     zupt_active = ""
                
#                 # Visualization (Her karede görselleştirme akmaya devam eder)
#                 frames_processed += 1
#                 elapsed = time.time() - start_time
#                 fps = frames_processed / elapsed if elapsed > 0 else 0
                
#                 if frames_processed % 30 == 0:
#                     gt_p = loader.get_gt_pose(sensor_data.timestamp)
#                     p_viz = viz_R_align @ msckf.state.p + viz_t_align
#                     R_viz = viz_R_align @ quaternion_to_rotation_matrix(msckf.state.q)
#                     q_viz = rotation_matrix_to_quaternion(R_viz)
#                     # p_viz = msckf.state.p
#                     # q_viz = msckf.state.q
                    
#                     viz.update_trajectory(p_viz, q_viz, gt_p_W=gt_p)
#                     num_features = len(feature_tracker.active_tracks)
#                     print(f"{zupt_active}Frame {frames_processed} | FPS: {fps:.1f} | Features: {num_features} | Pos: {p_viz} | GT: {gt_p}")
                
#                 viz.update_camera(img, feature_tracker.active_tracks, fps=fps)
                
#     except KeyboardInterrupt:
#         print("Interrupted by user.")
        
#     print("VIO Loop Finished.")
#     viz.show_interactive()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()



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
    imu_window = []          
    IMU_WINDOW_SIZE = 40
    stationary_count = 0     # consecutive stationary camera frames (debounce)
    ZUPT_DEBOUNCE = 5        # require N consecutive still frames before ZUPT fires
    # Visualization-only alignment: maps VIO frame → GT frame at t=0
    viz_R_align = np.eye(3)
    viz_t_align = np.zeros(3)
    
    # Tüm logları hafızada tutacak liste
    log_records = []
    
    print("Starting VIO Loop...")
    
    try:
        for sensor_data in loader:
            if not initialized:
                if sensor_data.type == 'imu':
                    imu_buffer.append((sensor_data.timestamp, sensor_data.data))
                elif sensor_data.type == 'image':
                    img = sensor_data.data
                    viz.update_camera(img, [], fps=0, initializing=True)
                    
                    success = initializer.add_frame(img, imu_buffer)
                    imu_buffer_snapshot = list(imu_buffer)  # keep for b_a estimate
                    imu_buffer = [] # Clear buffer
                    
                    if success:
                        final_state = initializer.final_states[-1]
                        
                        msckf.state.p = final_state['p'].copy()
                        msckf.state.q = final_state['q'].copy()
                        msckf.state.v = final_state['v'].copy()
                        msckf.state.b_g = initializer.b_g.copy()
                        # Estimate b_a from static period: mean(a) - g_body
                        if len(imu_buffer_snapshot) > 10:
                            acc_vecs = np.array([a for _, (_, a) in imu_buffer_snapshot])
                            mean_a = np.mean(acc_vecs, axis=0)
                            R0 = quaternion_to_rotation_matrix(final_state['q'])
                            g_body = R0.T @ np.array([0, 0, 9.81])
                            msckf.state.b_a = mean_a - g_body
                            print(f"[Init] b_a estimate: {msckf.state.b_a}")
                        else:
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

                # Maintain rolling IMU window for ZUPT
                imu_window.append(sensor_data.data)
                if len(imu_window) > IMU_WINDOW_SIZE:
                    imu_window.pop(0)
                
            elif sensor_data.type == 'image':
                img = sensor_data.data
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                
                # Zamanlama takibi için değişkenleri sıfırla
                t_orb = 0.0
                t_match = 0.0
                t_msckf = 0.0
                
                # -------------------------------------------------------------
                # ORB TABANLI PARALAKS (KEYFRAME) KONTROLÜ
                # -------------------------------------------------------------
                is_keyframe = True
                MIN_PARALLAX_PIXELS = 3.0
                
                # Gelen karedeki ORB özelliklerini çıkar (Zaman Ölçümü: ORB)
                start_orb = time.time()
                kps, des = feature_tracker.orb.detectAndCompute(gray, None)
                t_orb += (time.time() - start_orb) * 1000.0 # ms cinsinden
                
                # Eğer daha öneden kaydedilmiş bir özellik havuzu varsa karşılaştır
                if feature_tracker.prev_des is not None and des is not None and len(des) > 0:
                    # Mevcut kareyi, kabul edilen son ANAHTAR KARE (prev_des) ile eşleştir (Zaman Ölçümü: Match)
                    start_match = time.time()
                    matches = feature_tracker.matcher.match(feature_tracker.prev_des, des)
                    t_match += (time.time() - start_match) * 1000.0
                    
                    if len(matches) > 8:
                        src_pts = np.float32([feature_tracker.prev_kps[m.queryIdx].pt for m in matches])
                        dst_pts = np.float32([kps[m.trainIdx].pt for m in matches])
                        
                        # Eşleşen noktalar arasındaki ortalama piksel mesafesini (parallax) hesapla
                        avg_parallax = np.mean(np.linalg.norm(src_pts - dst_pts, axis=1))
                        
                        # Dron duruyorsa veya çok yavaşsa, bu kareyi filtreye sokma (marginalize et)
                        if avg_parallax < MIN_PARALLAX_PIXELS:
                            is_keyframe = False
                # -------------------------------------------------------------
                
                # Sadece yeterli hareket varsa sistemi güncelle
                if is_keyframe:
                    cam_state_idx += 1
                    
                    # MSCKF işlemlerinin tamamını süre ölçümüne dahil ediyoruz
                    start_msckf = time.time()
                    
                    # 1. Augment MSCKF state (Yeni bir kamera pozisyonu ekle)
                    msckf.augment_state()
                    
                    # 2. Track features (Mevcut ORB track algoritman olduğu gibi çalışır)
                    feature_tracker.track(img, cam_state_idx)
                    
                    # 3. MSCKF Update with mature tracks
                    mature_tracks = feature_tracker.get_mature_tracks()
                    if len(mature_tracks) > 0:
                        msckf.update(mature_tracks)
                        
                    # 4. Prune old camera states
                    msckf.prune_cam_states()
                    
                    t_msckf += (time.time() - start_msckf) * 1000.0

                # 5. Zero-Velocity Update (ZUPT)
                # Dronun tamamen durduğu anları IMU ile yakala (Bir önceki ZUPT mantığımız)
                few_visual = (is_keyframe == False) or (len(feature_tracker.get_mature_tracks()) == 0)
                is_still = MSCKF.is_stationary(imu_window, acc_thresh=0.12, gyro_thresh=0.008)
                
                if is_still and few_visual:
                    stationary_count += 1
                else:
                    stationary_count = 0

                if stationary_count >= ZUPT_DEBOUNCE:
                    start_zupt = time.time()
                    msckf.zero_velocity_update()
                    t_msckf += (time.time() - start_zupt) * 1000.0
                    zupt_active = "[ZUPT] "
                else:
                    zupt_active = ""
                
                # Visualization (Her karede görselleştirme akmaya devam eder)
                frames_processed += 1
                elapsed = time.time() - start_time
                fps = frames_processed / elapsed if elapsed > 0 else 0
                
                if frames_processed % 30 == 0:
                    gt_p = loader.get_gt_pose(sensor_data.timestamp)
                    p_viz = viz_R_align @ msckf.state.p + viz_t_align
                    R_viz = viz_R_align @ quaternion_to_rotation_matrix(msckf.state.q)
                    q_viz = rotation_matrix_to_quaternion(R_viz)
                    # p_viz = msckf.state.p
                    # q_viz = msckf.state.q
                    
                    viz.update_trajectory(p_viz, q_viz, gt_p_W=gt_p)
                    num_features = len(feature_tracker.active_tracks)
                    
                    # Formata uygun log stringlerini oluşturma
                    pos_str = f"[{p_viz[0]:.8f} {p_viz[1]:.8f} {p_viz[2]:.8f}]"
                    gt_str = f"[{gt_p[0]:.6f} {gt_p[1]:.6f} {gt_p[2]:.6f}]" if gt_p is not None else "[N/A]"
                    
                    line1 = f"{zupt_active}Frame {frames_processed} | FPS: {fps:.1f} | Features: {num_features} | Pos: {pos_str} | GT: {gt_str}"
                    line2 = f"   [Zamanlar] ORB: {t_orb:.1f}ms | Match: {t_match:.1f}ms | MSCKF: {t_msckf:.1f}ms"
                    
                    # Terminale bas
                    print(line1)
                    print(line2)
                    
                    # Listeye kaydet
                    log_records.append(line1)
                    log_records.append(line2)
                
                viz.update_camera(img, feature_tracker.active_tracks, fps=fps)
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
        
    print("VIO Loop Finished.")
    
    # ── LOGLARI TXT DOSYASINA YAZMA ──────────────────────────
    if log_records:
        output_filename = "vio_estimation_logs.txt"
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(log_records) + "\n")
            print(f"\n[Başarılı] Tüm log kayıtları '{output_filename}' dosyasına kaydedildi.")
        except Exception as e:
            print(f"\n[Hata] Loglar dosyaya yazılamadı: {e}")
    else:
        print("\n[Uyarı] Kaydedilecek herhangi bir log kaydı oluşmadı.")
    # ─────────────────────────────────────────────────────────

    viz.show_interactive()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()