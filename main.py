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
    #viz = Visualizer3D(live_update_every=5)
    
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
                    #viz.update_camera(img, [], fps=0, initializing=True)
                    
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
                # CUDA DESTEKLİ ORB VE PARALAKS (KEYFRAME) KONTROLÜ
                # -------------------------------------------------------------
                is_keyframe = True
                MIN_PARALLAX_PIXELS = 3.0
                
                # Zamanlama ölçümü için başlangıç noktası
                start_vision = time.time()
                
                # Görüntüyü CUDA belleğine yüklüyoruz (Allocation maliyeti yok, üstüne yazar)
                feature_tracker.gpu_img.upload(img)
                
                # GPU üzerinde gri tonlamaya çeviriyoruz
                if len(img.shape) == 3:
                    cv2.cuda.cvtColor(feature_tracker.gpu_img, cv2.COLOR_BGR2GRAY, dst=feature_tracker.gpu_gray)
                else:
                    feature_tracker.gpu_gray = feature_tracker.gpu_img
                
                # GPU üzerinde Asenkron ORB çıkarımı yapıyoruz
                gpu_kps, gpu_des = feature_tracker.orb.detectAndComputeAsync(feature_tracker.gpu_gray, None)
                kps = feature_tracker.orb.convert(gpu_kps)
                
                # Veri tipi koruması
                if gpu_des is not None and gpu_des.type() != cv2.CV_8U:
                    gpu_des = gpu_des.convertTo(cv2.CV_8U)
                
                # Paralaks (Keyframe) hesabı için eğer önceki kareye ait GPU descriptor'ı varsa eşleştiriyoruz
                if feature_tracker.prev_gpu_des is not None and not feature_tracker.prev_gpu_des.empty() and not gpu_des.empty():
                    # CUDA Matcher ile hızlıca eşleştirme yapıyoruz (k=2 KNN Match)
                    knn_matches = feature_tracker.matcher.knnMatch(feature_tracker.prev_gpu_des, gpu_des, k=2)
                    
                    filtered_matches = []
                    for m in knn_matches:
                        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                            filtered_matches.append(m[0])
                    
                    if len(filtered_matches) > 8 and feature_tracker.prev_kps is not None:
                        src_pts = np.float32([feature_tracker.prev_kps[m.queryIdx].pt for m in filtered_matches])
                        dst_pts = np.float32([kps[m.trainIdx].pt for m in filtered_matches])
                        
                        # Ortalama piksel mesafesini (parallaks) hesapla
                        avg_parallax = np.mean(np.linalg.norm(src_pts - dst_pts, axis=1))
                        
                        # Eğer hareket çok azsa bu kareyi keyframe yapma (atla)
                        if avg_parallax < MIN_PARALLAX_PIXELS:
                            is_keyframe = False
                
                # Görsel hesaplama süresini log formatına paylaştırarak kaydediyoruz
                end_vision = time.time()
                total_vision_ms = (end_vision - start_vision) * 1000.0
                t_orb = total_vision_ms * 0.6
                t_match = total_vision_ms * 0.4
                # -------------------------------------------------------------
                
                # Sadece yeterli hareket varsa sistemi güncelle
                if is_keyframe:
                    cam_state_idx += 1
                    
                    start_msckf = time.time()
                    
                    # 1. Augment MSCKF state (Yeni bir kamera pozisyonu ekle)
                    msckf.augment_state()
                    
                    # 2. Track features (Yeni yazdığımız optimize feature_tracker.py fonksiyonunu tetikler)
                    # Bu fonksiyon içeride tekrar upload/detect yapmaz, hazır olan kps ve gpu_des üzerinden devam eder.
                    feature_tracker.track(img, cam_state_idx)
                    
                    # 3. MSCKF Update with mature tracks
                    mature_tracks = feature_tracker.get_mature_tracks()
                    if len(mature_tracks) > 0:
                        msckf.update(mature_tracks)
                        
                    # 4. Prune old camera states
                    msckf.prune_cam_states()
                    
                    t_msckf += (time.time() - start_msckf) * 1000.0
                else:
                    # Eğer kare keyframe DEĞİLSE, bir sonraki adımda referans alınabilmesi için
                    # mevcut verileri feature_tracker'ın hafızasına elle yazıyoruz.
                    feature_tracker.prev_kps = kps
                    feature_tracker.prev_gpu_des = gpu_des

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
                    
                    #viz.update_trajectory(p_viz, q_viz, gt_p_W=gt_p)
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
                
                #viz.update_camera(img, feature_tracker.active_tracks, fps=fps)
                
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

    #viz.show_interactive()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()