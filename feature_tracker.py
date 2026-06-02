import numpy as np
import cv2

class FeatureTrack:
    def __init__(self, feature_id, first_frame_idx):
        self.feature_id = feature_id
        self.first_frame_idx = first_frame_idx
        # observations: list of (camera_state_idx, norm_x, norm_y)
        self.observations = []
        self.last_seen_frame = first_frame_idx
        self.keypoint = None # latest cv2.KeyPoint

class FeatureTracker:
    def __init__(self, config, camera):
        self.config = config
        self.camera = camera
        
        # 1. CUDA ORB ve CUDA Matcher Başlatma
        self.orb = cv2.cuda.ORB_create(
            nfeatures=config.orb_nfeatures,
            scaleFactor=config.orb_scaleFactor,
            nlevels=config.orb_nlevels,
            fastThreshold=config.orb_fastThreshold
        )
        # CUDA için Hamming uzaklığı kullanan Brute-Force Matcher
        self.matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
        
        # 2. Sürekli Yeniden Bellek Tahsisini (Allocation) Önlemek İçin GpuMat'leri Önceden Tanımlıyoruz
        self.gpu_img = cv2.cuda_GpuMat()
        self.gpu_gray = cv2.cuda_GpuMat()
        self.prev_gpu_des = cv2.cuda.GpuMat()
        
        self.active_tracks = [] # Currently tracked features
        self.mature_tracks = [] # Features ready for MSCKF update
        self.next_feature_id = 0
        
        self.prev_kps = None
        self.prev_track_map = {} # Maps prev_kps index to FeatureTrack object
        
        self.cam_state_idx = -1 # Matches MSCKF sliding window index
        
    def track(self, img, new_cam_state_idx):
        """
        Processes a new image using Jetson CUDA cores, extracts ORB features, 
        matches with previous frame using Lowe's Ratio Test, and maintains feature tracks.
        """
        if new_cam_state_idx > self.cam_state_idx:
            self.cam_state_idx = new_cam_state_idx
            
        # 3. Görüntüyü CPU'dan GPU Belleğine Yükleme (Upload)
        # Var olan bellek alanını (self.gpu_img) ezerek allocation yükünü sıfırlıyoruz.
        self.gpu_img.upload(img)
        
        # 4. GPU Üzerinde Gri Tonlamaya Dönüştürme
        if len(img.shape) == 3:
            cv2.cuda.cvtColor(self.gpu_img, cv2.COLOR_BGR2GRAY, dst=self.gpu_gray)
        else:
            self.gpu_gray = self.gpu_img
            
        # 5. GPU Üzerinde Asenkron ORB Tespiti ve Descriptor Hesaplama
        # gpu_kps: CUDA formatında anahtar noktalar, gpu_des: GpuMat formatında tanımlayıcılar
        gpu_kps, gpu_des = self.orb.detectAndComputeAsync(self.gpu_gray, None)
        
        # Geometrik hesaplamalar ve takip mantığı için Keypoint verisini CPU'ya indiriyoruz
        kps = self.orb.convert(gpu_kps)
        
        if kps is None or len(kps) == 0 or gpu_des.empty():
            self.prev_kps = None
            self.prev_gpu_des.upload(np.empty((0, 0), dtype=np.uint8)) # Güvenli temizleme
            self.prev_track_map = {}
            return
            
        # Veri Tipi Ön Engellemesi: Descriptor'ların CV_8U tipinde olduğundan emin oluyoruz
        if gpu_des.type() != cv2.CV_8U:
            gpu_des = gpu_des.convertTo(cv2.CV_8U)
            
        pts_2d = np.array([kp.pt for kp in kps])
        norm_pts = self.camera.unproject(pts_2d)
        
        curr_track_map = {} # Maps current kps index to FeatureTrack
        
        # 6. GPU Üzerinde Özellik Eşleştirme (Feature Matching)
        if self.prev_gpu_des is not None and not self.prev_gpu_des.empty():
            # CUDA Matcher ile KnnMatch (k=2) yapıyoruz
            knn_matches = self.matcher.knnMatch(self.prev_gpu_des, gpu_des, k=2)
            
            # CUDA'dan gelen ham eşleşmeleri ayıklamak ve Lowe's Ratio Test uygulamak
            filtered_matches = []
            for m in knn_matches:
                if len(m) == 2:
                    # m[0] -> en yakın komşu, m[1] -> ikinci en yakın komşu
                    if m[0].distance < 0.75 * m[1].distance:
                        filtered_matches.append(m[0])
            
            # Geometrik RANSAC doğrulaması için yeterli eşleşme var mı kontrolü
            if len(filtered_matches) >= 8:
                src_pts = np.float32([self.prev_kps[m.queryIdx].pt for m in filtered_matches])
                dst_pts = np.float32([kps[m.trainIdx].pt for m in filtered_matches])
                
                # RANSAC işlemi CPU üzerinde çok az nokta ile yapıldığı için darboğaz yaratmaz
                E, mask = cv2.findEssentialMat(
                    src_pts, dst_pts, self.camera.K, 
                    method=cv2.RANSAC, prob=0.999, threshold=1.0
                )
                if mask is not None:
                    valid_matches = [m for i, m in enumerate(filtered_matches) if mask[i][0] == 1]
                else:
                    valid_matches = []
            else:
                valid_matches = filtered_matches
                
            # Eşleşen izlekleri (tracks) güncelleme
            for match in valid_matches:
                prev_idx = match.queryIdx
                curr_idx = match.trainIdx
                
                if prev_idx in self.prev_track_map:
                    track = self.prev_track_map[prev_idx]
                    track.observations.append((self.cam_state_idx, norm_pts[curr_idx][0], norm_pts[curr_idx][1]))
                    track.last_seen_frame = self.cam_state_idx
                    track.keypoint = kps[curr_idx]
                    curr_track_map[curr_idx] = track
                    
        # 7. Eşleşmeyen Keypoint'ler İçin Yeni İzlekler (Tracks) Başlatma
        for i in range(len(kps)):
            if i not in curr_track_map:
                new_track = FeatureTrack(self.next_feature_id, self.cam_state_idx)
                self.next_feature_id += 1
                new_track.observations.append((self.cam_state_idx, norm_pts[i][0], norm_pts[i][1]))
                new_track.keypoint = kps[i]
                self.active_tracks.append(new_track)
                curr_track_map[i] = new_track
                
        # 8. Aktif İzlekleri MSCKF İçin Olgunluk Durumuna Göre Filtreleme
        surviving_tracks = []
        for track in self.active_tracks:
            frames_unseen = self.cam_state_idx - track.last_seen_frame
            track_length = self.cam_state_idx - track.first_frame_idx + 1
            
            # İzlek kaybedildiyse veya maksimum pencere sınırına ulaştıysa olgunlaşmıştır
            if frames_unseen > 0 or track_length >= self.config.max_track_horizon:
                if len(track.observations) >= self.config.min_track_length:
                    obs_first = np.array(track.observations[0][1:3])
                    obs_last = np.array(track.observations[-1][1:3])
                    parallax = np.linalg.norm(obs_first - obs_last)
                    
                    if parallax > self.config.min_parallax:
                        self.mature_tracks.append(track)
                        continue # Aktif listeden çıkarılıyor (Mature listesine taşındı)
            
            if frames_unseen == 0:
                surviving_tracks.append(track)
                
        # Bir sonraki kare için durum güncellemeleri
        self.active_tracks = surviving_tracks
        self.prev_kps = kps
        self.prev_gpu_des = gpu_des # Bir sonraki adımda kullanmak üzere GPU matrisini saklıyoruz
        self.prev_track_map = curr_track_map
        
    def get_mature_tracks(self):
        """
        Returns the list of mature tracks ready for MSCKF update and clears the internal list.
        """
        tracks = self.mature_tracks
        self.mature_tracks = []
        return tracks