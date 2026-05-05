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
        self.orb = cv2.ORB_create(
            nfeatures=config.orb_nfeatures,
            scaleFactor=config.orb_scaleFactor,
            nlevels=config.orb_nlevels,
            fastThreshold=config.orb_fastThreshold
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.active_tracks = [] # Currently tracked features
        self.mature_tracks = [] # Features ready for MSCKF update
        self.next_feature_id = 0
        
        self.prev_kps = None
        self.prev_des = None
        self.prev_track_map = {} # Maps prev_kps index to FeatureTrack object
        
        self.cam_state_idx = -1 # Matches MSCKF sliding window index
        
    def track(self, img, new_cam_state_idx):
        """
        Processes a new image, extracts ORB features, matches with previous,
        and maintains feature tracks.
        new_cam_state_idx: the index of the current camera state in MSCKF window.
        """
        if new_cam_state_idx > self.cam_state_idx:
            self.cam_state_idx = new_cam_state_idx
            
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        kps, des = self.orb.detectAndCompute(gray, None)
        
        if kps is None or len(kps) == 0:
            self.prev_kps = None
            self.prev_des = None
            self.prev_track_map = {}
            return
            
        pts_2d = np.array([kp.pt for kp in kps])
        norm_pts = self.camera.unproject(pts_2d)
        
        curr_track_map = {} # Maps current kps index to FeatureTrack
        
        if self.prev_des is not None and len(self.prev_des) > 0 and len(des) > 0:
            matches = self.matcher.match(self.prev_des, des)
            
            if len(matches) >= 8:
                src_pts = np.float32([self.prev_kps[m.queryIdx].pt for m in matches])
                dst_pts = np.float32([kps[m.trainIdx].pt for m in matches])
                
                E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.camera.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if mask is not None:
                    valid_matches = [m for i, m in enumerate(matches) if mask[i][0] == 1]
                else:
                    valid_matches = []
            else:
                valid_matches = matches
                
            for match in valid_matches:
                prev_idx = match.queryIdx
                curr_idx = match.trainIdx
                
                if prev_idx in self.prev_track_map:
                    track = self.prev_track_map[prev_idx]
                    track.observations.append((self.cam_state_idx, norm_pts[curr_idx][0], norm_pts[curr_idx][1]))
                    track.last_seen_frame = self.cam_state_idx
                    track.keypoint = kps[curr_idx]
                    curr_track_map[curr_idx] = track
                    
        # Initialize new tracks for unmatched keypoints
        for i in range(len(kps)):
            if i not in curr_track_map:
                new_track = FeatureTrack(self.next_feature_id, self.cam_state_idx)
                self.next_feature_id += 1
                new_track.observations.append((self.cam_state_idx, norm_pts[i][0], norm_pts[i][1]))
                new_track.keypoint = kps[i]
                self.active_tracks.append(new_track)
                curr_track_map[i] = new_track
                
        # Check active tracks for maturity
        surviving_tracks = []
        for track in self.active_tracks:
            frames_unseen = self.cam_state_idx - track.last_seen_frame
            track_length = self.cam_state_idx - track.first_frame_idx + 1
            
            # A track is mature if it's lost, OR if it has reached max horizon
            if frames_unseen > 0 or track_length >= self.config.max_track_horizon:
                if len(track.observations) >= self.config.min_track_length:
                    obs_first = np.array(track.observations[0][1:3])
                    obs_last = np.array(track.observations[-1][1:3])
                    parallax = np.linalg.norm(obs_first - obs_last)
                    
                    if parallax > self.config.min_parallax:
                        self.mature_tracks.append(track)
                        continue # Removed from active
            
            if frames_unseen == 0:
                surviving_tracks.append(track)
                
        self.active_tracks = surviving_tracks
        self.prev_kps = kps
        self.prev_des = des
        self.prev_track_map = curr_track_map
        
    def get_mature_tracks(self):
        """
        Returns the list of mature tracks ready for MSCKF update and clears the internal list.
        """
        tracks = self.mature_tracks
        self.mature_tracks = []
        return tracks
