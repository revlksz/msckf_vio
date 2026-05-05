import numpy as np
import cv2

class Camera:
    def __init__(self, config):
        self.K = config.K
        self.D = config.D
        
    def project(self, pts_3d):
        """
        Projects 3D points in camera frame to pixel coordinates.
        pts_3d: Nx3 array
        """
        if len(pts_3d) == 0:
            return np.empty((0, 2))
        pts_2d, _ = cv2.projectPoints(pts_3d, np.zeros(3), np.zeros(3), self.K, self.D)
        return pts_2d.reshape(-1, 2)
        
    def unproject(self, pts_2d):
        """
        Unprojects pixel coordinates to normalized camera coordinates.
        pts_2d: Nx2 array
        """
        if len(pts_2d) == 0:
            return np.empty((0, 2))
        
        pts_2d = pts_2d.reshape(-1, 1, 2).astype(np.float32)
        norm_pts = cv2.undistortPoints(pts_2d, self.K, self.D)
        return norm_pts.reshape(-1, 2)
