import numpy as np
import cv2
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        # 3D Trajectory Setup
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("VIO Trajectory")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        
        # Set isometric aspect ratio if supported
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass
        
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []
        self.line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Estimation')
        
        self.gt_x = []
        self.gt_y = []
        self.gt_z = []
        self.gt_line, = self.ax.plot([], [], [], 'g--', linewidth=2, label='Ground Truth')
        
        self.ax.legend()
        
        self.follow_mode = True # Set to False to stop auto-updating limits
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        # Disable auto-scaling when user clicks on the plot to interact
        self.follow_mode = False
        
    def update_trajectory(self, p_W, q_W, gt_p_W=None):
        self.trajectory_x.append(p_W[0])
        self.trajectory_y.append(p_W[1])
        self.trajectory_z.append(p_W[2])
        
        if gt_p_W is not None:
            self.gt_x.append(gt_p_W[0])
            self.gt_y.append(gt_p_W[1])
            self.gt_z.append(gt_p_W[2])
        
        # Keep last 5000 points
        if len(self.trajectory_x) > 5000:
            self.trajectory_x.pop(0)
            self.trajectory_y.pop(0)
            self.trajectory_z.pop(0)
            
        if len(self.gt_x) > 5000:
            self.gt_x.pop(0)
            self.gt_y.pop(0)
            self.gt_z.pop(0)
            
        self.line.set_data(self.trajectory_x, self.trajectory_y)
        self.line.set_3d_properties(self.trajectory_z)
        
        if gt_p_W is not None:
            self.gt_line.set_data(self.gt_x, self.gt_y)
            self.gt_line.set_3d_properties(self.gt_z)
        
        if self.follow_mode:
            # Adjust limits dynamically
            all_x = self.trajectory_x + self.gt_x
            all_y = self.trajectory_y + self.gt_y
            all_z = self.trajectory_z + self.gt_z
            
            if len(all_x) > 0:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                min_z, max_z = min(all_z), max(all_z)
                
                mid_x = (min_x + max_x) / 2.0
                mid_y = (min_y + max_y) / 2.0
                mid_z = (min_z + max_z) / 2.0
                
                max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2.0
                margin = 0.5
                
                self.ax.set_xlim(mid_x - max_range - margin, mid_x + max_range + margin)
                self.ax.set_ylim(mid_y - max_range - margin, mid_y + max_range + margin)
                self.ax.set_zlim(mid_z - max_range - margin, mid_z + max_range + margin)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def update_camera(self, img, active_tracks, fps=0, initializing=False):
        annotated_img = img.copy()
        if len(annotated_img.shape) == 2:
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_GRAY2BGR)
            
        # Draw 4x3 grid
        h, w = annotated_img.shape[:2]
        cols, rows = 4, 3
        for i in range(1, cols):
            x = int(i * w / cols)
            cv2.line(annotated_img, (x, 0), (x, h), (255, 0, 0), 1)
        for i in range(1, rows):
            y = int(i * h / rows)
            cv2.line(annotated_img, (0, y), (w, y), (255, 0, 0), 1)
            
        # Draw features
        for track in active_tracks:
            if track.keypoint is not None:
                pt = (int(track.keypoint.pt[0]), int(track.keypoint.pt[1]))
                cv2.circle(annotated_img, pt, 3, (0, 255, 0), -1)
                
        # Info text
        if initializing:
            cv2.putText(annotated_img, "Initializing... Please wait", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(annotated_img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_img, f"Active Tracks: {len(active_tracks)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_img, "Click 3D plot to rotate", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("VIO Camera", annotated_img)
        cv2.waitKey(1)
