"""
3D Visualizer using Open3D – single interactive window.

During estimation:
  - A non-blocking Open3D window stays open and updates every N calls.
  - You can ALREADY orbit/pan/zoom with the mouse during estimation.

After estimation:
  - The same window enters a blocking event loop so you can freely navigate.
  - No window rebuild, no data loss.

Controls (during and after estimation)
---------------------------------------
  Left-drag        : orbit
  Right-drag/Scroll: zoom
  Middle-drag      : pan
  R                : reset view
  Q / Esc          : close
"""

import threading
import cv2
import time
import numpy as np

try:
    import open3d as o3d
    OPEN3D_OK = True
except ImportError:
    OPEN3D_OK = False
    print("[Visualizer3D] open3d not found – running without visualization.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_axes(size=0.4):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def _lineset(pts, color):
    """Build a LineSet connecting consecutive points."""
    arr = np.array(pts, dtype=np.float64)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(arr),
        lines=o3d.utility.Vector2iVector([[i, i+1] for i in range(len(arr)-1)])
    )
    ls.paint_uniform_color(color)
    return ls


def _sphere(center, radius=0.07, color=(1.0, 0.45, 0.0)):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    s.translate(np.asarray(center, dtype=np.float64))
    s.paint_uniform_color(list(color))
    s.compute_vertex_normals()
    return s


# ── main class ────────────────────────────────────────────────────────────────

class Visualizer3D:
    """
    Single Open3D window that stays alive from estimation start to finish.

    Parameters
    ----------
    live_update_every : int
        How many `update_trajectory` calls between window refreshes.
        Smaller = smoother live view but slightly more CPU usage.
    """

    def __init__(self, live_update_every: int = 5):
        self._vio_pts: list = []
        self._gt_pts:  list = []
        self._lock = threading.Lock()
        self._call_count = 0
        self._every = live_update_every

        self._vis = None
        self._geoms: dict = {}
        self._alive = False

        if OPEN3D_OK:
            self._create_window()

    # ── window creation ───────────────────────────────────────────────────────

    def _create_window(self):
        try:
            self._vis = o3d.visualization.Visualizer()
            self._vis.create_window(
                window_name="VIO Trajectory  |  Blue = Estimated   Green = Ground Truth",
                width=1280, height=800,
                visible=True,
            )

            opt = self._vis.get_render_option()
            opt.background_color = np.array([0.06, 0.06, 0.10])
            opt.point_size       = 2.0
            opt.line_width       = 3.0

            # Static geometry: world-origin axes
            axes = _make_axes(0.4)
            self._vis.add_geometry(axes)

            # Dynamic geometries (start empty)
            vio_ls = o3d.geometry.LineSet()
            gt_ls  = o3d.geometry.LineSet()
            head   = _sphere([0, 0, 0], radius=0.07)

            self._vis.add_geometry(vio_ls)
            self._vis.add_geometry(gt_ls)
            self._vis.add_geometry(head)

            self._geoms = {"vio": vio_ls, "gt": gt_ls, "head": head}
            self._alive = True

            # Initial pump so window appears immediately
            self._vis.poll_events()
            self._vis.update_renderer()

        except Exception as exc:
            print(f"[Visualizer3D] Could not create window: {exc}")
            self._alive = False

    # ── internal refresh ──────────────────────────────────────────────────────

    def _refresh(self, reset_view: bool = False):
        """Push latest trajectory data into the window and repaint."""
        if not self._alive:
            return

        with self._lock:
            vio_pts = list(self._vio_pts)
            gt_pts  = list(self._gt_pts)

        try:
            # ── VIO line ──
            if len(vio_pts) >= 2:
                vio_arr = np.array(vio_pts, dtype=np.float64)
                self._geoms["vio"].points = o3d.utility.Vector3dVector(vio_arr)
                self._geoms["vio"].lines  = o3d.utility.Vector2iVector(
                    [[i, i+1] for i in range(len(vio_arr)-1)]
                )
                self._geoms["vio"].paint_uniform_color([0.2, 0.55, 1.0])
                self._vis.update_geometry(self._geoms["vio"])

                # Rebuild head sphere at current position
                old_head = self._geoms["head"]
                new_head = _sphere(vio_arr[-1], radius=0.07, color=(1.0, 0.45, 0.0))
                self._vis.remove_geometry(old_head, reset_bounding_box=False)
                self._vis.add_geometry(new_head, reset_bounding_box=False)
                self._geoms["head"] = new_head

            # ── GT line ──
            if len(gt_pts) >= 2:
                gt_arr = np.array(gt_pts, dtype=np.float64)
                self._geoms["gt"].points = o3d.utility.Vector3dVector(gt_arr)
                self._geoms["gt"].lines  = o3d.utility.Vector2iVector(
                    [[i, i+1] for i in range(len(gt_arr)-1)]
                )
                self._geoms["gt"].paint_uniform_color([0.2, 1.0, 0.45])
                self._vis.update_geometry(self._geoms["gt"])

            if reset_view:
                self._vis.reset_view_point(True)

            if not self._vis.poll_events():
                self._alive = False
                return

            self._vis.update_renderer()

        except Exception as exc:
            print(f"[Visualizer3D] Refresh error: {exc}")
            self._alive = False

    # ── camera feed window ─────────────────────────────────────────────────────

    def update_camera(self, img, tracks, fps: float = 0.0, initializing: bool = False):
        """
        Show camera image with overlaid feature tracks in a separate OpenCV window.

        Parameters
        ----------
        img          : np.ndarray  – BGR or grayscale frame from the dataset
        tracks       : list        – list of FeatureTrack objects (or empty list)
        fps          : float       – current FPS to display in corner
        initializing : bool        – show 'Initializing…' overlay instead of feature count
        """
        if img is None:
            return

        # Ensure colour so we can draw coloured circles
        if len(img.shape) == 2 or img.shape[2] == 1:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()

        h, w = vis.shape[:2]

        # ── draw feature points ───────────────────────────────────────────────
        for track in tracks:
            kp = getattr(track, 'keypoint', None)
            if kp is None:
                continue
            x, y = int(kp.pt[0]), int(kp.pt[1])
            age = getattr(track, 'cam_state_idx', 0) - getattr(track, 'first_frame_idx', 0)
            # colour shifts green→yellow→red as track ages
            r = min(255, age * 20)
            g = max(0, 255 - age * 15)
            color = (0, g, r)          # BGR
            cv2.circle(vis, (x, y), 4, color, -1)
            cv2.circle(vis, (x, y), 5, (255, 255, 255), 1)  # white ring

        # ── HUD overlay ───────────────────────────────────────────────────────
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

        if initializing:
            label = "Initializing VIO..."
            cv2.putText(vis, label, (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 200, 255), 2, cv2.LINE_AA)
        else:
            n_feat = len(tracks)
            label = f"Features: {n_feat}   FPS: {fps:.1f}"
            cv2.putText(vis, label, (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 2, cv2.LINE_AA)

        cv2.imshow("VIO Camera Feed", vis)
        cv2.waitKey(1)   # non-blocking – 1 ms

    # ── public API ────────────────────────────────────────────────────────────

    def update_trajectory(self, p_W, q_W=None, gt_p_W=None):
        """
        Call every frame from the VIO loop.

        Parameters
        ----------
        p_W    : array-like (3,)  – estimated position in world frame
        q_W    : ignored (kept for drop-in compatibility with visualizer.py)
        gt_p_W : array-like (3,) or None – ground-truth position
        """
        if not OPEN3D_OK:
            return

        with self._lock:
            self._vio_pts.append(list(p_W))
            if gt_p_W is not None:
                self._gt_pts.append(list(gt_p_W))

        self._call_count += 1
        if self._call_count % self._every == 0:
            self._refresh(reset_view=(self._call_count == self._every))

    def show_interactive(self):
        """
        Call after the VIO loop finishes.
        Blocks until the user closes the window – no data is lost or rebuilt.
        The window is already interactive during estimation; this just keeps it open.
        """
        if not OPEN3D_OK:
            return

        if not self._alive:
            print("[Visualizer3D] Window was already closed.")
            return

        # Final refresh so the last few frames are visible
        self._refresh()

        print("\n[Visualizer3D] Estimation complete – window is now in free-navigation mode.")
        print("  Left-drag: orbit  |  Scroll/Right-drag: zoom  |  Middle-drag: pan")
        print("  R: reset view  |  Q / Esc: close\n")

        # Blocking event loop – keeps window alive until user closes it
        while self._alive:
            if not self._vis.poll_events():
                break
            self._vis.update_renderer()
