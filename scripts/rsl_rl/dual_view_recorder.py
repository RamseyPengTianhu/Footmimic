"""Dual-view video recorder for Isaac Lab environments.

Creates two camera views (front and back) that FOLLOW the robot,
renders them side-by-side into a single MP4 video.
Works in headless mode over SSH.

Usage in play_multi.py:
    recorder = DualViewRecorder(env, output_dir, ...)
    recorder.setup()
    for step in range(N):
        ...env.step(actions)...
        recorder.capture()
    recorder.save()
"""

from __future__ import annotations

import os
import datetime
import numpy as np

# Lazy imports — only available inside Isaac Sim runtime.
_rep = None
_UsdGeom = None
_Gf = None


def _lazy_imports():
    """Import Omniverse modules lazily (they need the runtime)."""
    global _rep, _UsdGeom, _Gf
    if _rep is None:
        import omni.replicator.core as rep
        from pxr import UsdGeom, Gf

        _rep = rep
        _UsdGeom = UsdGeom
        _Gf = Gf


def _get_stage():
    import omni.usd
    return omni.usd.get_context().get_stage()


class DualViewRecorder:
    """Records a split-screen (front + back) video from dual cameras.

    Both cameras track the robot each frame by reading its root position
    and offsetting the camera accordingly.

    Args:
        env: The Isaac Lab environment (unwrapped ManagerBasedRLEnv).
        output_dir: Directory to write the MP4 file.
        resolution: (width, height) per single camera view.
        front_offset: Camera offset from robot for the front view (dx, dy, dz).
        back_offset: Camera offset from robot for the back view (dx, dy, dz).
        lookat_offset: Vertical offset for the look-at point above robot root.
        fps: Frames per second for the output video.
    """

    def __init__(
        self,
        env,
        output_dir: str,
        resolution: tuple[int, int] = (960, 540),
        front_offset: tuple[float, float, float] = (4.0, 3.0, 2.5),
        back_offset: tuple[float, float, float] = (-4.0, -3.0, 2.5),
        lookat_offset: float = 0.5,
        fps: int = 30,
        path_tracing: bool = False,
        spp: int = 32,
    ):
        _lazy_imports()
        self._env = env
        self._output_dir = output_dir
        self._resolution = resolution
        self._front_offset = front_offset
        self._back_offset = back_offset
        self._lookat_offset = lookat_offset
        self._fps = fps
        self._frames: list[np.ndarray] = []
        self._front_annotator = None
        self._back_annotator = None
        self._front_cam = None
        self._back_cam = None
        self._path_tracing = path_tracing
        self._spp = spp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self):
        """Create two camera prims, render products, and RGB annotators."""
        if self._path_tracing:
            self._enable_path_tracing(self._spp)

        stage = _get_stage()

        # -- Create cameras at initial position --
        robot_pos = self._get_robot_pos()

        self._front_cam = self._create_camera(
            stage, "/World/DualView/FrontCamera",
            eye=self._offset_pos(robot_pos, self._front_offset),
            target=(robot_pos[0], robot_pos[1], robot_pos[2] + self._lookat_offset),
        )
        front_rp = _rep.create.render_product(
            str(self._front_cam.GetPath()), self._resolution
        )
        self._front_annotator = _rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._front_annotator.attach([front_rp])

        self._back_cam = self._create_camera(
            stage, "/World/DualView/BackCamera",
            eye=self._offset_pos(robot_pos, self._back_offset),
            target=(robot_pos[0], robot_pos[1], robot_pos[2] + self._lookat_offset),
        )
        back_rp = _rep.create.render_product(
            str(self._back_cam.GetPath()), self._resolution
        )
        self._back_annotator = _rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._back_annotator.attach([back_rp])

        os.makedirs(self._output_dir, exist_ok=True)
        print(f"[DualViewRecorder] Setup complete. Resolution per view: {self._resolution}")
        print(f"[DualViewRecorder] Front offset={self._front_offset}, Back offset={self._back_offset}")

    def capture(self, overlay_text: str | None = None):
        """Update cameras to follow robot, then capture and stitch one frame.

        Args:
            overlay_text: Optional multi-line text to overlay on the video
                          (e.g. CG phase, timestep, ball speed).
        """
        # 1. Update camera positions to follow robot.
        robot_pos = self._get_robot_pos()
        self._update_camera(
            self._front_cam,
            eye=self._offset_pos(robot_pos, self._front_offset),
            target=(robot_pos[0], robot_pos[1], robot_pos[2] + self._lookat_offset),
        )
        self._update_camera(
            self._back_cam,
            eye=self._offset_pos(robot_pos, self._back_offset),
            target=(robot_pos[0], robot_pos[1], robot_pos[2] + self._lookat_offset),
        )

        # 2. Force a render so annotators get fresh data.
        self._env.sim.render()

        # 3. Read RGB from both cameras.
        front_rgb = self._read_annotator(self._front_annotator)
        back_rgb = self._read_annotator(self._back_annotator)

        # 4. Side-by-side: [front | back]
        combined = np.concatenate([front_rgb, back_rgb], axis=1)

        # 5. Overlay text HUD if provided.
        if overlay_text:
            combined = self._draw_text(combined, overlay_text)

        self._frames.append(combined)

    def save(self, filename: str | None = None) -> str:
        """Write collected frames as an MP4 video. Returns the file path."""
        if not self._frames:
            print("[DualViewRecorder] No frames captured — skipping save.")
            return ""

        if filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dual_view_{ts}.mp4"

        filepath = os.path.join(self._output_dir, filename)
        import imageio.v2 as iio

        writer = iio.get_writer(filepath, fps=self._fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")
        for frame in self._frames:
            writer.append_data(frame)
        writer.close()

        print(f"[DualViewRecorder] Saved {len(self._frames)} frames → {filepath}")
        self._frames.clear()
        return filepath

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_robot_pos(self) -> tuple[float, float, float]:
        """Get the robot's root XYZ position from the environment."""
        try:
            robot = self._env.scene["robot"]
            pos = robot.data.root_pos_w[0].cpu().numpy()  # env 0
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        except Exception:
            return (0.0, 0.0, 0.0)

    @staticmethod
    def _offset_pos(base, offset):
        return (base[0] + offset[0], base[1] + offset[1], base[2] + offset[2])

    @staticmethod
    def _compute_lookat_matrix(eye, target):
        """Compute a Gf.Matrix4d for a camera looking from eye to target."""
        eye_gf = _Gf.Vec3d(*eye)
        target_gf = _Gf.Vec3d(*target)
        up = _Gf.Vec3d(0, 0, 1)

        fwd = (target_gf - eye_gf).GetNormalized()
        right = (fwd ^ up).GetNormalized()
        new_up = (right ^ fwd).GetNormalized()

        rot = _Gf.Matrix3d()
        rot.SetRow(0, right)
        rot.SetRow(1, new_up)
        rot.SetRow(2, -fwd)

        mat4 = _Gf.Matrix4d()
        mat4.SetRotateOnly(rot)
        mat4.SetTranslateOnly(eye_gf)
        return mat4

    @staticmethod
    def _create_camera(stage, prim_path: str, eye, target):
        """Create a USD Camera prim with look-at transform."""
        cam = _UsdGeom.Camera.Define(stage, prim_path)
        xform = _UsdGeom.Xformable(cam.GetPrim())
        xform.ClearXformOpOrder()

        mat4 = DualViewRecorder._compute_lookat_matrix(eye, target)
        op = xform.AddTransformOp()
        op.Set(mat4)

        cam.GetFocalLengthAttr().Set(18.0)
        cam.GetClippingRangeAttr().Set(_Gf.Vec2f(0.1, 100.0))
        return cam

    @staticmethod
    def _update_camera(cam, eye, target):
        """Update an existing camera's transform to a new look-at position."""
        xform = _UsdGeom.Xformable(cam.GetPrim())
        mat4 = DualViewRecorder._compute_lookat_matrix(eye, target)
        # Get existing transform op and update it.
        ops = xform.GetOrderedXformOps()
        if ops:
            ops[0].Set(mat4)

    @staticmethod
    def _read_annotator(annotator) -> np.ndarray:
        """Read RGB data from a replicator annotator."""
        data = annotator.get_data()
        if data is None or (hasattr(data, 'size') and data.size == 0):
            return np.zeros((540, 960, 3), dtype=np.uint8)
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape)
        return rgb[:, :, :3]

    @staticmethod
    def _draw_text(frame: np.ndarray, text: str) -> np.ndarray:
        """Draw multi-line text overlay on the upper-left of the frame."""
        import cv2
        frame = frame.copy()
        lines = text.strip().split("\n")
        y = 30
        for line in lines:
            # Black outline for readability.
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 3, cv2.LINE_AA)
            # White text.
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28
        return frame

    @staticmethod
    def _enable_path_tracing(spp: int = 32):
        """Switch renderer to interactive path tracing."""
        try:
            import carb
            s = carb.settings.get_settings()
            s.set("/rtx/rendermode", "PathTracing")
            s.set("/rtx/pathtracing/spp", spp)
            s.set("/rtx/pathtracing/totalSpp", spp)
            s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
            print(f"[DualViewRecorder] Path Tracing enabled (SPP={spp})")
        except Exception as e:
            print(f"[DualViewRecorder] WARNING: Could not enable path tracing: {e}")
