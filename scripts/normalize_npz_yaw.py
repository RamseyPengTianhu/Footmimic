#!/usr/bin/env python3
"""Normalize the yaw of an existing .npz motion file.

Rotates the entire trajectory so that the pelvis at frame 0 faces -Y
(the convention used by the soccer training pipeline).

Usage:
    python scripts/normalize_npz_yaw.py motions/Video/bad_file.npz
    python scripts/normalize_npz_yaw.py motions/Video/bad_file.npz --output motions/Video/fixed.npz
    python scripts/normalize_npz_yaw.py motions/Video/bad_file.npz --inplace
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation


def extract_yaw(quat_wxyz: np.ndarray) -> float:
    """Extract yaw (Z-rotation) from a WXYZ quaternion."""
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # scipy uses xyzw
    euler = r.as_euler("ZYX")
    return euler[0]


def rotate_trajectory(data: dict, yaw_correction: float) -> dict:
    """Rotate all body positions and quaternions by yaw_correction around Z."""
    cos_y = np.cos(yaw_correction)
    sin_y = np.sin(yaw_correction)
    
    # 2D rotation matrix for XY
    rot2d = np.array([[cos_y, -sin_y],
                      [sin_y,  cos_y]])
    
    # Correction quaternion (XYZW for scipy)
    r_corr = Rotation.from_euler("Z", yaw_correction)
    
    result = dict(data)
    
    # Rotate body_pos_w: (T, N_bodies, 3)
    if "body_pos_w" in data:
        bp = data["body_pos_w"].copy()
        # Rotate XY around origin
        xy = bp[..., :2]  # (T, N, 2)
        orig_shape = xy.shape
        xy_flat = xy.reshape(-1, 2)
        xy_rot = (rot2d @ xy_flat.T).T
        bp[..., :2] = xy_rot.reshape(orig_shape)
        result["body_pos_w"] = bp
    
    # Rotate body_quat_w: (T, N_bodies, 4) in WXYZ
    if "body_quat_w" in data:
        bq = data["body_quat_w"].copy()
        T, N, _ = bq.shape
        for t in range(T):
            for n in range(N):
                q_wxyz = bq[t, n]
                r_orig = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                r_new = r_corr * r_orig
                q_xyzw = r_new.as_quat()
                bq[t, n] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # back to WXYZ
        result["body_quat_w"] = bq
    
    # Rotate body_lin_vel_w: (T, N_bodies, 3)
    if "body_lin_vel_w" in data:
        blv = data["body_lin_vel_w"].copy()
        xy = blv[..., :2]
        orig_shape = xy.shape
        xy_flat = xy.reshape(-1, 2)
        xy_rot = (rot2d @ xy_flat.T).T
        blv[..., :2] = xy_rot.reshape(orig_shape)
        result["body_lin_vel_w"] = blv
    
    # Rotate body_ang_vel_w: (T, N_bodies, 3)
    if "body_ang_vel_w" in data:
        bav = data["body_ang_vel_w"].copy()
        xy = bav[..., :2]
        orig_shape = xy.shape
        xy_flat = xy.reshape(-1, 2)
        xy_rot = (rot2d @ xy_flat.T).T
        bav[..., :2] = xy_rot.reshape(orig_shape)
        result["body_ang_vel_w"] = bav
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Normalize yaw of .npz motion file")
    parser.add_argument("input", help="Path to input .npz file")
    parser.add_argument("--output", help="Path to output .npz file (default: <input>_normalized.npz)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file")
    args = parser.parse_args()
    
    data = dict(np.load(args.input, allow_pickle=True))
    
    # Get pelvis quaternion at the LAST frame (kicking orientation)
    num_frames = data["body_quat_w"].shape[0]
    pelvis_quat_ref = data["body_quat_w"][num_frames - 1, 0]  # WXYZ
    current_yaw = extract_yaw(pelvis_quat_ref)
    
    # Target: face -Y direction = yaw of -π/2
    target_yaw = -np.pi / 2
    yaw_correction = target_yaw - current_yaw
    
    print(f"File: {args.input}")
    print(f"  Current pelvis yaw at LAST frame: {np.degrees(current_yaw):.1f}°")
    print(f"  Target yaw: {np.degrees(target_yaw):.1f}° (-Y direction)")
    print(f"  Correction: {np.degrees(yaw_correction):.1f}°")
    
    if abs(yaw_correction) < np.radians(5):
        print("  Already aligned (< 5° off). Skipping.")
        return
    
    result = rotate_trajectory(data, yaw_correction)
    
    # Verify
    new_yaw = extract_yaw(result["body_quat_w"][num_frames - 1, 0])
    print(f"  New pelvis yaw at LAST frame: {np.degrees(new_yaw):.1f}°")
    
    if args.inplace:
        out_path = args.input
    elif args.output:
        out_path = args.output
    else:
        out_path = args.input.replace(".npz", "_normalized.npz")
    
    np.savez(out_path, **result)
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
