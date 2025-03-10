#!/usr/bin/env python3
"""
Process two RGB-D frames to estimate camera motion using direct photometric alignment.

Usage:
    python process_two_frames.py -i /path/to/data_dir [--debug] [--depth-scale 5000.0]
"""

import os
import sys
import argparse
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path to import from package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from camera_model import CameraModel
from tracker import Tracker
from utils import read_camera_intrinsics, rotation_matrix_to_euler_angles


def load_frames_from_images(input_dir, depth_scale=5000.0):
    """
    Load frames directly from image files.

    Args:
        input_dir: Directory containing the image files
        depth_scale: Scale factor to convert depth image to meters

    Returns:
        dict with 'ref' and 'cur' frames containing grayscale and depth images
    """
    # Expected file patterns
    ref_gray_path = os.path.join(input_dir, "frame_0_gray.png")
    ref_depth_path = os.path.join(input_dir, "frame_0_depth.png")
    cur_gray_path = os.path.join(input_dir, "frame_1_gray.png")
    cur_depth_path = os.path.join(input_dir, "frame_1_depth.png")

    # Check if files exist
    for path in [ref_gray_path, ref_depth_path, cur_gray_path, cur_depth_path]:
        if not os.path.exists(path):
            print(f"Error: Image file not found: {path}")
            sys.exit(1)

    # Load grayscale images
    ref_gray = cv2.imread(ref_gray_path, cv2.IMREAD_GRAYSCALE)
    cur_gray = cv2.imread(cur_gray_path, cv2.IMREAD_GRAYSCALE)

    # Load depth images
    ref_depth_img = cv2.imread(ref_depth_path, cv2.IMREAD_UNCHANGED)
    cur_depth_img = cv2.imread(cur_depth_path, cv2.IMREAD_UNCHANGED)

    # Convert depth to meters
    ref_depth = ref_depth_img.astype(np.float32) / depth_scale
    cur_depth = cur_depth_img.astype(np.float32) / depth_scale

    print(f"Loaded reference frame: {os.path.basename(ref_gray_path)}, {os.path.basename(ref_depth_path)}")
    print(f"Loaded current frame: {os.path.basename(cur_gray_path)}, {os.path.basename(cur_depth_path)}")
    print(f"Image dimensions: {ref_gray.shape}")

    return {
        'ref': {
            'gray': ref_gray,
            'depth': ref_depth,
            'gray_path': ref_gray_path,
            'depth_path': ref_depth_path
        },
        'cur': {
            'gray': cur_gray,
            'depth': cur_depth,
            'gray_path': cur_gray_path,
            'depth_path': cur_depth_path
        }
    }


def load_intrinsics(input_dir, intrinsics_file=None):
    """
    Load intrinsics from a text file or pickle file, prioritizing text files.

    Args:
        input_dir: Directory containing the frames and possibly intrinsics files
        intrinsics_file: Optional path to a text-based intrinsics file

    Returns:
        intrinsics dictionary
    """
    # Priority 1: If a specific intrinsics file is provided via command line, use it
    if intrinsics_file and os.path.exists(intrinsics_file):
        print(f"Reading intrinsics from specified text file: {intrinsics_file}")
        return read_camera_intrinsics(intrinsics_file)

    # Priority 2: Look for intrinsics.txt in the input directory
    intrinsics_txt_path = os.path.join(input_dir, "intrinsics.txt")
    if os.path.exists(intrinsics_txt_path):
        print(f"Reading intrinsics from text file in input directory: {intrinsics_txt_path}")
        return read_camera_intrinsics(intrinsics_txt_path)

    # Priority 3: Look for the original intrinsics file that might have been copied
    # Check for files with 'intrinsics' in their name and .txt extension
    potential_intrinsics_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if 'intrinsics' in f.lower() and f.endswith('.txt') and f != "intrinsics.txt"
    ]

    if potential_intrinsics_files:
        # Use the first one found
        intrinsics_file = potential_intrinsics_files[0]
        print(f"Reading intrinsics from found text file: {intrinsics_file}")
        return read_camera_intrinsics(intrinsics_file)

    # Priority 4: If no intrinsics found, use default values
    print("Warning: No intrinsics files found, using default values")
    # Default TUM RGB-D camera intrinsics
    intrinsics = {
        'fx': 525.0,
        'fy': 525.0,
        'cx': 319.5,
        'cy': 239.5,
        'width': 640,
        'height': 480,
        'K': np.array([
            [525.0, 0, 319.5],
            [0, 525.0, 239.5],
            [0, 0, 1]
        ], dtype=np.float32)
    }

    return intrinsics


def visualize_residuals(gray_ref, gray_cur, residuals, height, width):
    """
    Visualize residuals between two frames.

    Args:
        gray_ref: Reference grayscale image
        gray_cur: Current grayscale image
        residuals: Residuals array
        height: Image height
        width: Image width
    """
    # Create a visualization of the frames side by side
    vis_img = np.hstack((gray_ref, gray_cur))
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    # Reshape residuals to image shape
    residuals_img = np.reshape(residuals, (height, width))

    # Convert residuals to visualization (normalize)
    residuals_viz = np.zeros_like(residuals_img, dtype=np.uint8)
    valid_mask = np.isfinite(residuals_img)

    if np.any(valid_mask):
        residuals_valid = residuals_img[valid_mask]
        min_val, max_val = np.min(residuals_valid), np.max(residuals_valid)

        # Normalize to 0-255
        range_val = max_val - min_val
        if range_val > 0:
            normalized = ((residuals_valid - min_val) / range_val * 255).astype(np.uint8)
            residuals_viz[valid_mask] = normalized

    # Apply color map for better visualization
    residuals_viz = cv2.applyColorMap(residuals_viz, cv2.COLORMAP_JET)

    # Show images
    cv2.imshow("Gray Images", vis_img)
    cv2.imshow("Residuals", residuals_viz)
    cv2.waitKey(0)


def visualize_tracking(gray_ref, depth_ref, gray_cur, pose):
    """
    Visualize tracking result.

    Args:
        gray_ref: Reference grayscale image
        depth_ref: Reference depth image
        gray_cur: Current grayscale image
        pose: Estimated pose (4x4 transformation matrix)
    """
    # Create a visualization of the frames side by side
    vis_img = np.hstack((gray_ref, gray_cur))
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    # Add text with pose information
    translation = pose[:3, 3]
    tx, ty, tz = translation

    info_text = f"Translation: tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}"
    cv2.putText(vis_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Visualize depth image
    depth_vis = np.clip(depth_ref, 0, 5)  # Clip to 5 meters for better visualization
    depth_vis = (depth_vis / 5.0 * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Show images
    cv2.imshow("Tracking Result", vis_img)
    cv2.imshow("Depth", depth_vis)
    cv2.waitKey(0)


def load_groundtruth_relative(input_dir):
    """
    Load relative groundtruth motion between consecutive frames.

    Args:
        input_dir: Directory containing the groundtruth_relative.txt file

    Returns:
        dict mapping frame pair strings (e.g., 'frame_0_to_frame_1') to 4x4 transformation matrices
        or None if no groundtruth file is found
    """
    gt_path = os.path.join(input_dir, "groundtruth_relative.txt")
    if not os.path.exists(gt_path):
        return None

    relative_motions = {}

    with open(gt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue

            parts = line.strip().split()
            if len(parts) == 8:  # frame_pair tx ty tz qx qy qz qw
                frame_pair = parts[0]
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                # Convert quaternion to rotation matrix
                # Formula from: http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
                xx = qx * qx
                xy = qx * qy
                xz = qx * qz
                xw = qx * qw
                yy = qy * qy
                yz = qy * qz
                yw = qy * qw
                zz = qz * qz
                zw = qz * qw

                r00 = 1 - 2 * (yy + zz)
                r01 = 2 * (xy - zw)
                r02 = 2 * (xz + yw)
                r10 = 2 * (xy + zw)
                r11 = 1 - 2 * (xx + zz)
                r12 = 2 * (yz - xw)
                r20 = 2 * (xz - yw)
                r21 = 2 * (yz + xw)
                r22 = 1 - 2 * (xx + yy)

                # Create 4x4 transformation matrix
                transform = np.eye(4)
                transform[0, 0] = r00
                transform[0, 1] = r01
                transform[0, 2] = r02
                transform[1, 0] = r10
                transform[1, 1] = r11
                transform[1, 2] = r12
                transform[2, 0] = r20
                transform[2, 1] = r21
                transform[2, 2] = r22
                transform[0, 3] = tx
                transform[1, 3] = ty
                transform[2, 3] = tz

                relative_motions[frame_pair] = transform

    if relative_motions:
        print(f"Loaded {len(relative_motions)} relative groundtruth motions")

    return relative_motions


def calculate_pose_error(estimated_pose, groundtruth_pose):
    """
    Calculate the error between estimated and groundtruth poses.

    Args:
        estimated_pose: 4x4 transformation matrix (estimated)
        groundtruth_pose: 4x4 transformation matrix (groundtruth)

    Returns:
        dict containing translation error (meters) and rotation error (degrees)
    """
    # Compute the relative transformation error
    error_transform = np.linalg.inv(groundtruth_pose) @ estimated_pose

    # Translation error (Euclidean distance)
    trans_error = np.linalg.norm(error_transform[:3, 3])

    # Rotation error (convert to angle in degrees)
    # Use the formula: θ = arccos((trace(R) - 1) / 2)
    R_error = error_transform[:3, :3]
    trace = np.trace(R_error)

    # Clamp trace to valid range for arccos
    trace_clamped = min(max(trace, -1.0), 3.0)
    rot_error_rad = np.arccos((trace_clamped - 1.0) / 2.0)
    rot_error_deg = np.degrees(rot_error_rad)

    return {
        "translation_error": trans_error,
        "rotation_error": rot_error_deg
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process two RGB-D frames to estimate camera motion")
    parser.add_argument('-i', '--input', type=str, default="./data", help="Path to directory containing frames")
    parser.add_argument('-c', '--intrinsics', type=str, help="Path to camera intrinsics file (optional)")
    parser.add_argument('--debug', action='store_true', help="Enable debug output and visualization")
    parser.add_argument('--depth-scale', type=float, default=5000.0, help="Depth scale factor (default: 5000.0)")
    args = parser.parse_args()

    print(f"Using depth scale of {args.depth_scale}")

    # Load frames
    try:
        frames = load_frames_from_images(args.input, args.depth_scale)
    except Exception as e:
        print(f"Error loading frames: {e}")
        sys.exit(1)

    # Load intrinsics
    try:
        intrinsics = load_intrinsics(args.input, args.intrinsics)
    except Exception as e:
        print(f"Error loading intrinsics: {e}")
        sys.exit(1)

    # Create camera model
    cam_model = CameraModel(
        fx=intrinsics['fx'],
        fy=intrinsics['fy'],
        cx=intrinsics['cx'],
        cy=intrinsics['cy'],
        width=intrinsics['width'],
        height=intrinsics['height']
    )

    # Load groundtruth if available
    groundtruth = load_groundtruth_relative(args.input)
    gt_pose = None
    if groundtruth and 'frame_0_to_frame_1' in groundtruth:
        gt_pose = groundtruth['frame_0_to_frame_1']
        print(f"Loaded groundtruth motion from frame 0 to frame 1")

    # Extract reference and current frames
    gray_ref = frames['ref']['gray']
    depth_ref = frames['ref']['depth']
    gray_cur = frames['cur']['gray']

    # Print some stats about the depth image
    print(f"\nDepth image stats:")
    print(f"Shape: {depth_ref.shape}")
    print(f"Data type: {depth_ref.dtype}")
    valid_depths = depth_ref[depth_ref > 0]
    if len(valid_depths) > 0:
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        mean_depth = np.mean(valid_depths)
        print(f"Min depth: {min_depth}")
        print(f"Max depth: {max_depth}")
        print(f"Mean depth: {mean_depth}")
    else:
        print("No valid depth values found!")
        sys.exit(1)
    print(
        f"Valid depth pixels: {np.sum(depth_ref > 0)} / {depth_ref.size} ({np.sum(depth_ref > 0) / depth_ref.size * 100:.2f}%)")
    print(f"First few depth values: {depth_ref.flatten()[:20]}")

    # Auto-adjust max_depth based on the input data
    # Use 2x the max observed depth
    auto_max_depth = max(20.0, max_depth * 1.5)
    print(f"Auto-adjusting max_depth to {auto_max_depth}")

    # Create tracker with adjusted max_depth
    tracker = Tracker(cam_model, debug=True, max_depth=auto_max_depth)

    # Perform tracking
    print("\nPerforming motion estimation...")
    start_time = time.time()
    pose = tracker.track(gray_ref, depth_ref, gray_cur)
    end_time = time.time()

    # Calculate residuals after tracking
    residuals = tracker.calculate_residuals(gray_ref, depth_ref, gray_cur, pose)
    valid_residuals = np.sum(~np.isnan(residuals))
    total_residuals = residuals.size
    valid_residuals_percent = (valid_residuals / total_residuals) * 100

    # Report results
    print(f"\nMotion Estimation Results:")
    print(f"Processing time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Used resolution: {gray_ref.shape[1]}x{gray_ref.shape[0]}")
    print(f"Valid residuals: {valid_residuals_percent:.2f}%")

    # Print the estimated transformation matrix
    print("\nEstimated transformation matrix:")
    print(np.array2string(pose, precision=6, suppress_small=True))

    # Extract and print translation components
    tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]
    print(f"\nTranslation (meters): tx={tx:.6f}, ty={ty:.6f}, tz={tz:.6f}")

    # Calculate translation magnitude
    trans_magnitude = np.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
    print(f"Translation magnitude: {trans_magnitude:.6f} meters")

    # Extract rotation as Euler angles
    R = pose[:3, :3]
    euler_angles = rotation_matrix_to_euler_angles(R)
    rx, ry, rz = np.degrees(euler_angles)
    print(f"Rotation (degrees): rx={rx:.6f}°, ry={ry:.6f}°, rz={rz:.6f}°")

    # If groundtruth is available, compute and report error
    if gt_pose is not None:
        error = calculate_pose_error(pose, gt_pose)
        print("\nError Analysis:")
        print(f"Translation error: {error['translation_error']:.6f} meters")
        print(f"Rotation error: {error['rotation_error']:.6f} degrees")

        # Save error to file
        error_file = os.path.join(args.input, "motion_error.txt")
        with open(error_file, 'w') as f:
            f.write(f"# Translation error (m), Rotation error (deg)\n")
            f.write(f"{error['translation_error']:.6f} {error['rotation_error']:.6f}\n")
        print(f"Saved error metrics to {error_file}")

        # Print groundtruth transformation for comparison
        print("\nGroundtruth transformation matrix:")
        print(np.array2string(gt_pose, precision=6, suppress_small=True))

        # Extract and print groundtruth translation components
        gt_tx, gt_ty, gt_tz = gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3]
        print(f"\nGroundtruth translation (meters): tx={gt_tx:.6f}, ty={gt_ty:.6f}, tz={gt_tz:.6f}")

        # Calculate groundtruth translation magnitude
        gt_trans_magnitude = np.sqrt(gt_tx ** 2 + gt_ty ** 2 + gt_tz ** 2)
        print(f"Groundtruth translation magnitude: {gt_trans_magnitude:.6f} meters")

        # Extract groundtruth rotation as Euler angles
        gt_R = gt_pose[:3, :3]
        gt_euler_angles = rotation_matrix_to_euler_angles(gt_R)
        gt_rx, gt_ry, gt_rz = np.degrees(gt_euler_angles)
        print(f"Groundtruth rotation (degrees): rx={gt_rx:.6f}°, ry={gt_ry:.6f}°, rz={gt_rz:.6f}°")

    # Debug visualization
    if args.debug:
        height, width = gray_ref.shape

        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))

        # Show grayscale images
        plt.subplot(2, 2, 1)
        visualize_tracking(gray_ref, depth_ref, gray_cur, pose)

        # Show depth image
        plt.subplot(2, 2, 2)
        plt.imshow(depth_ref, cmap='jet')
        plt.colorbar(label='Depth (m)')
        plt.title('Reference Frame Depth')

        # Show residuals
        plt.subplot(2, 2, 3)
        visualize_residuals(gray_ref, gray_cur, residuals, height, width)

        # Show histogram of residuals
        plt.subplot(2, 2, 4)
        valid_residuals_array = residuals[~np.isnan(residuals)]
        plt.hist(valid_residuals_array, bins=50, range=(-50, 50))
        plt.title(f'Residual Histogram ({len(valid_residuals_array)} valid points)')
        plt.xlabel('Residual Value')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.show()

    # Return to indicate everything was successful
    return 0


if __name__ == "__main__":
    main() 