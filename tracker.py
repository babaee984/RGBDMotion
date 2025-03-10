import numpy as np
import cv2

from camera_model import CameraModel
from utils import (
    bilinear_interpolation,
    compute_image_gradients,
    transform_points,
    create_pixel_grid,
    compute_se3_jacobian
)


class Tracker:
    """
    Direct visual odometry tracker that estimates camera pose by minimizing photometric error.
    """

    def __init__(self, camera_model, max_iterations=20, min_gradient_magnitude=5.0,
                 min_depth=0.1, max_depth=10.0, debug=False):
        """
        Initialize the tracker.

        Args:
            camera_model: CameraModel instance
            max_iterations: Maximum number of iterations for optimization
            min_gradient_magnitude: Minimum gradient magnitude to consider a pixel
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
            debug: Enable debug output
        """
        self.camera_model = camera_model
        self.max_iterations = max_iterations
        self.min_gradient_magnitude = min_gradient_magnitude
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.debug = debug

        # Create pixel grid
        width = camera_model.width()
        height = camera_model.height()
        self.pixel_grid = create_pixel_grid(width, height)

        # Create row, col indices for reshape operations
        self.rows, self.cols = np.indices((height, width))

    def calculate_residuals(self, gray_ref, depth_ref, gray_cur, pose_prev_to_cur):
        """
        Calculate residuals between reference and current frame.

        Args:
            gray_ref: Reference grayscale image (numpy array)
            depth_ref: Reference depth image (numpy array)
            gray_cur: Current grayscale image (numpy array)
            pose_prev_to_cur: 4x4 transformation matrix from reference to current frame

        Returns:
            residuals: numpy array of residuals for each pixel
        """
        height, width = gray_ref.shape
        n_pixels = height * width

        # Initialize residuals with NaN
        residuals = np.full(n_pixels, np.nan, dtype=np.float32)

        # Get valid depths
        depths = depth_ref.reshape(-1)
        valid_depths = (depths > self.min_depth) & (depths < self.max_depth) & np.isfinite(depths)

        if self.debug:
            print(f"Depth range for residuals: min={self.min_depth}, max={self.max_depth}")
            print(f"Depth stats: min={np.min(depths[np.isfinite(depths)])}, max={np.max(depths[np.isfinite(depths)])}")
            print(f"Pixels with valid finite depth: {np.sum(np.isfinite(depths))}")
            print(f"Pixels with depth > min_depth: {np.sum(depths > self.min_depth)}")
            print(f"Pixels with depth < max_depth: {np.sum(depths < self.max_depth)}")
            print(f"Pixels satisfying all depth criteria: {np.sum(valid_depths)}")

        if not np.any(valid_depths):
            if self.debug:
                print("No valid depths found for residual calculation!")
            return residuals

        # Back-project valid pixels to 3D
        valid_indices = np.where(valid_depths)[0]
        valid_pixel_coords = self.pixel_grid[valid_indices]
        valid_depths_values = depths[valid_indices]

        if self.debug:
            print(f"Valid pixel coords: {valid_pixel_coords.shape}, valid depths: {valid_depths_values.shape}")

        points_3d = self.camera_model.back_project(valid_pixel_coords, valid_depths_values)

        # Transform 3D points to current frame
        transformed_points = transform_points(points_3d, pose_prev_to_cur)

        # Project 3D points to current frame
        projected_points = self.camera_model.project(transformed_points)

        # Filter points that project outside image boundaries
        in_bounds = ((projected_points[:, 0] >= 0) &
                     (projected_points[:, 0] < width - 1) &
                     (projected_points[:, 1] >= 0) &
                     (projected_points[:, 1] < height - 1))

        if self.debug:
            print(f"Projected points: {projected_points.shape}, in bounds: {np.sum(in_bounds)}")

        if not np.any(in_bounds):
            if self.debug:
                print("No valid projections found for residual calculation!")
            return residuals

        # Get valid projected points
        valid_indices_in_bounds = valid_indices[in_bounds]
        valid_projected_points = projected_points[in_bounds]

        # Interpolate intensity in current frame
        interpolated = bilinear_interpolation(gray_cur, valid_projected_points)

        # Get reference intensities for valid pixels
        ref_intensities = gray_ref.reshape(-1)[valid_indices_in_bounds]

        # Calculate residuals
        residuals[valid_indices_in_bounds] = interpolated - ref_intensities

        if self.debug:
            print(f"Valid residuals: {np.sum(np.isfinite(residuals))} out of {n_pixels}")

        return residuals

    def calculate_jacobians(self, gray_ref, depth_ref, pose_prev_to_cur):
        """
        Calculate Jacobians for each pixel.

        Args:
            gray_ref: Reference grayscale image (numpy array)
            depth_ref: Reference depth image (numpy array)
            pose_prev_to_cur: 4x4 transformation matrix from reference to current frame

        Returns:
            jacobians: (N, 6) numpy array of jacobians for each pixel
        """
        height, width = gray_ref.shape
        n_pixels = height * width

        # Initialize jacobians with NaN
        jacobians = np.full((n_pixels, 6), np.nan, dtype=np.float32)

        # Calculate image gradients
        grad_x, grad_y = compute_image_gradients(gray_ref)
        grad_x_flat = grad_x.reshape(-1)
        grad_y_flat = grad_y.reshape(-1)

        # Get gradient magnitude
        grad_magnitude = np.sqrt(grad_x_flat ** 2 + grad_y_flat ** 2)

        # Get valid depths and gradients
        depths = depth_ref.reshape(-1)
        valid_pixels = ((depths > self.min_depth) &
                        (depths < self.max_depth) &
                        np.isfinite(depths) &
                        (grad_magnitude > self.min_gradient_magnitude))

        if not np.any(valid_pixels):
            if self.debug:
                print("No valid pixels found for Jacobian calculation!")
            return jacobians

        # Get indices of valid pixels
        valid_indices = np.where(valid_pixels)[0]

        # Back-project valid pixels to 3D
        valid_pixel_coords = self.pixel_grid[valid_indices]
        valid_depths_values = depths[valid_indices]

        points_3d = self.camera_model.back_project(valid_pixel_coords, valid_depths_values)

        # Transform 3D points to current frame
        transformed_points = transform_points(points_3d, pose_prev_to_cur)

        # Compute Jacobian of SE3 transformation
        se3_jacobian = compute_se3_jacobian(transformed_points, self.camera_model)

        # Compute image Jacobian (combine image gradient with geometric Jacobian)
        for i, idx in enumerate(valid_indices):
            if not np.all(np.isfinite(se3_jacobian[i])):
                continue

            # Get intensity gradients at this pixel
            dx = grad_x_flat[idx]
            dy = grad_y_flat[idx]

            # Combine with SE3 Jacobian - using the new format with 12 values
            # For each parameter, we combine the x and y derivatives with dx and dy
            # tx
            jacobians[idx, 0] = dx * se3_jacobian[i, 0] + dy * se3_jacobian[i, 1]
            # ty
            jacobians[idx, 1] = dx * se3_jacobian[i, 2] + dy * se3_jacobian[i, 3]
            # tz
            jacobians[idx, 2] = dx * se3_jacobian[i, 4] + dy * se3_jacobian[i, 5]
            # rx
            jacobians[idx, 3] = dx * se3_jacobian[i, 6] + dy * se3_jacobian[i, 7]
            # ry
            jacobians[idx, 4] = dx * se3_jacobian[i, 8] + dy * se3_jacobian[i, 9]
            # rz
            jacobians[idx, 5] = dx * se3_jacobian[i, 10] + dy * se3_jacobian[i, 11]

        if self.debug:
            print(f"Valid jacobians: {np.sum(np.isfinite(jacobians[:, 0]))} out of {n_pixels}")

        return jacobians

    def calculate_hessian_and_gradient(self, residuals, jacobians):
        """
        Calculate Hessian matrix and gradient vector for optimization.

        Args:
            residuals: numpy array of residuals
            jacobians: (N, 6) numpy array of jacobians

        Returns:
            tuple (hessian, gradient) where hessian is (6, 6) and gradient is (6,)
        """
        # Initialize Hessian and gradient
        hessian = np.zeros((6, 6), dtype=np.float32)
        gradient = np.zeros(6, dtype=np.float32)

        # Find valid residuals and jacobians
        valid = np.isfinite(residuals) & np.all(np.isfinite(jacobians), axis=1)

        if not np.any(valid):
            if self.debug:
                print("No valid residuals and jacobians for Hessian and gradient calculation!")
            return hessian, gradient

        # Extract valid residuals and jacobians
        valid_residuals = residuals[valid]
        valid_jacobians = jacobians[valid]

        # Calculate Hessian: J^T * J
        for i in range(6):
            for j in range(6):
                hessian[i, j] = np.sum(valid_jacobians[:, i] * valid_jacobians[:, j])

        # Calculate gradient: J^T * r
        for i in range(6):
            gradient[i] = np.sum(valid_jacobians[:, i] * valid_residuals)

        if self.debug:
            print(f"Hessian condition number: {np.linalg.cond(hessian)}")
            valid_count = np.sum(valid)
            print(f"Valid points for optimization: {valid_count}")

        return hessian, gradient

    def solve_linear_system(self, hessian, gradient):
        """
        Solve the linear system H * x = -g for the pose update.

        Args:
            hessian: (6, 6) Hessian matrix
            gradient: (6, ) gradient vector

        Returns:
            (6, ) numpy array of pose update parameters
        """
        # Check if Hessian is invertible
        if np.linalg.det(hessian) < 1e-10:
            # Add small regularization if needed
            hessian = hessian + np.eye(6) * 1e-4

        # Solve the system H * x = -g
        try:
            pose_update = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            print("Linear algebra error when solving system. Using pseudo-inverse instead.")
            pose_update = -np.linalg.pinv(hessian) @ gradient

        return pose_update

    def update_pose(self, pose, update):
        """
        Update the pose using the computed update.

        Args:
            pose: 4x4 transformation matrix
            update: (6,) numpy array of pose update parameters [tx, ty, tz, rx, ry, rz]

        Returns:
            Updated 4x4 transformation matrix
        """
        # Extract translation and rotation components
        tx, ty, tz, rx, ry, rz = update

        # Create translation and rotation matrices
        translation = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

        # Create rotation matrices for each axis
        def create_rotation_matrix(axis, angle):
            c = np.cos(angle)
            s = np.sin(angle)

            if axis == 'x':
                return np.array([
                    [1, 0, 0, 0],
                    [0, c, -s, 0],
                    [0, s, c, 0],
                    [0, 0, 0, 1]
                ])
            elif axis == 'y':
                return np.array([
                    [c, 0, s, 0],
                    [0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [0, 0, 0, 1]
                ])
            elif axis == 'z':
                return np.array([
                    [c, -s, 0, 0],
                    [s, c, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])

        # Combine rotation matrices
        rotation_x = create_rotation_matrix('x', rx)
        rotation_y = create_rotation_matrix('y', ry)
        rotation_z = create_rotation_matrix('z', rz)

        rotation = rotation_z @ rotation_y @ rotation_x

        # Combine translation and rotation into a single transformation
        update_matrix = translation @ rotation

        # Apply update to the current pose
        updated_pose = pose @ update_matrix

        return updated_pose

    def track(self, gray_ref, depth_ref, gray_cur, initial_guess=None):
        """
        Track the camera pose from reference to current frame.

        Args:
            gray_ref: Reference grayscale image (numpy array)
            depth_ref: Reference depth image (numpy array)
            gray_cur: Current grayscale image (numpy array)
            initial_guess: Initial guess for the pose (default: identity)

        Returns:
            4x4 transformation matrix from reference to current frame
        """
        # Initialize pose
        if initial_guess is None:
            pose = np.eye(4)
        else:
            pose = initial_guess.copy()

        # Convert images to float32 if needed
        if gray_ref.dtype != np.float32:
            gray_ref = gray_ref.astype(np.float32)
        if depth_ref.dtype != np.float32:
            depth_ref = depth_ref.astype(np.float32)
        if gray_cur.dtype != np.float32:
            gray_cur = gray_cur.astype(np.float32)

        # Perform optimization iterations
        for iteration in range(self.max_iterations):
            # Calculate residuals
            residuals = self.calculate_residuals(gray_ref, depth_ref, gray_cur, pose)

            # Calculate Jacobians
            jacobians = self.calculate_jacobians(gray_ref, depth_ref, pose)

            # Calculate Hessian and gradient
            hessian, gradient = self.calculate_hessian_and_gradient(residuals, jacobians)

            # Solve linear system
            pose_update = self.solve_linear_system(hessian, gradient)

            # Check for convergence
            update_norm = np.linalg.norm(pose_update)
            if self.debug:
                print(f"Iteration {iteration}: update norm = {update_norm:.6f}")

            if update_norm < 1e-6:
                break

            # Update pose
            pose = self.update_pose(pose, pose_update)

        if self.debug:
            print(f"Final pose:\n{pose}")

        return pose
