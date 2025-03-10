import numpy as np
import cv2


def bilinear_interpolation(image, points):
    """
    Perform bilinear interpolation at the given points.

    Args:
        image: 2D numpy array
        points: (N, 2) numpy array of (x, y) coordinates

    Returns:
        numpy array of shape (N,) containing interpolated values
    """
    height, width = image.shape[:2]

    # Initialize with NaN
    values = np.full(points.shape[0], np.nan, dtype=np.float32)

    # Find valid points (within image bounds with 1-pixel border for interpolation)
    valid = (points[:, 0] >= 0) & (points[:, 0] < width - 1) & \
            (points[:, 1] >= 0) & (points[:, 1] < height - 1)

    if not np.any(valid):
        return values

    # Get integer and fractional parts
    x, y = points[valid, 0], points[valid, 1]
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1

    # Calculate interpolation weights
    dx, dy = x - x0, y - y0

    # Get pixel values at the four corners
    v00 = image[y0, x0]
    v01 = image[y0, x1]
    v10 = image[y1, x0]
    v11 = image[y1, x1]

    # Perform bilinear interpolation
    values[valid] = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v01 + (1 - dx) * dy * v10 + dx * dy * v11

    return values


def compute_image_gradients(image):
    """
    Compute image gradients in x and y directions.

    Args:
        image: 2D numpy array of the image

    Returns:
        tuple (gradients_x, gradients_y) of numpy arrays
    """
    # Use Sobel operators for gradient computation
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    return grad_x, grad_y


def transform_points(points, transformation):
    """
    Transform 3D points using a 4x4 transformation matrix.

    Args:
        points: (N, 3) numpy array of 3D points
        transformation: 4x4 transformation matrix

    Returns:
        (N, 3) numpy array of transformed points
    """
    # Ensure points is a 2D array
    if points.ndim == 1:
        points = points.reshape(1, 3)

    # Homogenize points (add 1 as the 4th coordinate)
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

    # Transform points
    transformed_h = homogeneous_points @ transformation.T

    # Dehomogenize (divide by the 4th coordinate)
    transformed = transformed_h[:, :3] / transformed_h[:, 3:4]

    return transformed


def create_pixel_grid(width, height):
    """
    Create a grid of pixel coordinates.

    Args:
        width: Image width
        height: Image height

    Returns:
        (height*width, 2) numpy array of (x, y) pixel coordinates
    """
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    return np.column_stack((xx.flatten(), yy.flatten()))


def compute_se3_jacobian(points_3d, camera_model):
    """
    Compute the Jacobian of the projection function with respect to se(3) parameters.

    Args:
        points_3d: (N, 3) numpy array of 3D points in camera coordinates
        camera_model: CameraModel instance

    Returns:
        (N, 12) numpy array of Jacobians for each point, where for each point:
        [du/dtx, dv/dtx, du/dty, dv/dty, du/dtz, dv/dtz, du/drx, dv/drx, du/dry, dv/dry, du/drz, dv/drz]
        where (u,v) are image coordinates and (tx,ty,tz,rx,ry,rz) are the SE(3) parameters
    """
    n_points = points_3d.shape[0]
    jacobians = np.zeros((n_points, 12), dtype=np.float32)

    fx = camera_model.fx()
    fy = camera_model.fy()

    for i in range(n_points):
        x, y, z = points_3d[i]

        if z <= 0 or not np.isfinite(z):
            # Point is behind the camera or invalid, set Jacobian to NaN
            jacobians[i] = np.nan
            continue

        inv_z = 1.0 / z
        inv_z_squared = inv_z * inv_z

        # Translation part
        # du/dtx (x-coordinate w.r.t. tx)
        jacobians[i, 0] = fx * inv_z
        # dv/dtx (y-coordinate w.r.t. tx)
        jacobians[i, 1] = 0

        # du/dty (x-coordinate w.r.t. ty)
        jacobians[i, 2] = 0
        # dv/dty (y-coordinate w.r.t. ty)
        jacobians[i, 3] = fy * inv_z

        # du/dtz (x-coordinate w.r.t. tz)
        jacobians[i, 4] = -fx * x * inv_z_squared
        # dv/dtz (y-coordinate w.r.t. tz)
        jacobians[i, 5] = -fy * y * inv_z_squared

        # Rotation part
        # du/drx (x-coordinate w.r.t. rx)
        jacobians[i, 6] = -fx * x * y * inv_z_squared
        # dv/drx (y-coordinate w.r.t. rx)
        jacobians[i, 7] = -fy * (1 + y * y * inv_z_squared)

        # du/dry (x-coordinate w.r.t. ry)
        jacobians[i, 8] = fx * (1 + x * x * inv_z_squared)
        # dv/dry (y-coordinate w.r.t. ry)
        jacobians[i, 9] = fy * x * y * inv_z_squared

        # du/drz (x-coordinate w.r.t. rz)
        jacobians[i, 10] = -fx * y * inv_z
        # dv/drz (y-coordinate w.r.t. rz)
        jacobians[i, 11] = fy * x * inv_z

    return jacobians


def read_camera_intrinsics(intrinsics_file):
    """
    Read camera intrinsics from a text file.

    This function supports two formats:
    1. Simple format: "fx fy cx cy width height" (all in one line)
    2. TUM format:
       ```
       width height
       fx 0 cx
       0 fy cy
       0 0 1
       distortion_coefficients (optional)
       ```

    Args:
        intrinsics_file: Path to the intrinsics text file

    Returns:
        dictionary with camera intrinsics parameters
    """
    with open(intrinsics_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Check if it's TUM format (multiple lines) or simple format (single line)
    if len(lines) >= 3:  # TUM format has at least 3 lines (4 or 5 with distortion)
        try:
            # First line: width height
            width_height = lines[0].split()
            width = int(width_height[0])
            height = int(width_height[1])

            # Second line: fx 0 cx
            fx_line = lines[1].split()
            fx = float(fx_line[0])
            cx = float(fx_line[2])

            # Third line: 0 fy cy
            fy_line = lines[2].split()
            fy = float(fy_line[1])
            cy = float(fy_line[2])

            print(f"Read TUM format intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}, width={width}, height={height}")

            return {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'width': width,
                'height': height
            }
        except (IndexError, ValueError) as e:
            print(f"Error parsing TUM format intrinsics file: {e}")
            print("Falling back to simple format parsing")

    # Simple format or fallback: fx fy cx cy width height
    try:
        params = [float(p) for p in lines[0].split()]
        if len(params) < 6:
            raise ValueError(f"Expected at least 6 parameters, got {len(params)}")

        fx, fy, cx, cy, width, height = params[:6]
        width, height = int(width), int(height)

        print(f"Read simple format intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}, width={width}, height={height}")

        return {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'width': int(width),
            'height': int(height)
        }
    except Exception as e:
        print(f"Error reading camera intrinsics file: {e}")
        print("Using default intrinsics for 640x480 resolution")

        # Default intrinsics for 640x480 resolution
        return {
            'fx': 525.0,
            'fy': 525.0,
            'cx': 319.5,
            'cy': 239.5,
            'width': 640,
            'height': 480
        }


def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (in radians).

    Args:
        R: 3x3 rotation matrix

    Returns:
        numpy array [rx, ry, rz] containing Euler angles in radians
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z]) 