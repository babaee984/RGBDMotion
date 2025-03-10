import numpy as np

class CameraModel:
    """Camera model for projecting 3D points to 2D and vice versa."""
    
    def __init__(self, fx, fy, cx, cy, width, height):
        """
        Initialize camera model with intrinsic parameters.
        
        Args:
            fx (float): Focal length in x direction
            fy (float): Focal length in y direction
            cx (float): Principal point x coordinate
            cy (float): Principal point y coordinate
            width (int): Image width in pixels
            height (int): Image height in pixels
        """
        self._fx = float(fx)
        self._fy = float(fy)
        self._cx = float(cx)
        self._cy = float(cy)
        self._width = int(width)
        self._height = int(height)
        
        # Create camera matrix
        self._camera_matrix = np.array([
            [self._fx, 0.0, self._cx],
            [0.0, self._fy, self._cy],
            [0.0, 0.0, 1.0]
        ])
        
    def fx(self):
        """Get focal length in x direction."""
        return self._fx
    
    def fy(self):
        """Get focal length in y direction."""
        return self._fy
    
    def cx(self):
        """Get principal point x coordinate."""
        return self._cx
    
    def cy(self):
        """Get principal point y coordinate."""
        return self._cy
    
    def width(self):
        """Get image width."""
        return self._width
    
    def height(self):
        """Get image height."""
        return self._height
    
    def camera_matrix(self):
        """Get the 3x3 camera intrinsic matrix."""
        return self._camera_matrix.copy()
    
    def project(self, points_3d):
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: numpy array of shape (N, 3) containing 3D points
            
        Returns:
            numpy array of shape (N, 2) containing 2D image coordinates
        """
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)
            
        # Extract components
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        
        # Check for points behind the camera
        valid_points = z > 0
        
        # Initialize projected points
        projected = np.zeros((points_3d.shape[0], 2), dtype=np.float32)
        projected.fill(np.nan)
        
        # Project valid points
        if np.any(valid_points):
            inv_z = 1.0 / z[valid_points]
            projected[valid_points, 0] = self._fx * x[valid_points] * inv_z + self._cx
            projected[valid_points, 1] = self._fy * y[valid_points] * inv_z + self._cy
            
        return projected
    
    def back_project(self, pixels, depths):
        """
        Back-project 2D pixel coordinates with depth to 3D points.
        
        Args:
            pixels: numpy array of shape (N, 2) containing 2D pixel coordinates [u, v]
            depths: numpy array of shape (N,) containing depth values
            
        Returns:
            numpy array of shape (N, 3) containing 3D points
        """
        if pixels.ndim == 1:
            pixels = pixels.reshape(1, 2)
        
        if depths.ndim == 0:
            depths = np.array([depths])
            
        # Initialize 3D points
        points_3d = np.zeros((pixels.shape[0], 3), dtype=np.float32)
        
        # Extract pixel coordinates
        u, v = pixels[:, 0], pixels[:, 1]
        
        # Back-project
        points_3d[:, 0] = (u - self._cx) * depths / self._fx
        points_3d[:, 1] = (v - self._cy) * depths / self._fy
        points_3d[:, 2] = depths
        
        return points_3d
    
    def is_in_frame(self, pixels):
        """
        Check if 2D pixel coordinates are within image boundaries.
        
        Args:
            pixels: numpy array of shape (N, 2) containing 2D pixel coordinates [u, v]
            
        Returns:
            numpy array of shape (N,) with boolean values
        """
        if pixels.ndim == 1:
            pixels = pixels.reshape(1, 2)
            
        u, v = pixels[:, 0], pixels[:, 1]
        return (u >= 0) & (u < self._width) & (v >= 0) & (v < self._height) 