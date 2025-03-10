# Motion Estimation from a Pair of RGB-D Images

This is a Python implementation for estimating camera motion from a pair of RGB-D images using direct photometric alignment.

## Dependencies

- Python 3.6+
- NumPy
- OpenCV
- Matplotlib (for visualization)

You can install the required packages using pip:

```bash
pip install numpy opencv-python matplotlib
```

## Directory Structure

```
Python/
├── camera_model.py  # Camera model implementation
├── tracker.py       # Main tracking algorithm
├── utils.py         # Utility functions
├── process_two_frames.py  # Process two frames for motion estimation
├── README.md           # Detailed documentation
├── data/                   # Place your image data here
│   ├── frame_0_gray.png    # Reference frame grayscale image
│   ├── frame_0_depth.png   # Reference frame depth image
│   ├── frame_1_gray.png    # Current frame grayscale image
│   ├── frame_1_depth.png   # Current frame depth image
│   ├── intrinsics.txt      # Camera intrinsics (optional)
│   └── groundtruth_relative.txt  # Groundtruth motion data (optional)
```

## Quick Start

1. Place your RGB-D image pairs in the `data` directory as shown above
2. Process the frames to estimate motion:

```bash
python examples/process_two_frames.py -i ./data --debug
```

## Camera Intrinsics

Camera intrinsics should be provided in a simple text file format:
```
fx fy cx cy width height
```

For example:
```
517.3 516.5 318.6 255.3 640 480
```

Where:
- `fx`, `fy`: Focal length in pixels
- `cx`, `cy`: Principal point coordinates
- `width`, `height`: Image dimensions in pixels

## Evaluation with Groundtruth

If you provide groundtruth data when extracting frames, the system will:

1. Automatically calculate error metrics between estimated and groundtruth motion
2. Display and save error metrics during processing
3. Generate enhanced visualizations comparing estimated and groundtruth trajectories

Error metrics include:
- Translation error (meters): Euclidean distance between estimated and groundtruth translations
- Rotation error (degrees): Angular difference between estimated and groundtruth rotations



 