"""
Create camera calibration configuration file.

This script helps you create a camera configuration file for stereo depth estimation.
You can either:
1. Use default parameters (for quick testing)
2. Load from intrinsic file (FoundationStereo format)
3. Enter custom parameters

Usage:
    # Create with defaults
    python create_camera_config.py --output_path ../config/camera_params.npz

    # Load from FoundationStereo format
    python create_camera_config.py --intrinsic_file ../../FoundationStereo/assets/K.txt \
                                    --output_path ../config/camera_params.npz
    
    # Custom parameters
    python create_camera_config.py --focal_length 200.0 --cx 50.0 --cy 58.0 \
                                    --baseline 0.063 --output_path ../config/camera_params.npz
"""

import numpy as np
import argparse
import os


def load_from_intrinsic_file(intrinsic_file):
    """
    Load camera parameters from FoundationStereo format intrinsic file.
    
    Format:
        Line 1: flattened 3x3 intrinsic matrix K
        Line 2: baseline in meters
    """
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
        baseline = float(lines[1])
    
    return K, baseline


def create_default_params():
    """
    Create default camera parameters.
    
    These are example values - you should calibrate your own cameras!
    """
    # Default intrinsic matrix for Go1 belly cameras (example values)
    # Adjust these based on your actual camera calibration
    K = np.array([
        [200.0, 0.0, 50.0],   # fx, 0, cx
        [0.0, 200.0, 58.0],    # 0, fy, cy
        [0.0, 0.0, 1.0]        # 0, 0, 1
    ], dtype=np.float32)
    
    # Default baseline (distance between left and right cameras)
    baseline = 0.063  # meters (6.3 cm)
    
    return K, baseline


def create_custom_params(focal_length, cx, cy, baseline):
    """Create camera parameters from custom values"""
    K = np.array([
        [focal_length, 0.0, cx],
        [0.0, focal_length, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    return K, float(baseline)


def save_camera_config(K, baseline, output_path):
    """Save camera configuration to npz file"""
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to npz
    np.savez(output_path, K=K, baseline=baseline)
    
    print(f"\n{'='*80}")
    print("✅ Camera configuration saved!")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"\nParameters:")
    print(f"  Intrinsic matrix K:")
    print(f"    {K[0]}")
    print(f"    {K[1]}")
    print(f"    {K[2]}")
    print(f"\n  Focal length: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
    print(f"  Principal point: cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    print(f"  Baseline: {baseline:.4f} m ({baseline*100:.2f} cm)")
    print(f"\n{'='*80}")


def visualize_camera_params(K, baseline, image_width=100, image_height=116):
    """Visualize camera parameters"""
    print(f"\n{'='*80}")
    print("Camera Parameter Visualization")
    print(f"{'='*80}")
    
    # Field of view
    fov_x = 2 * np.arctan(image_width / (2 * K[0, 0])) * 180 / np.pi
    fov_y = 2 * np.arctan(image_height / (2 * K[1, 1])) * 180 / np.pi
    
    print(f"\nField of View (assuming {image_width}x{image_height} image):")
    print(f"  Horizontal FOV: {fov_x:.1f}°")
    print(f"  Vertical FOV: {fov_y:.1f}°")
    
    # Depth range estimates
    min_disparity = 1.0  # pixels
    max_disparity = 50.0  # pixels
    
    max_depth = K[0, 0] * baseline / min_disparity
    min_depth = K[0, 0] * baseline / max_disparity
    
    print(f"\nEstimated Depth Range:")
    print(f"  Min depth: {min_depth:.2f} m (at {max_disparity:.0f} px disparity)")
    print(f"  Max depth: {max_depth:.2f} m (at {min_disparity:.0f} px disparity)")
    
    print(f"\nStereo Baseline:")
    print(f"  {baseline*100:.2f} cm between cameras")
    print(f"  Depth resolution at 1m: {1.0**2 / (K[0,0] * baseline)*1000:.2f} mm/px")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create camera calibration configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults
  python create_camera_config.py --output_path ../config/camera_params.npz
  
  # Load from intrinsic file
  python create_camera_config.py --intrinsic_file ../../FoundationStereo/assets/K.txt \\
                                  --output_path ../config/camera_params.npz
  
  # Custom parameters  
  python create_camera_config.py --focal_length 200.0 --cx 50.0 --cy 58.0 \\
                                  --baseline 0.063 --output_path ../config/camera_params.npz
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--intrinsic_file', type=str,
                             help='Path to intrinsic file (FoundationStereo format)')
    input_group.add_argument('--use_defaults', action='store_true',
                             help='Use default parameters')
    
    # Custom parameters
    parser.add_argument('--focal_length', type=float,
                        help='Focal length in pixels')
    parser.add_argument('--cx', type=float,
                        help='Principal point x coordinate')
    parser.add_argument('--cy', type=float,
                        help='Principal point y coordinate')
    parser.add_argument('--baseline', type=float,
                        help='Baseline distance in meters')
    
    # Output
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for camera config (.npz file)')
    
    # Options
    parser.add_argument('--image_width', type=int, default=100,
                        help='Image width for FOV calculation')
    parser.add_argument('--image_height', type=int, default=116,
                        help='Image height for FOV calculation')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Visualize camera parameters')
    
    args = parser.parse_args()
    
    # Determine which method to use
    if args.intrinsic_file:
        print(f"Loading parameters from: {args.intrinsic_file}")
        K, baseline = load_from_intrinsic_file(args.intrinsic_file)
    elif args.focal_length is not None:
        # Custom parameters
        if args.cx is None or args.cy is None or args.baseline is None:
            parser.error("--focal_length requires --cx, --cy, and --baseline")
        print("Using custom parameters")
        K, baseline = create_custom_params(args.focal_length, args.cx, args.cy, args.baseline)
    else:
        print("Using default parameters (⚠️  You should calibrate your cameras!)")
        K, baseline = create_default_params()
    
    # Save configuration
    save_camera_config(K, baseline, args.output_path)
    
    # Visualize if requested
    if args.visualize:
        visualize_camera_params(K, baseline, args.image_width, args.image_height)
    
    print("\n💡 Next Steps:")
    print("1. Verify the parameters match your camera setup")
    print("2. Consider calibrating your cameras for better accuracy")
    print("   (Use OpenCV calibration tools or similar)")
    print("3. Test depth estimation:")
    print(f"   python go1_gym_deploy/scripts/test_stereo_inference.py \\")
    print(f"          --model_path <your_model.onnx>")
    print()
