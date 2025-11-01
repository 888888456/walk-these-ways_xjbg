"""
Standalone test for stereo depth estimation module.

This script tests the depth estimation system independently from the robot deployment.
Useful for verifying the system works before full deployment.
"""

import sys
import os
import time
import numpy as np
import cv2

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from go1_gym_deploy.envs.stereo_depth_estimator import StereoDepthEstimator


def test_basic_inference():
    """Test 1: Basic inference with dummy images"""
    print("\n" + "="*80)
    print("Test 1: Basic Inference")
    print("="*80)
    
    # Create dummy camera params
    camera_params = {
        'K': np.array([[200.0, 0, 50.0], [0, 200.0, 58.0], [0, 0, 1]], dtype=np.float32),
        'baseline': 0.063
    }
    
    # Dummy terrain config
    terrain_cfg = {
        'measure_heights': True,
        'measured_points_x': np.linspace(-0.5, 0.5, 17).tolist(),
        'measured_points_y': np.linspace(-0.5, 0.5, 11).tolist()
    }
    
    # Check if model exists
    model_path = os.path.join(os.path.dirname(__file__), '../models/stereo_lightweight.onnx')
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please run: python go1_gym_deploy/scripts/export_lightweight_stereo_onnx.py")
        return False
    
    print(f"✓ Using model: {model_path}")
    
    # Initialize estimator
    try:
        estimator = StereoDepthEstimator(
            model_path=model_path,
            camera_params=camera_params,
            terrain_cfg=terrain_cfg,
            use_tensorrt=False,
            inference_fps=30,
            enable_visualization=False
        )
        print("✓ Estimator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize estimator: {e}")
        return False
    
    # Create dummy images
    H, W = 116, 100
    left_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    right_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    print(f"✓ Created test images: {left_img.shape}")
    
    # Update images
    estimator.update_images(left_img, right_img)
    print("✓ Images updated (non-blocking)")
    
    # Wait for inference
    print("⏳ Waiting for inference...")
    time.sleep(0.5)
    
    # Get heightmap
    heightmap = estimator.get_measured_heights()
    print(f"✓ Heightmap retrieved: {heightmap.shape}")
    print(f"  - Min: {heightmap.min():.3f}")
    print(f"  - Max: {heightmap.max():.3f}")
    print(f"  - Mean: {heightmap.mean():.3f}")
    
    # Get stats
    stats = estimator.get_stats()
    print(f"✓ Statistics:")
    print(f"  - Total frames: {stats['total_frames']}")
    print(f"  - FPS: {stats['fps']:.1f}")
    print(f"  - Last inference: {stats['last_inference_time']*1000:.1f} ms")
    
    # Cleanup
    estimator.stop()
    print("✓ Estimator stopped")
    
    return True


def test_real_images():
    """Test 2: Test with real stereo images"""
    print("\n" + "="*80)
    print("Test 2: Real Image Inference")
    print("="*80)
    
    # Paths to test images
    left_path = os.path.join(os.path.dirname(__file__), '../../FoundationStereo/assets/left.png')
    right_path = os.path.join(os.path.dirname(__file__), '../../FoundationStereo/assets/right.png')
    
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        print(f"⚠️  Test images not found, skipping this test")
        return True
    
    print(f"✓ Loading test images:")
    print(f"  - Left: {left_path}")
    print(f"  - Right: {right_path}")
    
    # Load images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    print(f"✓ Images loaded: {left_img.shape}")
    
    # Camera params
    camera_params = {
        'K': np.array([[754.67, 0, 489.38], [0, 754.67, 265.16], [0, 0, 1]], dtype=np.float32),
        'baseline': 0.063
    }
    
    terrain_cfg = {
        'measure_heights': True,
        'measured_points_x': np.linspace(-1.0, 1.0, 21).tolist(),
        'measured_points_y': np.linspace(-0.8, 0.8, 15).tolist()
    }
    
    # Model path
    model_path = os.path.join(os.path.dirname(__file__), '../models/stereo_lightweight.onnx')
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found, skipping this test")
        return True
    
    # Initialize estimator with visualization
    print("Initializing estimator with visualization...")
    estimator = StereoDepthEstimator(
        model_path=model_path,
        camera_params=camera_params,
        terrain_cfg=terrain_cfg,
        use_tensorrt=False,
        inference_fps=30,
        enable_visualization=True
    )
    
    # Process images multiple times to test performance
    print("\nProcessing images 10 times...")
    for i in range(10):
        estimator.update_images(left_img, right_img)
        time.sleep(0.1)
        
        if (i + 1) % 5 == 0:
            stats = estimator.get_stats()
            print(f"  Frame {i+1}: {stats['fps']:.1f} FPS, {stats['last_inference_time']*1000:.1f} ms")
    
    # Final stats
    stats = estimator.get_stats()
    print(f"\n✓ Final statistics:")
    print(f"  - Total frames: {stats['total_frames']}")
    print(f"  - Average FPS: {stats['fps']:.1f}")
    if len(stats['inference_times']) > 0:
        avg_time = np.mean(stats['inference_times'])
        print(f"  - Average inference: {avg_time*1000:.1f} ms")
    
    # Get heightmap
    heightmap = estimator.get_measured_heights()
    print(f"\n✓ Final heightmap:")
    print(f"  - Shape: {heightmap.shape}")
    print(f"  - Range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")
    
    # Visualization window
    print("\n⏳ Close the visualization window to continue...")
    time.sleep(2)
    
    # Cleanup
    estimator.stop()
    cv2.destroyAllWindows()
    print("✓ Test complete")
    
    return True


def test_performance():
    """Test 3: Performance benchmark"""
    print("\n" + "="*80)
    print("Test 3: Performance Benchmark")
    print("="*80)
    
    model_path = os.path.join(os.path.dirname(__file__), '../models/stereo_lightweight.onnx')
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found, skipping this test")
        return True
    
    camera_params = {
        'K': np.array([[200.0, 0, 50.0], [0, 200.0, 58.0], [0, 0, 1]], dtype=np.float32),
        'baseline': 0.063
    }
    
    terrain_cfg = {
        'measure_heights': True,
        'measured_points_x': np.linspace(-0.5, 0.5, 17).tolist(),
        'measured_points_y': np.linspace(-0.5, 0.5, 11).tolist()
    }
    
    # Test different image sizes
    sizes = [
        (116, 100, "Go1 Camera"),
        (240, 320, "Small"),
        (480, 640, "Medium"),
    ]
    
    for H, W, name in sizes:
        print(f"\n📊 Testing {name} size: {H}x{W}")
        
        estimator = StereoDepthEstimator(
            model_path=model_path,
            camera_params=camera_params,
            terrain_cfg=terrain_cfg,
            use_tensorrt=False,
            inference_fps=1000,  # No FPS limit for benchmark
            enable_visualization=False
        )
        
        # Create test images
        left_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        right_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(5):
            estimator.update_images(left_img, right_img)
            time.sleep(0.05)
        
        # Benchmark
        num_frames = 50
        print(f"  Running {num_frames} frames...")
        start_time = time.time()
        
        for _ in range(num_frames):
            estimator.update_images(left_img, right_img)
            time.sleep(0.01)  # Small delay to allow processing
        
        # Wait for all frames to complete
        time.sleep(1.0)
        
        # Get stats
        stats = estimator.get_stats()
        if len(stats['inference_times']) > 0:
            times = np.array(stats['inference_times'])
            print(f"  ✓ Results:")
            print(f"    - Frames processed: {stats['total_frames']}")
            print(f"    - Mean time: {times.mean()*1000:.1f} ms ({1.0/times.mean():.1f} FPS)")
            print(f"    - Min time: {times.min()*1000:.1f} ms ({1.0/times.min():.1f} FPS)")
            print(f"    - Max time: {times.max()*1000:.1f} ms ({1.0/times.max():.1f} FPS)")
            print(f"    - Std: {times.std()*1000:.1f} ms")
        
        estimator.stop()
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("Stereo Depth Estimation Module Tests")
    print("="*80)
    
    tests = [
        ("Basic Inference", test_basic_inference),
        ("Real Images", test_real_images),
        ("Performance", test_performance),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with error:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check output above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
