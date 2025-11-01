"""
Background thread stereo depth estimator for real-time deployment.

Architecture:
    [Main Thread - LCM Callback]           [Background Thread - Inference]
            │                                      │
            │ Camera images arrive                 │ (Idle waiting)
            ▼                                      │
        _rect_camera_cb()                          │
            │                                      │
            ├─ Decode image (<1ms)                 │
            ├─ Update buffer                       │
            └─ Set flag ──────────────────────▶    │ Woken up
            │                                      │
            │ Return immediately                   ▼
            │                              Check new image
            │                                      │
            │                              ├─ Read image buffer
            │                              ├─ Read current pose
            │                              ├─ ONNX inference (15ms)
            │                              ├─ Transform to heightmap
            │                              └─ Update measured_heights
            │                                      │
            ▼                                      ▼
    (Process other messages)              (Wait for next frame)
"""

import threading
import time
import queue
import numpy as np
import cv2
import torch


class StereoDepthEstimator:
    """
    Background thread stereo depth estimator for real-time robot deployment.
    
    Features:
    - Non-blocking LCM callback
    - Background inference thread
    - Automatic coordinate transformation to robot heightmap
    - Thread-safe state management
    """
    
    def __init__(self, 
                 model_path,
                 camera_params,
                 terrain_cfg,
                 use_tensorrt=False,
                 inference_fps=30,
                 enable_visualization=False):
        """
        Args:
            model_path: Path to ONNX or TensorRT model
            camera_params: Dict with 'K' (intrinsic matrix) and 'baseline' (meters)
            terrain_cfg: Terrain configuration from training config
            use_tensorrt: Use TensorRT engine instead of ONNX
            inference_fps: Target inference FPS (will skip frames if necessary)
            enable_visualization: Show real-time depth visualization
        """
        self.model_path = model_path
        self.camera_params = camera_params
        self.terrain_cfg = terrain_cfg
        self.use_tensorrt = use_tensorrt
        self.inference_fps = inference_fps
        self.enable_visualization = enable_visualization
        
        # Thread synchronization
        self.lock = threading.Lock()
        self.new_image_event = threading.Event()
        self.stop_event = threading.Event()
        
        # Image buffers (double buffering for thread safety)
        self.left_image_buffer = None
        self.right_image_buffer = None
        self.new_image_available = False
        
        # Pose buffer
        self.current_pose = None  # [x, y, z, roll, pitch, yaw]
        
        # Output buffer - terrain heightmap
        self.measured_heights = None
        self._init_measured_heights()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'inference_times': [],
            'transform_times': [],
            'last_inference_time': 0,
            'fps': 0
        }
        
        # Initialize inference engine
        self._init_inference_engine()
        
        # Start background thread
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        print("✓ StereoDepthEstimator initialized")
        print(f"  - Model: {model_path}")
        print(f"  - Target FPS: {inference_fps}")
        print(f"  - Heightmap size: {self.measured_heights.shape}")
    
    def _init_measured_heights(self):
        """Initialize measured heights array based on terrain config"""
        if self.terrain_cfg is None or not self.terrain_cfg.get('measure_heights', False):
            # Default heightmap
            self.measured_heights = np.zeros((1, 187))  # Default from check
            self.grid_x = np.linspace(-0.5, 0.5, 17)
            self.grid_y = np.linspace(-0.5, 0.5, 11)
        else:
            # Use config from terrain
            points_x = self.terrain_cfg['measured_points_x']
            points_y = self.terrain_cfg['measured_points_y']
            self.measured_heights = np.zeros((1, len(points_x) * len(points_y)))
            self.grid_x = np.array(points_x)
            self.grid_y = np.array(points_y)
        
        print(f"  - Heightmap grid: {len(self.grid_x)} x {len(self.grid_y)}")
    
    def _init_inference_engine(self):
        """Initialize ONNX or TensorRT inference engine"""
        if self.use_tensorrt:
            self._init_tensorrt()
        else:
            self._init_onnx()
    
    def _init_onnx(self):
        """Initialize ONNX Runtime"""
        import onnxruntime as ort
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # Get input info
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"  - ONNX Runtime initialized: {input_shape}")
    
    def _init_tensorrt(self):
        """Initialize TensorRT engine"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(self.model_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            self.input_height = self.engine.get_binding_shape(0)[2]
            self.input_width = self.engine.get_binding_shape(0)[3]
            
            print(f"  - TensorRT engine initialized")
        except ImportError:
            raise ImportError("TensorRT not available. Use ONNX instead or install TensorRT.")
    
    def update_images(self, left_img, right_img):
        """
        Update image buffer from LCM callback (non-blocking).
        
        This should be called from the main LCM thread.
        
        Args:
            left_img: numpy array (H, W, 3) BGR
            right_img: numpy array (H, W, 3) BGR
        """
        with self.lock:
            self.left_image_buffer = left_img.copy()
            self.right_image_buffer = right_img.copy()
            self.new_image_available = True
        
        # Signal the inference thread
        self.new_image_event.set()
    
    def update_pose(self, pose):
        """
        Update robot pose (non-blocking).
        
        Args:
            pose: numpy array [x, y, z, roll, pitch, yaw] or 4x4 transformation matrix
        """
        with self.lock:
            self.current_pose = pose.copy()
    
    def get_measured_heights(self):
        """
        Get the latest terrain heightmap (thread-safe).
        
        Returns:
            numpy array (1, N) where N is number of measurement points
        """
        with self.lock:
            return self.measured_heights.copy()
    
    def get_stats(self):
        """Get inference statistics"""
        with self.lock:
            return self.stats.copy()
    
    def _inference_loop(self):
        """Background thread main loop"""
        print("✓ Inference thread started")
        
        min_interval = 1.0 / self.inference_fps
        last_inference_time = 0
        
        while not self.stop_event.is_set():
            # Wait for new image or timeout
            self.new_image_event.wait(timeout=0.1)
            
            # Check if we should process (respect FPS limit)
            current_time = time.time()
            if current_time - last_inference_time < min_interval:
                continue
            
            # Check if new image is available
            with self.lock:
                if not self.new_image_available:
                    continue
                
                # Copy images from buffer
                left_img = self.left_image_buffer
                right_img = self.right_image_buffer
                current_pose = self.current_pose
                self.new_image_available = False
            
            # Clear event
            self.new_image_event.clear()
            
            if left_img is None or right_img is None:
                continue
            
            # Run inference
            try:
                start_time = time.time()
                disparity = self._run_inference(left_img, right_img)
                inference_time = time.time() - start_time
                
                # Transform to heightmap
                transform_start = time.time()
                heightmap = self._transform_to_heightmap(disparity, current_pose)
                transform_time = time.time() - transform_start
                
                # Update output
                with self.lock:
                    self.measured_heights = heightmap
                    self.stats['total_frames'] += 1
                    self.stats['inference_times'].append(inference_time)
                    self.stats['transform_times'].append(transform_time)
                    self.stats['last_inference_time'] = inference_time
                    self.stats['fps'] = 1.0 / (time.time() - last_inference_time)
                    
                    # Keep only last 100 measurements
                    if len(self.stats['inference_times']) > 100:
                        self.stats['inference_times'].pop(0)
                        self.stats['transform_times'].pop(0)
                
                last_inference_time = current_time
                
                # Visualization (if enabled)
                if self.enable_visualization:
                    self._visualize(left_img, disparity, heightmap)
                
            except Exception as e:
                print(f"❌ Error in inference loop: {e}")
                import traceback
                traceback.print_exc()
        
        print("✓ Inference thread stopped")
    
    def _run_inference(self, left_img, right_img):
        """Run stereo inference and return disparity map"""
        # Preprocess
        left_resized = cv2.resize(left_img, (self.input_width, self.input_height))
        right_resized = cv2.resize(right_img, (self.input_width, self.input_height))
        
        left_tensor = left_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        right_tensor = right_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        
        # Run inference
        if self.use_tensorrt:
            disparity = self._infer_tensorrt(left_tensor, right_tensor)
        else:
            disparity = self._infer_onnx(left_tensor, right_tensor)
        
        # Resize back to original size
        disparity = disparity.squeeze()
        H, W = left_img.shape[:2]
        disparity_resized = cv2.resize(disparity, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Scale disparity
        scale_x = W / self.input_width
        disparity_resized *= scale_x
        
        return disparity_resized
    
    def _infer_onnx(self, left_tensor, right_tensor):
        """ONNX inference"""
        ort_inputs = {
            self.input_names[0]: left_tensor,
            self.input_names[1]: right_tensor
        }
        disparity = self.session.run(self.output_names, ort_inputs)[0]
        return disparity
    
    def _infer_tensorrt(self, left_tensor, right_tensor):
        """TensorRT inference"""
        import pycuda.driver as cuda
        
        # Allocate memory
        d_left = cuda.mem_alloc(left_tensor.nbytes)
        d_right = cuda.mem_alloc(right_tensor.nbytes)
        output_shape = (1, 1, self.input_height, self.input_width)
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)
        
        # Transfer and run
        cuda.memcpy_htod(d_left, left_tensor)
        cuda.memcpy_htod(d_right, right_tensor)
        bindings = [int(d_left), int(d_right), int(d_output)]
        self.context.execute_v2(bindings)
        cuda.memcpy_dtoh(output, d_output)
        
        return output
    
    def _transform_to_heightmap(self, disparity, pose):
        """
        Transform disparity map to robot-centric terrain heightmap.
        
        Args:
            disparity: (H, W) disparity map
            pose: Current robot pose [x, y, z, roll, pitch, yaw] or None
        
        Returns:
            heightmap: (1, N) flattened heightmap
        """
        # Convert disparity to depth
        K = self.camera_params['K']
        baseline = self.camera_params['baseline']
        
        # Depth = f * baseline / disparity
        focal_length = K[0, 0]
        depth = focal_length * baseline / (disparity + 1e-6)
        
        # Clip unreasonable depths
        depth = np.clip(depth, 0.1, 10.0)
        
        # Convert depth image to 3D points in camera frame
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Camera coordinates
        x_cam = (u - K[0, 2]) * depth / K[0, 0]
        y_cam = (v - K[1, 2]) * depth / K[1, 1]
        z_cam = depth
        
        # Transform to robot body frame (assuming camera mounted on belly looking down/forward)
        # Typical transformation: camera_z -> robot_x, camera_x -> robot_y, -camera_y -> robot_z
        # This depends on your specific camera mounting
        
        # Example transformation (adjust based on your setup):
        # Camera on belly, looking forward and down at ~45 degrees
        x_robot = z_cam * 0.7 + y_cam * 0.7  # Forward
        y_robot = x_cam  # Lateral
        z_robot = -z_cam * 0.7 + y_cam * 0.7  # Height (negative is down)
        
        # Apply robot pose if available
        if pose is not None:
            # Extract rotation (roll, pitch, yaw)
            if len(pose.shape) == 1:
                roll, pitch, yaw = pose[3:6]
                
                # Rotation matrices
                R_z = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]
                ])
                R_y = np.array([
                    [np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]
                ])
                R_x = np.array([
                    [1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]
                ])
                
                R = R_z @ R_y @ R_x
                
                # Apply rotation
                points = np.stack([x_robot, y_robot, z_robot], axis=-1)
                points_rotated = points @ R.T
                
                x_robot = points_rotated[..., 0]
                y_robot = points_rotated[..., 1]
                z_robot = points_rotated[..., 2]
        
        # Sample heightmap at grid points
        heightmap_flat = []
        
        for x_grid in self.grid_x:
            for y_grid in self.grid_y:
                # Find nearest points in depth map
                dist_sq = (x_robot - x_grid)**2 + (y_robot - y_grid)**2
                
                # Get heights within a small radius
                mask = dist_sq < 0.01  # 10cm radius
                
                if mask.any():
                    # Average height of nearby points
                    height = np.median(z_robot[mask])
                else:
                    # No data, use default
                    height = 0.0
                
                heightmap_flat.append(height)
        
        return np.array(heightmap_flat).reshape(1, -1)
    
    def _visualize(self, left_img, disparity, heightmap):
        """Visualize depth and heightmap"""
        import sys
        sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/../../FoundationStereo/')
        from Utils import vis_disparity
        
        # Visualize disparity
        vis_disp = vis_disparity(disparity)
        
        # Resize for display
        left_small = cv2.resize(left_img, (320, 240))
        disp_small = cv2.resize(vis_disp, (320, 240))
        
        # Combine
        combined = np.hstack([left_small, disp_small])
        
        # Add stats
        stats = self.get_stats()
        cv2.putText(combined, f"FPS: {stats['fps']:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, f"Inference: {stats['last_inference_time']*1000:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Stereo Depth (Real-time)', combined)
        cv2.waitKey(1)
    
    def stop(self):
        """Stop the inference thread"""
        self.stop_event.set()
        self.new_image_event.set()
        self.inference_thread.join(timeout=2.0)
        print("✓ StereoDepthEstimator stopped")
