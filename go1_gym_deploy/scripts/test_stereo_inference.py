"""
Test stereo depth inference with visualization.
This script allows you to:
1. Test ONNX model inference
2. Visualize disparity/depth maps in real-time
3. Benchmark inference speed
4. Compare with ground truth (if available)
"""

import sys
import os
import argparse
import time
import cv2
import numpy as np

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../FoundationStereo/')
sys.path.append(f'{code_dir}/../../')

from Utils import vis_disparity


class StereoInferenceEngine:
    """Wrapper for stereo inference with ONNX or PyTorch"""
    
    def __init__(self, model_path, use_onnx=True, use_tensorrt=False):
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.use_tensorrt = use_tensorrt
        
        if use_tensorrt:
            self._init_tensorrt()
        elif use_onnx:
            self._init_onnx()
        else:
            self._init_pytorch()
            
        print(f"✓ Inference engine initialized: {'TensorRT' if use_tensorrt else 'ONNX' if use_onnx else 'PyTorch'}")
    
    def _init_onnx(self):
        """Initialize ONNX Runtime"""
        import onnxruntime as ort
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # Get input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        
        print(f"  - Input shape: {input_shape}")
        print(f"  - Providers: {self.session.get_providers()}")
    
    def _init_tensorrt(self):
        """Initialize TensorRT"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # Load TRT engine
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(self.model_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            
            # Get input/output info
            self.input_height = self.engine.get_binding_shape(0)[2]
            self.input_width = self.engine.get_binding_shape(0)[3]
            
            print(f"  - Input shape: {self.engine.get_binding_shape(0)}")
            print(f"  - TensorRT engine loaded")
        except ImportError:
            raise ImportError("TensorRT or PyCUDA not available. Install with: pip install nvidia-tensorrt pycuda")
    
    def _init_pytorch(self):
        """Initialize PyTorch model"""
        import torch
        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo
        
        ckpt_dir = self.model_path
        cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
        
        self.model = FoundationStereo(cfg)
        ckpt = torch.load(ckpt_dir, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model.cuda()
        
        # Default input size
        self.input_height = 224
        self.input_width = 384
        
        print(f"  - PyTorch model loaded")
    
    def preprocess(self, left_img, right_img):
        """Preprocess images for inference"""
        # Resize to model input size
        left_resized = cv2.resize(left_img, (self.input_width, self.input_height))
        right_resized = cv2.resize(right_img, (self.input_width, self.input_height))
        
        # Convert to float and add batch dimension
        left_tensor = left_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        right_tensor = right_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        
        return left_tensor, right_tensor, left_img.shape[:2]
    
    def infer(self, left_img, right_img):
        """Run inference and return disparity map"""
        # Preprocess
        left_tensor, right_tensor, orig_shape = self.preprocess(left_img, right_img)
        
        # Run inference
        start_time = time.time()
        
        if self.use_tensorrt:
            disparity = self._infer_tensorrt(left_tensor, right_tensor)
        elif self.use_onnx:
            disparity = self._infer_onnx(left_tensor, right_tensor)
        else:
            disparity = self._infer_pytorch(left_tensor, right_tensor)
        
        inference_time = time.time() - start_time
        
        # Postprocess: resize back to original size
        disparity = disparity.squeeze()
        disparity_resized = cv2.resize(disparity, (orig_shape[1], orig_shape[0]), 
                                       interpolation=cv2.INTER_LINEAR)
        
        # Scale disparity based on resize ratio
        scale_x = orig_shape[1] / self.input_width
        scale_y = orig_shape[0] / self.input_height
        disparity_resized *= scale_x
        
        return disparity_resized, inference_time
    
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
        
        # Allocate device memory
        d_left = cuda.mem_alloc(left_tensor.nbytes)
        d_right = cuda.mem_alloc(right_tensor.nbytes)
        output_shape = (1, 1, self.input_height, self.input_width)
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)
        
        # Transfer input data to device
        cuda.memcpy_htod(d_left, left_tensor)
        cuda.memcpy_htod(d_right, right_tensor)
        
        # Run inference
        bindings = [int(d_left), int(d_right), int(d_output)]
        self.context.execute_v2(bindings)
        
        # Transfer predictions back
        cuda.memcpy_dtoh(output, d_output)
        
        return output
    
    def _infer_pytorch(self, left_tensor, right_tensor):
        """PyTorch inference"""
        import torch
        
        left_torch = torch.from_numpy(left_tensor).cuda() if torch.cuda.is_available() else torch.from_numpy(left_tensor)
        right_torch = torch.from_numpy(right_tensor).cuda() if torch.cuda.is_available() else torch.from_numpy(right_tensor)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(True):
                disparity = self.model.forward(left_torch, right_torch, iters=12, test_mode=True)
        
        return disparity.cpu().numpy()


def visualize_disparity(disparity, left_img, min_disp=None, max_disp=None):
    """Create visualization of disparity map"""
    H, W = disparity.shape
    
    # Visualize disparity
    vis_dict = {}
    vis_disp = vis_disparity(disparity, min_val=min_disp, max_val=max_disp, other_output=vis_dict)
    
    # Resize left image to match
    left_resized = cv2.resize(left_img, (W, H))
    
    # Create side-by-side visualization
    combined = np.hstack([left_resized, vis_disp])
    
    # Add text info
    info_text = f"Disp Range: [{vis_dict.get('min_val', 0):.1f}, {vis_dict.get('max_val', 0):.1f}]"
    cv2.putText(combined, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return combined, vis_dict


def test_on_images(engine, left_path, right_path, visualize=True, save_output=None):
    """Test on a pair of images"""
    print(f"\n{'='*80}")
    print(f"Testing on image pair:")
    print(f"  Left:  {left_path}")
    print(f"  Right: {right_path}")
    print(f"{'='*80}")
    
    # Load images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        print("❌ Error: Could not load images")
        return
    
    print(f"✓ Images loaded: {left_img.shape}")
    
    # Run inference
    disparity, inference_time = engine.infer(left_img, right_img)
    
    print(f"✓ Inference complete:")
    print(f"  - Time: {inference_time*1000:.1f} ms")
    print(f"  - FPS: {1.0/inference_time:.1f}")
    print(f"  - Disparity range: [{disparity.min():.1f}, {disparity.max():.1f}]")
    
    if visualize:
        # Create visualization
        vis_combined, vis_dict = visualize_disparity(disparity, left_img)
        
        # Show result
        cv2.imshow('Stereo Depth Inference', vis_combined)
        print(f"\n✓ Visualization window opened. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if save_output:
        vis_combined, _ = visualize_disparity(disparity, left_img)
        cv2.imwrite(save_output, vis_combined)
        
        # Also save disparity as numpy array
        disp_save_path = save_output.replace('.png', '_disparity.npy')
        np.save(disp_save_path, disparity)
        
        print(f"✓ Output saved:")
        print(f"  - Visualization: {save_output}")
        print(f"  - Disparity array: {disp_save_path}")


def benchmark(engine, num_iterations=100):
    """Benchmark inference speed"""
    print(f"\n{'='*80}")
    print(f"Benchmarking (iterations={num_iterations})...")
    print(f"{'='*80}")
    
    # Create dummy inputs
    dummy_left = np.random.randint(0, 255, (engine.input_height, engine.input_width, 3), dtype=np.uint8)
    dummy_right = np.random.randint(0, 255, (engine.input_height, engine.input_width, 3), dtype=np.uint8)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        _, _ = engine.infer(dummy_left, dummy_right)
    
    # Benchmark
    print("Running benchmark...")
    times = []
    for i in range(num_iterations):
        _, inference_time = engine.infer(dummy_left, dummy_right)
        times.append(inference_time)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_iterations}")
    
    times = np.array(times)
    
    print(f"\n✓ Benchmark Results:")
    print(f"  - Mean: {times.mean()*1000:.2f} ms ({1.0/times.mean():.1f} FPS)")
    print(f"  - Std:  {times.std()*1000:.2f} ms")
    print(f"  - Min:  {times.min()*1000:.2f} ms ({1.0/times.min():.1f} FPS)")
    print(f"  - Max:  {times.max()*1000:.2f} ms ({1.0/times.max():.1f} FPS)")
    print(f"  - P50:  {np.percentile(times, 50)*1000:.2f} ms")
    print(f"  - P95:  {np.percentile(times, 95)*1000:.2f} ms")
    print(f"  - P99:  {np.percentile(times, 99)*1000:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test stereo depth inference')
    
    # Model configuration
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to ONNX/TRT model or PyTorch checkpoint')
    parser.add_argument('--use_tensorrt', action='store_true',
                        help='Use TensorRT engine (model_path should be .trt file)')
    parser.add_argument('--use_pytorch', action='store_true',
                        help='Use PyTorch model (model_path should be .pth file)')
    
    # Test images
    parser.add_argument('--left_img', type=str, 
                        default='./FoundationStereo/assets/left.png',
                        help='Path to left image')
    parser.add_argument('--right_img', type=str,
                        default='./FoundationStereo/assets/right.png',
                        help='Path to right image')
    
    # Options
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Show visualization')
    parser.add_argument('--save_output', type=str, default=None,
                        help='Path to save output visualization')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark')
    parser.add_argument('--benchmark_iters', type=int, default=100,
                        help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    # Initialize engine
    use_onnx = not (args.use_tensorrt or args.use_pytorch)
    engine = StereoInferenceEngine(
        args.model_path,
        use_onnx=use_onnx,
        use_tensorrt=args.use_tensorrt
    )
    
    # Test on images
    test_on_images(engine, args.left_img, args.right_img, 
                   visualize=args.visualize, save_output=args.save_output)
    
    # Benchmark if requested
    if args.benchmark:
        benchmark(engine, num_iterations=args.benchmark_iters)
    
    print(f"\n{'='*80}")
    print("✅ Testing complete!")
    print(f"{'='*80}\n")
