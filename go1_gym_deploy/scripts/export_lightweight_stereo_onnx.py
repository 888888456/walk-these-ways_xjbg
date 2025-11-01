"""
Export a lightweight FoundationStereo model to ONNX for Jetson NX deployment.
This script creates a simplified version optimized for real-time inference.

Key optimizations:
1. Use smaller VIT backbone (vits instead of vitl)
2. Reduce GRU iterations
3. Fixed input size for better TensorRT optimization
4. FP16 precision support
"""

import sys
import os
import argparse
import torch
import torch.nn as nn

# Add FoundationStereo to path
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../FoundationStereo/')

from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder


class LightweightStereoWrapper(nn.Module):
    """
    Wrapper for FoundationStereo model optimized for deployment.
    - Fixed input resolution
    - Reduced iterations
    - Single output (disparity map)
    """
    def __init__(self, model, iters=12, input_height=224, input_width=384):
        super().__init__()
        self.model = model
        self.iters = iters
        self.input_height = input_height
        self.input_width = input_width
        
    def forward(self, left_img, right_img):
        """
        Args:
            left_img: (B, 3, H, W) RGB image [0-255]
            right_img: (B, 3, H, W) RGB image [0-255]
        Returns:
            disparity: (B, 1, H, W) disparity map
        """
        # Ensure input is the correct size
        B, C, H, W = left_img.shape
        assert H == self.input_height and W == self.input_width, \
            f"Input size must be ({self.input_height}, {self.input_width}), got ({H}, {W})"
        
        # Forward pass through the model
        with torch.cuda.amp.autocast(False):  # Disable mixed precision for ONNX export
            disparity = self.model.forward(
                left_img, 
                right_img, 
                iters=self.iters, 
                test_mode=True,
                low_memory=False
            )
        
        return disparity


def export_lightweight_model(args):
    """Export a lightweight FoundationStereo model to ONNX"""
    
    print("="*80)
    print("Exporting Lightweight FoundationStereo Model to ONNX")
    print("="*80)
    
    # Load configuration
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    
    # Force use of small VIT for efficiency
    if args.use_small_vit:
        cfg['vit_size'] = 'vits'
        print("✓ Using VIT-Small for better performance on Jetson")
    
    print(f"✓ Loading model from: {ckpt_dir}")
    
    # Load model
    model = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_dir, map_location='cpu')
    print(f"  - Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  - Global step: {ckpt.get('global_step', 'N/A')}")
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.cuda()
    
    # Wrap model
    wrapped_model = LightweightStereoWrapper(
        model, 
        iters=args.iters,
        input_height=args.height,
        input_width=args.width
    )
    wrapped_model.eval()
    
    print(f"\n✓ Model Configuration:")
    print(f"  - Input size: {args.height}x{args.width}")
    print(f"  - GRU iterations: {args.iters}")
    print(f"  - VIT backbone: {cfg.get('vit_size', 'vitl')}")
    
    # Create dummy inputs
    dummy_left = torch.randn(1, 3, args.height, args.width).cuda()
    dummy_right = torch.randn(1, 3, args.height, args.width).cuda()
    
    print(f"\n✓ Testing forward pass...")
    with torch.no_grad():
        output = wrapped_model(dummy_left, dummy_right)
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output dtype: {output.dtype}")
    
    # Export to ONNX
    print(f"\n✓ Exporting to ONNX: {args.save_path}")
    torch.onnx.export(
        wrapped_model,
        (dummy_left, dummy_right),
        args.save_path,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=['left_image', 'right_image'],
        output_names=['disparity'],
        dynamic_axes={
            'left_image': {0: 'batch'},
            'right_image': {0: 'batch'},
            'disparity': {0: 'batch'}
        } if args.dynamic_batch else None,
        verbose=False
    )
    
    print(f"\n{'='*80}")
    print("✅ ONNX Export Complete!")
    print(f"{'='*80}")
    print(f"\nSaved to: {args.save_path}")
    print(f"Model size: {os.path.getsize(args.save_path) / (1024**2):.2f} MB")
    
    # Verify ONNX model
    if args.verify:
        print(f"\n{'='*80}")
        print("Verifying ONNX Model...")
        print(f"{'='*80}")
        import onnx
        import onnxruntime as ort
        
        # Check model
        onnx_model = onnx.load(args.save_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # Test inference
        ort_session = ort.InferenceSession(args.save_path, providers=['CPUExecutionProvider'])
        
        # Prepare inputs
        left_np = dummy_left.cpu().numpy()
        right_np = dummy_right.cpu().numpy()
        
        # Run inference
        ort_inputs = {
            'left_image': left_np,
            'right_image': right_np
        }
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        torch_output = output.cpu().numpy()
        diff = abs(ort_output - torch_output).max()
        print(f"✓ Max difference between PyTorch and ONNX: {diff:.6f}")
        
        if diff < 1e-3:
            print("✅ ONNX model verification passed!")
        else:
            print("⚠️  Warning: Large difference detected, please check model")
    
    print(f"\n{'='*80}")
    print("Next Steps:")
    print(f"{'='*80}")
    print("1. Convert to TensorRT for faster inference:")
    print(f"   trtexec --onnx={args.save_path} \\")
    print(f"           --saveEngine={args.save_path.replace('.onnx', '.trt')} \\")
    print(f"           --fp16 \\")
    print(f"           --verbose")
    print("\n2. Test the model:")
    print(f"   python go1_gym_deploy/scripts/test_stereo_inference.py \\")
    print(f"          --model_path {args.save_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export lightweight FoundationStereo to ONNX')
    
    # Model configuration
    parser.add_argument('--ckpt_dir', type=str, 
                        default='./FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth',
                        help='Path to pretrained model checkpoint (use vit-small model for Jetson)')
    parser.add_argument('--save_path', type=str,
                        default='./go1_gym_deploy/models/stereo_lightweight.onnx',
                        help='Path to save ONNX model')
    
    # Input configuration
    parser.add_argument('--height', type=int, default=224,
                        help='Input height (must be divisible by 32)')
    parser.add_argument('--width', type=int, default=384,
                        help='Input width (must be divisible by 32)')
    
    # Optimization parameters
    parser.add_argument('--iters', type=int, default=12,
                        help='Number of GRU refinement iterations (lower=faster, 8-16 recommended)')
    parser.add_argument('--use_small_vit', action='store_true', default=True,
                        help='Force use VIT-small instead of VIT-large')
    
    # Export options
    parser.add_argument('--opset_version', type=int, default=14,
                        help='ONNX opset version')
    parser.add_argument('--dynamic_batch', action='store_true', default=False,
                        help='Enable dynamic batch size (not recommended for TensorRT)')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify ONNX model after export')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Validate input dimensions
    assert args.height % 32 == 0, "Height must be divisible by 32"
    assert args.width % 32 == 0, "Width must be divisible by 32"
    
    # Export model
    export_lightweight_model(args)
