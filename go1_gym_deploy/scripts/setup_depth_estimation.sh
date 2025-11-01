#!/bin/bash
# Quick setup script for stereo depth estimation system

set -e  # Exit on error

echo "========================================="
echo "Stereo Depth Estimation Setup"
echo "========================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "Working directory: $(pwd)"
echo ""

# Check if FoundationStereo pretrained model exists
PRETRAINED_MODEL="../../FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth"
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "❌ Error: FoundationStereo pretrained model not found!"
    echo "   Expected: $PRETRAINED_MODEL"
    echo ""
    echo "   Please download the model:"
    echo "   1. Visit: https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf"
    echo "   2. Download '11-33-40' folder (VIT-Small model)"
    echo "   3. Place it in: ../../FoundationStereo/pretrained_models/"
    echo ""
    exit 1
fi

echo "✓ Found FoundationStereo pretrained model"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p models
mkdir -p config
mkdir -p ../tests
echo "✓ Directories created"
echo ""

# Step 1: Export ONNX model
echo "========================================="
echo "Step 1: Export Lightweight ONNX Model"
echo "========================================="
echo ""

if [ -f "models/stereo_lightweight.onnx" ]; then
    echo "⚠️  ONNX model already exists"
    read -p "   Overwrite? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Skipping export..."
    else
        python scripts/export_lightweight_stereo_onnx.py \
            --ckpt_dir "$PRETRAINED_MODEL" \
            --save_path models/stereo_lightweight.onnx \
            --height 224 \
            --width 384 \
            --iters 12 \
            --use_small_vit \
            --verify
    fi
else
    python scripts/export_lightweight_stereo_onnx.py \
        --ckpt_dir "$PRETRAINED_MODEL" \
        --save_path models/stereo_lightweight.onnx \
        --height 224 \
        --width 384 \
        --iters 12 \
        --use_small_vit \
        --verify
fi

echo ""
echo "✓ ONNX model ready"
echo ""

# Step 2: Create camera config
echo "========================================="
echo "Step 2: Create Camera Configuration"
echo "========================================="
echo ""

if [ -f "config/camera_params.npz" ]; then
    echo "⚠️  Camera config already exists"
    read -p "   Overwrite? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Skipping camera config..."
    else
        python scripts/create_camera_config.py \
            --output_path config/camera_params.npz \
            --use_defaults
    fi
else
    python scripts/create_camera_config.py \
        --output_path config/camera_params.npz \
        --use_defaults
fi

echo ""
echo "✓ Camera config ready"
echo ""

# Step 3: Test inference
echo "========================================="
echo "Step 3: Test Inference (Optional)"
echo "========================================="
echo ""

read -p "Run inference test? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "../../FoundationStereo/assets/left.png" ]; then
        python scripts/test_stereo_inference.py \
            --model_path models/stereo_lightweight.onnx \
            --left_img ../../FoundationStereo/assets/left.png \
            --right_img ../../FoundationStereo/assets/right.png \
            --visualize \
            --benchmark \
            --benchmark_iters 50
    else
        echo "⚠️  Test images not found, skipping visualization test"
        python scripts/test_stereo_inference.py \
            --model_path models/stereo_lightweight.onnx \
            --benchmark \
            --benchmark_iters 50
    fi
else
    echo "Skipping inference test"
fi

echo ""

# Step 4: Run module tests
echo "========================================="
echo "Step 4: Run Module Tests (Optional)"
echo "========================================="
echo ""

read -p "Run module tests? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python tests/test_depth_module.py
else
    echo "Skipping module tests"
fi

echo ""

# Summary
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Files created:"
echo "  ✓ models/stereo_lightweight.onnx"
echo "  ✓ config/camera_params.npz"
echo ""
echo "Next steps:"
echo ""
echo "1. (Optional) Convert to TensorRT for faster inference:"
echo "   trtexec --onnx=models/stereo_lightweight.onnx \\"
echo "           --saveEngine=models/stereo_lightweight.trt \\"
echo "           --fp16"
echo ""
echo "2. (Optional) Calibrate your cameras for accurate depth:"
echo "   python scripts/create_camera_config.py \\"
echo "          --focal_length <fx> --cx <cx> --cy <cy> \\"
echo "          --baseline <baseline_meters> \\"
echo "          --output_path config/camera_params.npz"
echo ""
echo "3. Deploy with depth estimation:"
echo "   python scripts/deploy_with_depth.py \\"
echo "          --label gait-conditioned-agility/2025-10-29/train \\"
echo "          --enable_depth_viz"
echo ""
echo "For more information, see:"
echo "  README_DEPTH_ESTIMATION.md"
echo ""
