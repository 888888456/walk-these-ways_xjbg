# 🚀 Quick Start: Stereo Depth Estimation

One-page guide to get depth estimation running on your Go1 robot.

## 📋 Prerequisites

- [x] Go1 robot with belly-mounted stereo cameras
- [x] Trained policy with terrain observation enabled
- [x] FoundationStereo pretrained model downloaded
- [x] CUDA-capable GPU (Jetson NX or better)

## ⚡ 5-Minute Setup

```bash
# 1. Run automated setup
cd /home/user/webapp/go1_gym_deploy
bash scripts/setup_depth_estimation.sh

# That's it! The script will:
# - Export lightweight ONNX model
# - Create camera configuration
# - Run tests (optional)
```

## 🎮 Deploy Now

```bash
# Basic deployment
python scripts/deploy_with_depth.py \
    --label gait-conditioned-agility/2025-10-29/train

# With real-time visualization (for debugging)
python scripts/deploy_with_depth.py \
    --label gait-conditioned-agility/2025-10-29/train \
    --enable_depth_viz
```

## 📊 Verify It's Working

Watch for these in the console output:

```
frq: 49.8 Hz                    # ✅ Control loop ~50 Hz
  Depth: 22.3 FPS, 17.2ms       # ✅ Depth inference running
```

**Good:**
- Control loop stays at 48-52 Hz
- Depth FPS is 15-30 Hz
- No error messages

**Bad:**
- Control loop drops below 40 Hz → Reduce depth FPS or use TensorRT
- Depth FPS below 10 Hz → Model too slow, reduce resolution
- Errors about missing files → Run setup script again

## 🔧 Common Issues

### Issue: Model too slow

```bash
# Solution 1: Use smaller model
python scripts/export_lightweight_stereo_onnx.py \
    --height 192 --width 320 --iters 8

# Solution 2: Convert to TensorRT
trtexec --onnx=models/stereo_lightweight.onnx \
        --saveEngine=models/stereo_lightweight.trt \
        --fp16

# Then deploy with TRT model
python scripts/deploy_with_depth.py \
    --label <your_label> \
    --stereo_model models/stereo_lightweight.trt
```

### Issue: Depth looks wrong

```bash
# Calibrate your cameras properly
python scripts/create_camera_config.py \
    --focal_length <measured_fx> \
    --cx <measured_cx> \
    --cy <measured_cy> \
    --baseline <measured_baseline_meters> \
    --output_path config/camera_params.npz
```

### Issue: Files not found

```bash
# Re-run setup
bash scripts/setup_depth_estimation.sh
```

## 📁 File Structure

After setup, you should have:

```
go1_gym_deploy/
├── models/
│   └── stereo_lightweight.onnx    # ✅ Created by setup
├── config/
│   └── camera_params.npz          # ✅ Created by setup
└── scripts/
    └── deploy_with_depth.py       # ✅ Use this to deploy
```

## 🎯 Expected Performance

| Hardware | Input Size | Inference Time | FPS |
|----------|------------|----------------|-----|
| Jetson NX (ONNX) | 224×384 | 20-30 ms | 33-50 |
| Jetson NX (TRT) | 224×384 | 10-15 ms | 66-100 |
| Jetson Orin (ONNX) | 256×448 | 15-20 ms | 50-66 |
| GPU 3090 (ONNX) | 448×672 | 5-8 ms | 125-200 |

Control loop should always stay at ~50 Hz regardless of depth inference speed (runs in background).

## 📚 Full Documentation

For detailed information, see: `README_DEPTH_ESTIMATION.md`

## 🆘 Quick Help

```bash
# Test model inference
python scripts/test_stereo_inference.py \
    --model_path models/stereo_lightweight.onnx \
    --benchmark

# Test depth module
python tests/test_depth_module.py

# Get help on deployment options
python scripts/deploy_with_depth.py --help
```

---

**Ready to deploy?** Just run:

```bash
python scripts/deploy_with_depth.py --label YOUR_POLICY_LABEL
```

That's all! 🎉
