# 🚀 从这里开始 - 双目深度估计系统

## 📋 三种使用方式

### 方式1: 自动化设置（推荐新手）⭐

```bash
cd /home/user/webapp/go1_gym_deploy
bash scripts/setup_depth_estimation.sh
```

按照提示操作，脚本会自动完成所有设置。

---

### 方式2: 手动设置（推荐有经验用户）

```bash
# 步骤1: 导出模型
python scripts/export_lightweight_stereo_onnx.py

# 步骤2: 创建配置
python scripts/create_camera_config.py \
    --output_path config/camera_params.npz \
    --use_defaults

# 步骤3: 测试
python scripts/test_stereo_inference.py \
    --model_path models/stereo_lightweight.onnx \
    --benchmark

# 步骤4: 部署
python scripts/deploy_with_depth.py \
    --label YOUR_POLICY_LABEL
```

---

### 方式3: 调试模式（带实时可视化）

```bash
python scripts/deploy_with_depth.py \
    --label YOUR_POLICY_LABEL \
    --enable_depth_viz
```

会打开窗口显示实时深度图。

---

## 📖 文档导航

### 快速参考（5分钟）
👉 **[QUICKSTART_DEPTH.md](QUICKSTART_DEPTH.md)**
- 最快速的上手指南
- 常见问题快速解决
- 性能参考表

### 中文完整指南（推荐）
👉 **[HOW_TO_USE.md](HOW_TO_USE.md)**
- 详细的步骤说明
- 中文解释
- 问题排查

### 英文完整文档（高级）
👉 **[README_DEPTH_ESTIMATION.md](README_DEPTH_ESTIMATION.md)**
- 16页完整文档
- 所有配置选项
- 高级主题

### 系统总结
👉 **[../DEPLOYMENT_SUMMARY.md](../DEPLOYMENT_SUMMARY.md)**
- 完整的系统概述
- 文件结构说明
- 工作流示例

### 实施报告
👉 **[../IMPLEMENTATION_COMPLETE.md](../IMPLEMENTATION_COMPLETE.md)**
- 所有组件的详细说明
- 性能验证结果
- 可视化输出示例

---

## ✅ 快速检查

运行这个命令确保一切就绪：

```bash
python tests/test_depth_module.py
```

如果输出：
```
✅ PASS: Basic Inference
✅ PASS: Real Images
✅ PASS: Performance
```

说明系统工作正常！

---

## 🆘 遇到问题？

### 第一步：运行诊断
```bash
# 测试模型
python scripts/test_stereo_inference.py \
    --model_path models/stereo_lightweight.onnx \
    --benchmark

# 测试模块
python tests/test_depth_module.py

# 检查文件
ls models/stereo_lightweight.onnx
ls config/camera_params.npz
```

### 第二步：查看文档
- 问题太慢 → [HOW_TO_USE.md#性能优化](HOW_TO_USE.md#性能优化)
- 深度不准 → [HOW_TO_USE.md#问题5-深度估计不准确](HOW_TO_USE.md#问题5-深度估计不准确)
- 文件未找到 → 重新运行 `setup_depth_estimation.sh`

### 第三步：启用调试
```bash
python scripts/deploy_with_depth.py \
    --label YOUR_LABEL \
    --enable_depth_viz  # 看到实时深度图
```

---

## 🎯 预期结果

部署成功后，你应该看到：

```
✓ Loading policy from: ../../runs/...
✓ StereoDepthEstimator initialized
  - Model: models/stereo_lightweight.onnx
  - Target FPS: 20

frq: 49.8 Hz              ✅ 控制正常
  Depth: 22.3 FPS, 17.2ms ✅ 深度正常
```

---

## 📞 需要帮助？

1. 查看 [HOW_TO_USE.md](HOW_TO_USE.md) 的问题排查部分
2. 运行诊断命令
3. 检查控制台错误消息
4. 查看相应文档章节

---

**记住：** 如果不确定从哪里开始，运行：
```bash
bash scripts/setup_depth_estimation.sh
```

它会引导你完成所有步骤！✨

---

**版本**: 1.0.0  
**更新**: 2025-11-01
