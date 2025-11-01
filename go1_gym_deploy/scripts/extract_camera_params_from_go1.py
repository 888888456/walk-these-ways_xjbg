"""
从Go1的配置文件中提取相机内外参数

基于：
1. trans_rect_config.yaml 中的配置
2. LCM消息中的图像尺寸
3. Go1相机的已知规格
"""

import numpy as np
import yaml
import argparse
import os


def opencv_matrix_constructor(loader, node):
    """Custom constructor for OpenCV matrix format"""
    mapping = loader.construct_mapping(node, deep=True)
    return mapping


# Register the constructor globally
yaml.SafeLoader.add_constructor(u'tag:yaml.org,2002:opencv-matrix', opencv_matrix_constructor)
yaml.FullLoader.add_constructor(u'tag:yaml.org,2002:opencv-matrix', opencv_matrix_constructor)


def load_rect_config(config_path):
    """读取trans_rect_config.yaml (OpenCV格式)"""
    # Read file and skip the %YAML:1.0 line
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Find start of actual YAML content (skip %YAML:1.0)
    yaml_content = []
    for line in lines:
        if line.strip().startswith('%YAML'):
            continue
        yaml_content.append(line)
    
    yaml_str = ''.join(yaml_content)
    
    config = yaml.safe_load(yaml_str)
    
    return config


def extract_camera_params_from_config(config_path):
    """从配置文件提取相机参数"""
    
    print("="*80)
    print("从Go1配置文件提取相机参数")
    print("="*80)
    
    # 读取配置
    config = load_rect_config(config_path)
    
    # 提取关键参数
    frame_size = config['FrameSize']['data']  # [1856, 800]
    rectify_size = config['RectifyFrameSize']['data']  # [928, 800]
    h_fov = config['hFov']['data'][0]  # 90度
    
    print(f"\n从配置文件读取的参数:")
    print(f"  原始图像尺寸: {frame_size[0]:.0f} × {frame_size[1]:.0f}")
    print(f"  校正后尺寸: {rectify_size[0]:.0f} × {rectify_size[1]:.0f}")
    print(f"  水平FOV: {h_fov}°")
    
    # Go1实际相机参数（基于已知信息）
    # 腹部相机是双目鱼眼相机
    # LCM传输的是已经校正和缩放后的图像
    
    # LCM传输的图像尺寸
    lcm_image_width = 100  # 从check_camera_msgs.py
    lcm_image_height = 116  # 从check_camera_msgs.py
    
    print(f"  LCM传输尺寸: {lcm_image_width} × {lcm_image_height}")
    
    # 计算内参
    # 对于校正后的图像，假设主点在图像中心
    cx_rectified = rectify_size[0] / 2.0
    cy_rectified = rectify_size[1] / 2.0
    
    # 根据FOV计算焦距
    # f = (width / 2) / tan(fov / 2)
    fov_rad = np.radians(h_fov)
    fx_rectified = (rectify_size[0] / 2.0) / np.tan(fov_rad / 2.0)
    fy_rectified = fx_rectified  # 假设fx=fy
    
    print(f"\n校正后图像的内参 (928×800):")
    print(f"  fx = {fx_rectified:.2f} pixels")
    print(f"  fy = {fy_rectified:.2f} pixels")
    print(f"  cx = {cx_rectified:.2f} pixels")
    print(f"  cy = {cy_rectified:.2f} pixels")
    
    # 缩放到LCM传输尺寸
    scale_x = lcm_image_width / rectify_size[0]
    scale_y = lcm_image_height / rectify_size[1]
    
    fx_lcm = fx_rectified * scale_x
    fy_lcm = fy_rectified * scale_y
    cx_lcm = cx_rectified * scale_x
    cy_lcm = cy_rectified * scale_y
    
    print(f"\nLCM图像的内参 (100×116):")
    print(f"  fx = {fx_lcm:.2f} pixels")
    print(f"  fy = {fy_lcm:.2f} pixels")
    print(f"  cx = {cx_lcm:.2f} pixels")
    print(f"  cy = {cy_lcm:.2f} pixels")
    
    # 构造内参矩阵
    K_lcm = np.array([
        [fx_lcm, 0, cx_lcm],
        [0, fy_lcm, cy_lcm],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 基线距离（需要测量或从规格书获得）
    # Go1腹部双目相机的典型基线距离
    baseline = 0.063  # 约6.3cm（估计值，需要实际测量确认）
    
    print(f"\n基线距离（估计值）:")
    print(f"  baseline = {baseline*100:.1f} cm = {baseline:.3f} m")
    print(f"  ⚠️  这是估计值！请用尺子实际测量左右相机镜头中心距离")
    
    # 外参（相机在机器人上的安装位置）
    print(f"\n相机外参（安装位置）:")
    print(f"  位置: 机器人腹部")
    print(f"  方向: 向前下方，约45°角")
    print(f"  双目排列: 水平放置（基线方向为横向）")
    
    # 坐标变换（从相机坐标系到机器人坐标系）
    print(f"\n坐标系定义:")
    print(f"  相机坐标系（左相机）:")
    print(f"    X: 右 (right)")
    print(f"    Y: 下 (down)")
    print(f"    Z: 前 (forward)")
    print(f"  机器人坐标系:")
    print(f"    X: 前 (forward)")
    print(f"    Y: 左 (left)")
    print(f"    Z: 上 (up)")
    
    return K_lcm, baseline


def save_camera_params(K, baseline, output_path):
    """保存相机参数"""
    np.savez(output_path, K=K, baseline=baseline)
    print(f"\n{'='*80}")
    print(f"✅ 相机参数已保存到: {output_path}")
    print(f"{'='*80}")


def create_intrinsic_file_format(K, baseline, output_path):
    """创建FoundationStereo格式的内参文件"""
    with open(output_path, 'w') as f:
        # 第一行：展平的3×3内参矩阵
        K_flat = K.flatten()
        f.write(' '.join([f'{x}' for x in K_flat]) + '\n')
        # 第二行：基线
        f.write(f'{baseline}\n')
    
    print(f"✅ FoundationStereo格式内参文件已保存到: {output_path}")


def visualize_params(K, baseline, image_width=100, image_height=116):
    """可视化相机参数"""
    print(f"\n{'='*80}")
    print("相机参数总结")
    print(f"{'='*80}")
    
    # 内参矩阵
    print(f"\n内参矩阵 K:")
    print(f"  [{K[0,0]:7.2f}  {K[0,1]:7.2f}  {K[0,2]:7.2f}]")
    print(f"  [{K[1,0]:7.2f}  {K[1,1]:7.2f}  {K[1,2]:7.2f}]")
    print(f"  [{K[2,0]:7.2f}  {K[2,1]:7.2f}  {K[2,2]:7.2f}]")
    
    # 参数解释
    print(f"\n参数解释:")
    print(f"  fx = {K[0,0]:.2f} pixels  (X方向焦距)")
    print(f"  fy = {K[1,1]:.2f} pixels  (Y方向焦距)")
    print(f"  cx = {K[0,2]:.2f} pixels  (主点X坐标)")
    print(f"  cy = {K[1,2]:.2f} pixels  (主点Y坐标)")
    print(f"  baseline = {baseline:.4f} m = {baseline*100:.2f} cm")
    
    # 视场角
    fov_x = 2 * np.arctan(image_width / (2 * K[0, 0])) * 180 / np.pi
    fov_y = 2 * np.arctan(image_height / (2 * K[1, 1])) * 180 / np.pi
    
    print(f"\n视场角 (FOV):")
    print(f"  水平FOV: {fov_x:.1f}°")
    print(f"  垂直FOV: {fov_y:.1f}°")
    
    # 深度范围估算
    min_disparity = 1.0  # 像素
    max_disparity = 50.0  # 像素
    
    max_depth = K[0, 0] * baseline / min_disparity
    min_depth = K[0, 0] * baseline / max_disparity
    
    print(f"\n深度测量范围（估算）:")
    print(f"  最小深度: {min_depth:.2f} m  (在视差 {max_disparity:.0f} px时)")
    print(f"  最大深度: {max_depth:.2f} m  (在视差 {min_disparity:.0f} px时)")
    print(f"  有效范围: {min_depth:.2f} - {max_depth:.2f} m")
    
    # 深度分辨率
    depth_at_1m = 1.0
    disparity_at_1m = K[0, 0] * baseline / depth_at_1m
    depth_resolution = depth_at_1m**2 / (K[0, 0] * baseline) * 1000  # mm/pixel
    
    print(f"\n深度分辨率:")
    print(f"  1米处的视差: {disparity_at_1m:.2f} pixels")
    print(f"  1米处的深度分辨率: {depth_resolution:.2f} mm/pixel")
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='从Go1配置文件提取相机内外参',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从默认配置提取
  python extract_camera_params_from_go1.py
  
  # 指定配置文件
  python extract_camera_params_from_go1.py \
      --config_path ../../trans_rect_config.yaml \
      --output_path ../config/go1_camera_params.npz
        """
    )
    
    parser.add_argument('--config_path', type=str,
                        default='../../trans_rect_config.yaml',
                        help='trans_rect_config.yaml的路径')
    parser.add_argument('--output_path', type=str,
                        default='../config/go1_camera_params.npz',
                        help='输出相机参数的路径 (.npz)')
    parser.add_argument('--intrinsic_file', type=str,
                        default='../config/go1_camera_intrinsic.txt',
                        help='输出FoundationStereo格式的内参文件 (.txt)')
    parser.add_argument('--baseline', type=float,
                        default=0.063,
                        help='基线距离（米），如果已知请提供实测值')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config_path):
        print(f"❌ 错误: 配置文件不存在: {args.config_path}")
        print(f"\n请确认trans_rect_config.yaml的位置")
        print(f"或使用 --config_path 参数指定正确路径")
        return
    
    # 提取参数
    K, baseline = extract_camera_params_from_config(args.config_path)
    
    # 如果用户提供了实测基线，使用实测值
    if args.baseline != 0.063:
        baseline = args.baseline
        print(f"\n✓ 使用提供的基线距离: {baseline*100:.2f} cm")
    
    # 可视化
    visualize_params(K, baseline)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.intrinsic_file), exist_ok=True)
    
    # 保存参数
    save_camera_params(K, baseline, args.output_path)
    create_intrinsic_file_format(K, baseline, args.intrinsic_file)
    
    print(f"\n{'='*80}")
    print("✅ 完成！")
    print(f"{'='*80}")
    print(f"\n生成的文件:")
    print(f"  1. {args.output_path}")
    print(f"     → 用于深度估计系统")
    print(f"  2. {args.intrinsic_file}")
    print(f"     → FoundationStereo标准格式")
    
    print(f"\n⚠️  重要提醒:")
    print(f"  1. 基线距离 ({baseline*100:.2f} cm) 是估计值")
    print(f"     请用尺子实际测量左右相机镜头中心距离")
    print(f"  2. 如果有实测值，重新运行:")
    print(f"     python {os.path.basename(__file__)} --baseline <实测值(米)>")
    print(f"  3. 内参是根据FOV估算的，可能需要通过标定板校准获得更准确值")
    
    print(f"\n下一步:")
    print(f"  1. 验证参数:")
    print(f"     python ../scripts/test_stereo_inference.py \\")
    print(f"            --model_path ../models/stereo_lightweight.onnx \\")
    print(f"            --visualize")
    print(f"  2. 部署:")
    print(f"     python ../scripts/deploy_with_depth.py \\")
    print(f"            --camera_config {args.output_path}")
    print()


if __name__ == "__main__":
    main()
