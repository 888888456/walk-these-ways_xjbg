"""
Deploy policy with stereo depth estimation for terrain-aware locomotion.

This script deploys a trained locomotion policy with real-time depth estimation
from belly-mounted stereo cameras.

Usage:
    python deploy_with_depth.py --label gait-conditioned-agility/2025-10-29/train \
                                  --stereo_model ../models/stereo_lightweight.onnx \
                                  --camera_config ../config/camera_params.npz \
                                  --enable_depth_viz
"""

import glob
import pickle as pkl
import lcm
import sys
import argparse
import torch
import pathlib

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner
from go1_gym_deploy.envs.lcm_agent_with_depth import LCMAgentWithDepth
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import *

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


def load_policy(logdir):
    """Load policy from checkpoint"""
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info):
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_and_run_policy(args):
    """Load policy and run deployment with depth estimation"""
    
    print(f"\n{'='*80}")
    print("Go1 Deployment with Stereo Depth Estimation")
    print(f"{'='*80}\n")
    
    # Load agent configuration
    dirs = glob.glob(f"../../runs/{args.label}/*")
    if len(dirs) == 0:
        print(f"❌ Error: No runs found at ../../runs/{args.label}/")
        print(f"   Please check the label path")
        sys.exit(1)
    
    logdir = sorted(dirs)[0]
    print(f"✓ Loading policy from: {logdir}")

    with open(logdir+"/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
    
    # Check if terrain observation is enabled
    if "terrain" not in cfg or not cfg["terrain"].get("measure_heights", False):
        print(f"\n⚠️  Warning: Policy was not trained with terrain heights!")
        print(f"   Consider retraining with terrain.measure_heights = True")
        print(f"   Depth estimation will still run but may not affect policy")
    else:
        print(f"✓ Policy trained with terrain observation")
        points_x = cfg["terrain"]["measured_points_x"]
        points_y = cfg["terrain"]["measured_points_y"]
        print(f"  - Heightmap size: {len(points_x)} x {len(points_y)} = {len(points_x)*len(points_y)} points")
    
    # Initialize state estimator
    se = StateEstimator(lc)
    
    # Initialize command profile
    control_dt = 0.02
    command_profile = RCControllerProfile(
        dt=control_dt, 
        state_estimator=se, 
        x_scale=args.max_vel, 
        y_scale=args.max_vel, 
        yaw_scale=args.max_yaw_vel
    )
    
    # Initialize hardware agent with depth estimation
    print(f"\n{'='*80}")
    print("Initializing Hardware Agent")
    print(f"{'='*80}")
    
    hardware_agent = LCMAgentWithDepth(
        cfg, 
        se, 
        command_profile,
        stereo_model_path=args.stereo_model,
        camera_config_path=args.camera_config,
        enable_depth=args.enable_depth,
        depth_inference_fps=args.depth_fps,
        enable_depth_viz=args.enable_depth_viz
    )
    
    # Start state estimator thread
    se.spin()
    
    # Wrap with history
    from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)
    
    # Load policy
    print(f"\n✓ Loading policy...")
    policy = load_policy(logdir)
    
    # Setup deployment runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    
    deployment_runner = DeploymentRunner(
        experiment_name=args.experiment_name,
        se=None,
        log_root=f"{root}/{args.experiment_name}"
    )
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)
    
    # Run deployment
    print(f"\n{'='*80}")
    print("Starting Deployment")
    print(f"{'='*80}")
    print(f"Max steps: {args.max_steps}")
    print(f"Logging: {args.logging}")
    print(f"Depth estimation: {args.enable_depth}")
    if args.enable_depth:
        print(f"  - Target FPS: {args.depth_fps}")
        print(f"  - Visualization: {args.enable_depth_viz}")
    print(f"\n{'='*80}\n")
    
    try:
        deployment_runner.run(max_steps=args.max_steps, logging=args.logging)
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    finally:
        # Clean shutdown
        print("\nShutting down...")
        hardware_agent.shutdown()
        print("✓ Shutdown complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy policy with depth estimation')
    
    # Policy configuration
    parser.add_argument('--label', type=str, 
                        default="gait-conditioned-agility/2025-10-29/train",
                        help='Policy label (path under runs/)')
    parser.add_argument('--experiment_name', type=str,
                        default="deployment_with_depth",
                        help='Experiment name for logging')
    
    # Stereo depth configuration
    parser.add_argument('--stereo_model', type=str,
                        default=None,
                        help='Path to ONNX/TRT stereo model (auto-detect if not specified)')
    parser.add_argument('--camera_config', type=str,
                        default=None,
                        help='Path to camera config (.npz file, auto-detect if not specified)')
    parser.add_argument('--enable_depth', action='store_true', default=True,
                        help='Enable depth estimation')
    parser.add_argument('--disable_depth', dest='enable_depth', action='store_false',
                        help='Disable depth estimation (for testing)')
    parser.add_argument('--depth_fps', type=int, default=20,
                        help='Target FPS for depth inference')
    parser.add_argument('--enable_depth_viz', action='store_true',
                        help='Show real-time depth visualization')
    
    # Command profile
    parser.add_argument('--max_vel', type=float, default=1.0,
                        help='Maximum linear velocity')
    parser.add_argument('--max_yaw_vel', type=float, default=1.0,
                        help='Maximum yaw velocity')
    
    # Deployment options
    parser.add_argument('--max_steps', type=int, default=10000000,
                        help='Maximum number of steps')
    parser.add_argument('--logging', action='store_true', default=True,
                        help='Enable logging')
    parser.add_argument('--no_logging', dest='logging', action='store_false',
                        help='Disable logging')
    
    args = parser.parse_args()
    
    # Run deployment
    load_and_run_policy(args)
