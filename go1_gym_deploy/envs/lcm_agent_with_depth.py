"""
Enhanced LCM Agent with stereo depth estimation for terrain-aware locomotion.

This module extends the base LCM Agent to include:
- Real-time stereo depth estimation from belly cameras
- Terrain heightmap generation
- Background thread processing (non-blocking LCM callbacks)
"""

import time
import lcm
import numpy as np
import torch
import cv2
import os
import sys

from go1_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from go1_gym_deploy.envs.stereo_depth_estimator import StereoDepthEstimator

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


def class_to_dict(obj) -> dict:
    """Convert class object to dictionary"""
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class LCMAgentWithDepth():
    """
    Enhanced LCM Agent with stereo depth estimation.
    
    Key features:
    - Non-blocking camera image processing
    - Background depth inference
    - Automatic terrain heightmap updates
    - Maintains compatibility with existing deployment pipeline
    """
    
    def __init__(self, cfg, se, command_profile, 
                 stereo_model_path=None,
                 camera_config_path=None,
                 enable_depth=True,
                 depth_inference_fps=20,
                 enable_depth_viz=False):
        """
        Args:
            cfg: Robot configuration
            se: State estimator
            command_profile: Command profile
            stereo_model_path: Path to ONNX/TRT stereo model
            camera_config_path: Path to camera calibration config
            enable_depth: Enable depth estimation (can disable for testing)
            depth_inference_fps: Target FPS for depth inference
            enable_depth_viz: Show real-time depth visualization
        """
        if not isinstance(cfg, dict):
            cfg = class_to_dict(cfg)
        self.cfg = cfg
        self.se = se
        self.command_profile = command_profile

        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        self.timestep = 0

        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_envs = 1
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.num_commands = self.cfg["commands"]["num_commands"]
        self.device = 'cpu'

        if "obs_scales" in self.cfg.keys():
            self.obs_scales = self.cfg["obs_scales"]
        else:
            self.obs_scales = self.cfg["normalization"]["obs_scales"]

        self.commands_scale = np.array(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"],
             self.obs_scales["ang_vel"], self.obs_scales["body_height_cmd"], 1, 1, 1, 1, 1,
             self.obs_scales["footswing_height_cmd"], self.obs_scales["body_pitch_cmd"],
             self.obs_scales["body_roll_cmd"], self.obs_scales["stance_width_cmd"],
             self.obs_scales["stance_length_cmd"], self.obs_scales["aux_reward_cmd"], 1, 1, 1, 1, 1, 1
             ])[:self.num_commands]

        # Joint configuration
        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", ]
        self.default_dof_pos = np.array([self.cfg["init_state"]["default_joint_angles"][name] for name in joint_names])
        try:
            self.default_dof_pos_scale = np.array([self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"]])
        except KeyError:
            self.default_dof_pos_scale = np.ones(12)
        self.default_dof_pos = self.default_dof_pos * self.default_dof_pos_scale

        # PD gains
        self.p_gains = np.zeros(12)
        self.d_gains = np.zeros(12)
        for i in range(12):
            joint_name = joint_names[i]
            found = False
            for dof_name in self.cfg["control"]["stiffness"].keys():
                if dof_name in joint_name:
                    self.p_gains[i] = self.cfg["control"]["stiffness"][dof_name]
                    self.d_gains[i] = self.cfg["control"]["damping"][dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg["control"]["control_type"] in ["P", "V"]:
                    print(f"PD gain of joint {joint_name} were not defined, setting them to zero")

        print(f"p_gains: {self.p_gains}")

        # State variables
        self.commands = np.zeros((1, self.num_commands))
        self.actions = torch.zeros(12)
        self.last_actions = torch.zeros(12)
        self.gravity_vector = np.zeros(3)
        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        self.body_linear_vel = np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(12)
        self.joint_vel_target = np.zeros(12)
        self.torques = np.zeros(12)
        self.contact_state = np.ones(4)

        self.joint_idxs = self.se.joint_idxs

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float)

        if "obs_scales" in self.cfg.keys():
            self.obs_scales = self.cfg["obs_scales"]
        else:
            self.obs_scales = self.cfg["normalization"]["obs_scales"]

        self.is_currently_probing = False
        
        # ==================== Depth Estimation Setup ====================
        self.enable_depth = enable_depth
        self.depth_estimator = None
        
        if self.enable_depth:
            if stereo_model_path is None:
                # Try to find model automatically
                model_dir = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    '../models'
                )
                stereo_model_path = os.path.join(model_dir, 'stereo_lightweight.onnx')
                
                if not os.path.exists(stereo_model_path):
                    print(f"⚠️  Warning: Stereo model not found at {stereo_model_path}")
                    print(f"   Depth estimation will be disabled.")
                    print(f"   Run: python go1_gym_deploy/scripts/export_lightweight_stereo_onnx.py")
                    self.enable_depth = False
            
            if self.enable_depth:
                # Load camera parameters
                camera_params = self._load_camera_params(camera_config_path)
                
                # Get terrain config
                terrain_cfg = self.cfg.get("terrain", None)
                
                # Initialize depth estimator
                print(f"\n{'='*80}")
                print("Initializing Stereo Depth Estimator")
                print(f"{'='*80}")
                
                self.depth_estimator = StereoDepthEstimator(
                    model_path=stereo_model_path,
                    camera_params=camera_params,
                    terrain_cfg=terrain_cfg,
                    use_tensorrt=stereo_model_path.endswith('.trt'),
                    inference_fps=depth_inference_fps,
                    enable_visualization=enable_depth_viz
                )
                
                print(f"{'='*80}")
                print("✅ Depth estimation initialized successfully!")
                print(f"{'='*80}\n")
        
        # Initialize measured heights
        if "terrain" in self.cfg.keys() and self.cfg["terrain"]["measure_heights"]:
            if self.enable_depth and self.depth_estimator is not None:
                # Will be updated by depth estimator
                self.measured_heights = self.depth_estimator.get_measured_heights()
            else:
                # Default flat terrain
                points_x = self.cfg["terrain"]["measured_points_x"]
                points_y = self.cfg["terrain"]["measured_points_y"]
                self.measured_heights = np.zeros((1, len(points_x) * len(points_y)))
        else:
            self.measured_heights = np.zeros((1, 187))  # Default size

    def _load_camera_params(self, config_path):
        """Load camera calibration parameters"""
        if config_path is None:
            # Try to find config automatically
            config_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                '../config'
            )
            config_path = os.path.join(config_dir, 'camera_params.npz')
        
        if os.path.exists(config_path):
            print(f"✓ Loading camera parameters from: {config_path}")
            data = np.load(config_path)
            camera_params = {
                'K': data['K'],
                'baseline': float(data['baseline'])
            }
        else:
            print(f"⚠️  Warning: Camera config not found at {config_path}")
            print(f"   Using default parameters (may not be accurate)")
            # Default parameters (you should calibrate your cameras!)
            camera_params = {
                'K': np.array([
                    [200.0, 0.0, 50.0],
                    [0.0, 200.0, 58.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32),
                'baseline': 0.063  # meters
            }
        
        print(f"  - Focal length: {camera_params['K'][0, 0]:.1f}")
        print(f"  - Baseline: {camera_params['baseline']*100:.1f} cm")
        
        return camera_params

    def set_probing(self, is_currently_probing):
        self.is_currently_probing = is_currently_probing

    def get_obs(self):
        """Get observations including terrain heightmap"""
        
        self.gravity_vector = self.se.get_gravity_vector()
        cmds, reset_timer = self.command_profile.get_command(self.timestep * self.dt, probe=self.is_currently_probing)
        self.commands[:, :] = cmds[:self.num_commands]
        if reset_timer:
            self.reset_gait_indices()
        
        self.dof_pos = self.se.get_dof_pos()
        self.dof_vel = self.se.get_dof_vel()
        self.body_linear_vel = self.se.get_body_linear_vel()
        self.body_angular_vel = self.se.get_body_angular_vel()

        ob = np.concatenate((self.gravity_vector.reshape(1, -1),
                             self.commands * self.commands_scale,
                             (self.dof_pos - self.default_dof_pos).reshape(1, -1) * self.obs_scales["dof_pos"],
                             self.dof_vel.reshape(1, -1) * self.obs_scales["dof_vel"],
                             torch.clip(self.actions, -self.cfg["normalization"]["clip_actions"],
                                        self.cfg["normalization"]["clip_actions"]).cpu().detach().numpy().reshape(1, -1)
                             ), axis=1)

        if self.cfg["env"]["observe_two_prev_actions"]:
            ob = np.concatenate((ob,
                            self.last_actions.cpu().detach().numpy().reshape(1, -1)), axis=1)

        if self.cfg["env"]["observe_clock_inputs"]:
            ob = np.concatenate((ob,
                            self.clock_inputs), axis=1)

        if self.cfg["env"]["observe_vel"]:
            ob = np.concatenate(
                (self.body_linear_vel.reshape(1, -1) * self.obs_scales["lin_vel"],
                 self.body_angular_vel.reshape(1, -1) * self.obs_scales["ang_vel"],
                 ob), axis=1)

        if self.cfg["env"]["observe_only_lin_vel"]:
            ob = np.concatenate(
                (self.body_linear_vel.reshape(1, -1) * self.obs_scales["lin_vel"],
                 ob), axis=1)

        if self.cfg["env"]["observe_yaw"]:
            heading = self.se.get_yaw()
            ob = np.concatenate((ob, heading.reshape(1, -1)), axis=-1)

        self.contact_state = self.se.get_contact_state()
        if "observe_contact_states" in self.cfg["env"].keys() and self.cfg["env"]["observe_contact_states"]:
            ob = np.concatenate((ob, self.contact_state.reshape(1, -1)), axis=-1)

        # ==================== Terrain Heights ====================
        if "terrain" in self.cfg.keys() and self.cfg["terrain"]["measure_heights"]:
            if self.enable_depth and self.depth_estimator is not None:
                # Get latest heightmap from depth estimator
                self.measured_heights = self.depth_estimator.get_measured_heights()
            
            # Scale and clip heights
            robot_height = 0.25
            heights = np.clip(robot_height - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales["height_measurements"]
            ob = np.concatenate((ob, heights), axis=1)

        return torch.tensor(ob, device=self.device).float()

    def get_privileged_observations(self):
        return None

    def publish_action(self, action, hard_reset=False):
        """Publish action to robot"""
        command_for_robot = pd_tau_targets_lcmt()
        self.joint_pos_target = \
            (action[0, :12].detach().cpu().numpy() * self.cfg["control"]["action_scale"]).flatten()
        self.joint_pos_target[[0, 3, 6, 9]] *= self.cfg["control"]["hip_scale_reduction"]
        self.joint_pos_target = self.joint_pos_target
        self.joint_pos_target += self.default_dof_pos
        joint_pos_target = self.joint_pos_target[self.joint_idxs]
        self.joint_vel_target = np.zeros(12)

        command_for_robot.q_des = joint_pos_target
        command_for_robot.qd_des = self.joint_vel_target
        command_for_robot.kp = self.p_gains
        command_for_robot.kd = self.d_gains
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0

        if hard_reset:
            command_for_robot.id = -1

        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (self.joint_vel_target - self.dof_vel) * self.d_gains

        lc.publish("pd_plustau_targets", command_for_robot.encode())

    def reset(self):
        self.actions = torch.zeros(12)
        self.time = time.time()
        self.timestep = 0
        return self.get_obs()

    def reset_gait_indices(self):
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)

    def step(self, actions, hard_reset=False):
        """Step the environment"""
        clip_actions = self.cfg["normalization"]["clip_actions"]
        self.last_actions = self.actions[:]
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        self.publish_action(self.actions, hard_reset=hard_reset)
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0: 
            print(f'frq: {1 / (time.time() - self.time)} Hz')
            
            # Print depth estimation stats if enabled
            if self.enable_depth and self.depth_estimator is not None:
                stats = self.depth_estimator.get_stats()
                print(f'  Depth: {stats["fps"]:.1f} FPS, {stats["last_inference_time"]*1000:.1f}ms')
        
        self.time = time.time()
        obs = self.get_obs()

        # Get camera images
        images = {'front': self.se.get_camera_front(),
                  'bottom': self.se.get_camera_bottom(),
                  'rear': self.se.get_camera_rear(),
                  'left': self.se.get_camera_left(),
                  'right': self.se.get_camera_right()
                  }
        
        # ==================== Update Depth Estimator ====================
        # Check if we have stereo pair from bottom cameras (left/right)
        if self.enable_depth and self.depth_estimator is not None:
            if images['left'] is not None and images['right'] is not None:
                # Update depth estimator (non-blocking)
                self.depth_estimator.update_images(images['left'], images['right'])
                
                # Update current robot pose
                pose = np.array([
                    0, 0, 0,  # x, y, z (world position, usually from state estimator)
                    self.gravity_vector[0],  # roll (from gravity)
                    self.gravity_vector[1],  # pitch (from gravity)
                    0  # yaw (from state estimator if available)
                ])
                self.depth_estimator.update_pose(pose)
        
        # Downscale images for logging
        downscale_factor = 2
        temporal_downscale = 3

        for k, v in images.items():
            if images[k] is not None:
                images[k] = cv2.resize(images[k], dsize=(images[k].shape[0]//downscale_factor, images[k].shape[1]//downscale_factor), interpolation=cv2.INTER_CUBIC)
            if self.timestep % temporal_downscale != 0:
                images[k] = None

        infos = {"joint_pos": self.dof_pos[np.newaxis, :],
                 "joint_vel": self.dof_vel[np.newaxis, :],
                 "joint_pos_target": self.joint_pos_target[np.newaxis, :],
                 "joint_vel_target": self.joint_vel_target[np.newaxis, :],
                 "body_linear_vel": self.body_linear_vel[np.newaxis, :],
                 "body_angular_vel": self.body_angular_vel[np.newaxis, :],
                 "contact_state": self.contact_state[np.newaxis, :],
                 "clock_inputs": self.clock_inputs[np.newaxis, :],
                 "body_linear_vel_cmd": self.commands[:, 0:2],
                 "body_angular_vel_cmd": self.commands[:, 2:],
                 "privileged_obs": None,
                 "camera_image_front": images['front'],
                 "camera_image_bottom": images['bottom'],
                 "camera_image_rear": images['rear'],
                 "camera_image_left": images['left'],
                 "camera_image_right": images['right'],
                 "measured_heights": self.measured_heights,  # Add heightmap to infos
                 }

        self.timestep += 1
        return obs, None, None, infos
    
    def shutdown(self):
        """Clean shutdown"""
        if self.enable_depth and self.depth_estimator is not None:
            self.depth_estimator.stop()
