import gymnasium as gym
import os
import jax
import jax.numpy as jnp
import numpy as np
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
    # KeyBoardIntervention2
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func
from flexiv_env.envs.flexiv_env import FlexivEnv
from experiments.config import DefaultTrainingConfig

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "141722072857",
            "dim": (640, 480),
        },
        "wrist_2": {
            "serial_number": "409122272714",
            "dim": (640,480),
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[:, :],
        "wrist_2": lambda img: img[:, :],
    }
    TARGET_POSE = np.array([0.60539,0.3526,0.28206 ,-np.pi, 0, np.pi])
    GRASP_POSE = np.array([0.60539,0.3526,0.28206 ,-np.pi, 0, np.pi])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.0, 0.1, 0.08, 0.02, 0.02, 0.2])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.17, 0.1, 0.0, 0.02, 0.02, 0.2])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.02, 0.04, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1500
    checkpoint_period = 1500
    steps_per_update = 50
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    encoder_type = "resnet-pretrained"
    # setup_mode = "single-arm-fixed-gripper"
    setup_mode = "single-arm-learned-gripper"
    reward_neg = -0.05
    task_desc = "pick up the red cube and put it on the green cube."
    octo_path = "/home/dx/waylen/conrft/octo-model"
    batch_size = 128

    def get_environment(self, fake_env=False, save_video=False, classifier=False,stack_obs_num=1, use_hardware=True):
        env = FlexivEnv(fake_env=fake_env,save_video=save_video,config=EnvConfig(),use_hardware=use_hardware)
        # env = GripperCloseEnv(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                pred_condition = sigmoid(classifier(obs)) > 0.5
                state_condition = obs['state'][0, 6] > 0.04
                result = (pred_condition & state_condition).astype(int)
                return int(result.item()) 
                # return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        if not fake_env:
            env = SpacemouseIntervention(env)
            # env = KeyBoardIntervention2(env)
        return env

import glfw
import gymnasium as gym
from pynput import keyboard
import threading


class KeyBoardIntervention2(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        # ----------------------------
        # action / gripper config
        # ----------------------------
        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.left, self.right = False, False
        self.action_indices = action_indices

        self.gripper_state = 'close'
        self.intervened = False
        self.action_length = 0.3

        self.current_action = np.zeros(6, dtype=np.float64)
        self.flag = False

        self.key_states = {
            'w': False,
            'a': False,
            's': False,
            'd': False,
            'j': False,
            'k': False,
            'l': False,
            ';': False,
        }

        # ----------------------------
        # start keyboard listener
        # ----------------------------
        self._start_keyboard_listener()

    # =====================================================
    # Keyboard listener (replace GLFW)
    # =====================================================
    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                k = key.char
            except AttributeError:
                return

            if k in self.key_states:
                print(f"{k} is pressed")
                self.key_states[k] = True

            if k == 'l':
                print("l is pressed")
                self.flag = True

            if k == ';':
                self.intervened = not self.intervened
                if hasattr(self.env, "intervened"):
                    self.env.intervened = self.intervened
                print(f"[Keyboard] Intervention toggled: {self.intervened}")

            self._update_current_action()

        def on_release(key):
            try:
                k = key.char
            except AttributeError:
                return

            if k in self.key_states:
                self.key_states[k] = False

            self._update_current_action()

        self.listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.listener.daemon = True
        self.listener.start()

    # =====================================================
    # Action computation
    # =====================================================
    def _update_current_action(self):
        self.current_action = np.array([
            int(self.key_states['w']) - int(self.key_states['s']),
            int(self.key_states['a']) - int(self.key_states['d']),
            int(self.key_states['j']) - int(self.key_states['k']),
            0,
            0,
            0,
        ], dtype=np.float64)

        self.current_action *= self.action_length

    def action(self, action: np.ndarray):
        expert_a = self.current_action.copy()

        # ----------------------------
        # gripper control
        # ----------------------------
        if self.gripper_enabled:
            if self.flag and self.gripper_state == 'open':
                self.gripper_state = 'close'
                self.flag = False
            elif self.flag and self.gripper_state == 'close':
                self.gripper_state = 'open'
                self.flag = False

            gripper_action = (
                np.random.uniform(0.9, 1.0, size=(1,))
                if self.gripper_state == 'close'
                else np.random.uniform(-1.0, -0.9, size=(1,))
            )
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        # ----------------------------
        # filter action dims
        # ----------------------------
        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if self.intervened:
            return expert_a, True
        else:
            return action, False

    # =====================================================
    # Gym interface
    # =====================================================
    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)

        if replaced:
            info["intervene_action"] = new_action

        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.gripper_state = 'open'
        self.current_action[:] = 0.0
        return obs, info
