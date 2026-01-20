"""Gym Interface for Franka/Flexiv"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.utils.rotations import euler_2_quat, quat_2_euler
import flexiv_interface
from flexiv_interface.robot_interface.flexiv_interface import FlexivInterface
from flexiv_interface.IK.pink_ik import PinkIKSolver


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break
            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.array([0.02, 0.05, 1.0])
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.2
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0


##############################################################################


class FlexivEnv(gym.Env):
    def __init__(
        self,
        hz=20,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
        use_hardware=True,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP
        
        if use_hardware:
            self.robot = FlexivInterface(robot_sn="Rizon4s-063274", enable_gripper=True)
            self.robot.start_joint_position_control()
            # self.robot.set_impedance(K=0.25)
        
        self.virtual_tcp_pose = None  # 维护虚拟的 TCP 位姿
        self.virtual_q = None         # 维护虚拟的关节角

        self.lastsentpos = None
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )
        # print("action space shape in env:", self.action_space.shape)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {"wrist_1": gym.spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8) ,
                     "wrist_2": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)}
                                # for key in config.REALSENSE_CAMERAS
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return
        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()

        if set_load:
            input("Put arm into programing mode and press enter.")
            requests.post(self.url + "set_load", json=self.config.LOAD_PARAM)
            input("Put arm into execution mode and press enter.")
            for _ in range(2):
                self._recover()
                time.sleep(1)

        if not fake_env:
            from pynput import keyboard
            self.terminate = False
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        print("Initialized Flexiv Env")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose_clipped = pose.copy()
        pose_clipped[:3] = np.clip(
            pose_clipped[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose_clipped[3:]).as_euler("xyz")

        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose_clipped[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose_clipped

    def go_to_reset(self, joint_reset=False):
        self.robot.open_gripper()
        self.robot.move_to_home_center()
        real_tcp = np.array(self.robot.get_tcp_pose())
        real_q = np.array(self.robot.get_joint_position())
        
        self.virtual_tcp_pose = real_tcp.copy()
        self.virtual_q = real_q.copy()
        
        self.solver = PinkIKSolver(
            "/home/dx/waylen/conrft/flexiv_env/resources/flexiv_Rizon4s_kinematics.urdf",
            ee_frame="flange",
            visualize=False,
            mesh_dir="/home/dx/waylen/conrft/flexiv_env/resources",
            init_q=real_q
        )
        
        self._update_currpos()
        self.robot.start_joint_impedance_control()
        self.robot.set_impedance(K=0.25)
        

    def step(self, action: np.ndarray) -> tuple:
        """Standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # print(action)
        
        xyz_delta = action[:3] * self.action_scale[0]
        self.virtual_tcp_pose[:3] += xyz_delta

        rot_action = Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
        rot_virtual = Rotation.from_quat(self.virtual_tcp_pose[3:])
        self.virtual_tcp_pose[3:] = (rot_action * rot_virtual).as_quat()

        # self.virtual_tcp_pose = self.clip_safety_box(self.virtual_tcp_pose)
        self._send_pos_command(self.virtual_tcp_pose)

        gripper_action = action[6] * self.action_scale[2]
        self._send_gripper_command(gripper_action)
        
        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))
        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)

        if (action[-1] < -0.5 and ob["state"]["gripper_pose"] > 0.9) or (
            action[-1] > 0.5 and ob["state"]["gripper_pose"] < 0.9
        ):
            grasp_penalty = -0.02
        else:
            grasp_penalty = 0.0
        
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward, "grasp_penalty": grasp_penalty}

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T  @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            return False

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}
        
        if self.cap is None: return images

        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)
            except queue.Empty:
                print(f"Camera {key} queue empty")
                continue
            except Exception as e:
                print(f"Camera {key} error: {e}")
                continue

        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle!=0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True
            
        self.go_to_reset(joint_reset=joint_reset)
        self.curr_path_length = 0
        
        self._update_currpos() 
        obs = self._get_obs()
        self.terminate = False
        
        time.sleep(10)
        print("recording start!!!!!!!!!!!!!!!!!")
        return obs, {"succeed": False}
    
    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        if camera_key in frame_dict:
                            video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        if self.cap is not None:
            self.close_cameras()
        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            try:
                cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
                self.cap[cam_name] = cap
            except Exception as e:
                print(f"Error initializing camera {cam_name}: {e}")

    def close_cameras(self):
        try:
            if self.cap:
                for cap in self.cap.values():
                    cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _send_pos_command(self, pos: np.ndarray):
        target_q = self.solver.solve_ik_pose(pos[:3], pos[3:])
        self.virtual_q = target_q.copy()
        self.robot.set_joint_target_positions(target_q)
        self.lastsentpos = pos

    def _send_gripper_command(self, pos: float, mode="binary"):
        if mode == "binary":
            if pos <= -0.5:
                self.robot.close_gripper()
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif pos >= 0.5:
                self.robot.open_gripper()
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            else: 
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_currpos(self):

        if self.virtual_tcp_pose is not None:
            self.currpos = self.virtual_tcp_pose.copy()
        else:
            self.currpos = np.array(self.robot.get_tcp_pose())

        if self.virtual_q is not None:
            self.q = self.virtual_q.copy()
        else:
            self.q = np.array(self.robot.get_joint_position())

        self.currvel = np.array(self.robot.get_tcp_velocity())
        
        external_force = self.robot.get_external_wrench_tcp()
        self.currforce = np.array(external_force[:3])
        self.currtorque = np.array(external_force[3:])
        
        self.dq = np.array(self.robot.get_joint_velocity())
        self.curr_gripper_pos = np.array(self.robot.get_gripper_state())

    update_currpos = _update_currpos

    def _get_obs(self) -> dict:
        """
        Get current observation.
        """
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos, # 现在这里是纯净的虚拟目标
            "tcp_vel": self.currvel,  # 真实速度
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce, # 真实力
            "tcp_torque": self.currtorque, # 真实力矩
        }
        return copy.deepcopy(dict(images=images, state=state_observation))
    
    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        self.robot.stop_control()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()