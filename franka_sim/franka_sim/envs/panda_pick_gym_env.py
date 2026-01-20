from pathlib import Path
from typing import Any, Literal, Tuple, Dict

# import gym  #Origin
import gymnasium as gym # startear
import mujoco
import numpy as np
from gym import spaces
import time 
import cv2

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv


_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4)) # Origin
# _PANDA_HOME = np.asarray((0, -0.785, -0.2, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]]) # Origin
# _SAMPLING_BOUNDS = np.asarray([[0.35, -0.15], [0.45, 0.15]])


class PandaPickCubeGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        config = None,
        hz = 20,
    ):
        self.hz = hz
        self._action_scale = action_scale
        # render_mode = "rgb_array"
        # control_dt = 0.1
        # physics_dt = 0.01

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0, 1)
        self.image_obs = image_obs
        self.env_step = 0
        self.intervened = False

        # Caching.
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]

        if self.image_obs:
            self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
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

        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0,-1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # NOTE: gymnasium is used here since MujocoRenderer is not available in gym. It
        # is possible to add a similar viewer feature with gym, but that can be a future TODO
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        # self._viewer = MujocoRenderer(
        #     self.model,
        #     self.data,
        # )
        self._viewer = MujocoRenderer(
            self.model,
            self.data,
            # width=128,
            # height=128,
        )
        self._viewer.render(self.render_mode)

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height.
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.2

        self.env_step = 0

        obs = self._compute_observation()
        return obs, {"succeed": False}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        start_time = time.time()
        # x, y, z, grasp = action
        x, y, z, grasp = action[0], action[1], action[2], action[-1]

        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale[0]
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        # Set gripper grasp.
        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * self._action_scale[1]
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0),
                damping_ratio=4
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)
        
        
        obs = self._compute_observation()
        
        rew = self._compute_reward()

        # print("obs[state].keys() = ", obs["state"].keys())

        if (action[-1] < -0.5 and obs["state"]["gripper_pose"] > 0.9) or (
            action[-1] > 0.5 and obs["state"]["gripper_pose"] < 0.9
        ):
            grasp_penalty = -0.02
        else:
            grasp_penalty = 0.0

        # print("obs:",obs["images"]["front"])


        # terminated = self.time_limit_exceeded()
        self.env_step += 1
        terminated = False
        if self.env_step >= 500:
            terminated = True

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)
            dt = time.time() - start_time
            if self.intervened == True:
                time.sleep(max(0, (1.0 / self.hz) - dt))
            # time.sleep(max(0, (1.0 / self.hz) - dt))
            # cv2.imshow('Concatenated Image', np.hstack((obs["images"]["wrist_1"], obs["images"]["wrist_2"])))
            # cv2.waitKey(1)


        success = self._compute_success()
        if success:
            print(f'success!')
            rew = 1
        else:
            rew = 0
            pass
        # if terminated:
        #     success = True
        terminated = terminated or success

        return obs, rew, terminated, False, {"succeed": success, "grasp_penalty": grasp_penalty}

    def _compute_success(self):
        block_pos = self._data.sensor("block_pos").data
        success = block_pos[2] >= self._z_success * 0.5
        return success
    
    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array")
            )
        return rendered_frames

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        obs["state"]["panda/tcp_pos"] = tcp_pos.astype(np.float32)

        tcp_vel = self._data.sensor("2f85/pinch_vel").data
        obs["state"]["panda/tcp_vel"] = tcp_vel.astype(np.float32)

        gripper_pos = np.array(
            self._data.ctrl[self._gripper_ctrl_id] / 255, dtype=np.float32
        )
        obs["state"]["panda/gripper_pos"] = gripper_pos

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["front"], obs["images"]["wrist"] = self.render()
            
        else:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos

        

        
        # startear add
        gripper_pos = np.array(
            [self._data.ctrl[self._gripper_ctrl_id] / 255], dtype=np.float32
        )
        final_tcp_pos = self.observation_space['state']['tcp_pose'].sample()
        final_tcp_pos[:3] = tcp_pos
        final_tcp_vel = self.observation_space['state']['tcp_vel'].sample()
        final_tcp_vel[:3] = tcp_vel

        obs['state'] = {
            "tcp_pose": final_tcp_pos,
            "tcp_vel": final_tcp_vel,
            "gripper_pose": gripper_pos,
            "tcp_force": self.observation_space['state']['tcp_force'].sample(),
            "tcp_torque": self.observation_space['state']['tcp_torque'].sample(),
        }
        obs["images"]["wrist_1"], obs["images"]["wrist_2"] = obs["images"]["front"], obs["images"]["wrist"]
        obs["images"]["wrist_1"] = cv2.resize(obs["images"]["wrist_1"],(256,256))
        obs["images"]["wrist_2"] = cv2.resize(obs["images"]["wrist_2"],(128,128))
        del obs["images"]["front"]
        del obs["images"]["wrist"]
        # print(obs["images"]["wrist_1"].shape)
        # print("wrist_1",obs["images"]["wrist_1"].shape)
        # print("wrist_2",obs["images"]["wrist_2"].shape)
        return obs

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        r_close = np.exp(-20 * dist)
        r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
        rew = 0.3 * r_close + 0.7 * r_lift
        return rew


if __name__ == "__main__":
    env = PandaPickCubeGymEnv(render_mode="human",image_obs=True)
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 4))
        env.render()
    env.close()
