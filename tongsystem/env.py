import os
import time
from copy import deepcopy
import numpy as np
import gym
from abc import ABC, abstractmethod
import cv2

from .communication import SocketManager
from .ur5 import UR5Manager
from .logger import Logger


class AbstTongSystem(gym.Env, ABC):
    metadata = {"render.modes": ["human", "rgbd_array"]}

    def __init__(
        self,
        state_type: str,
        action_type: str,
        image_width: int,
        image_height: int,
        init_poses: np.ndarray,
        log_dir: str,
        dt: float = 0.02,
        conf_dir: str = None,
    ):
        super().__init__()

        self._latest_obs = None
        self._latest_ation = None

        self.last_step_time = None

        self.socket = SocketManager()
        _ = self.socket.receive()  # For init zmq

        conf_dir = conf_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, "controller_params")
        self.ur5_manager = UR5Manager(conf_dir)

        self.image_width = image_width  # image_height
        self.image_height = image_height  # image_width
        self.init_poses = np.asarray(
            init_poses
        )  # (N, joint_angle:7+7+2) init trajectory: current -> init_poses[0] -> init_poses[1] -> ... -> init_poses[N]
        self.dt = dt  # sec

        self.state_type = state_type
        if state_type not in ["pos", "joint_angle"]:
            raise ValueError("state_type is invalid value")
        self.action_type = action_type
        if action_type not in ["state", "state_diff"]:
            raise ValueError("action_type is invalid value")

        self.state_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(44,)
        )  # (right_joint_angle + left_joint_angle + head_angle:7+7+2, vels: 7+7+2, force_sensor:6+6)
        self.image_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3 * 2, self.image_height, self.image_width)
        )  # (C*2, H, W), RGB (since 2023.10.17) *gymEnv is (H, W, C)
        self.depth_space = gym.spaces.Box(low=0, high=np.inf, shape=(1, self.image_height, self.image_width))  # distance (meter)
        self.observation_space = gym.spaces.Dict({"state": self.state_space, "image": self.image_space, "depth": self.depth_space})
        self.action_space = None

        self.logger = Logger(self.ur5_manager, log_dir)

    def reset(self, *args, **kwargs) -> dict[str, np.ndarray]:
        for i in range(len(self.init_poses)):
            for j in range(20):  # Repeat the command to make sure the robot has reached to the init pose
                self._set_cmd(self.init_poses[i], reset=(i == 0 and j == 0))  # NOTE Send reset signal only for the first time
                obs = self._get_obs()  # for zmq
            time.sleep(0.5)

        self.logger.reset()

        self.last_step_time = None

        # Conversion of state w.r.t self.state_type
        if self.state_type == "pos":
            obs["state"][:16] = self.solve_fk(obs["state"][:16])

        return obs

    @abstractmethod
    def step(self, action: np.ndarray, gaze: np.ndarray = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        # NOTE: 'gaze' is used for logging the gaze data
        pass

    def freeze_step(self, gaze: np.ndarray = None) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        # Do nothing but just step env
        action = self._latest_obs["state"][:16]
        action[6], action[13] = (
            np.round(action[6]),
            np.round(action[13]),
        )  # Round gripper angle to stabilize it: otherwise, gripper will gradually closed or opened because of the positive feedback

        return self._step(action, gaze)

    def render(self, mode: str = "human"):
        if mode == "rgbd_array":
            # Return RGB-D array suitable for video
            image = (
                np.concatenate((self._latest_obs["image"][:3], self._latest_obs["image"][3:6]), axis=2).transpose(1, 2, 0) * 255.0
            )  # (H, W, C), [0, 255]
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)  # RGB -> BGR
            depth = self._latest_obs["depth"].squeeze(0)
            depth = np.clip(depth, 0, 1500)
            depth = (depth / depth.max() * 255).astype(np.uint8)
            return {"image": image, "depth": depth}
        elif mode == "human":
            # Pop up a window and render
            image = (
                np.concatenate((self._latest_obs["image"][:3], self._latest_obs["image"][3:6]), axis=2).transpose(1, 2, 0) * 255.0
            )  # (H, W, C), [0, 255]
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)  # RGB -> BGR
            cv2.imshow("image", image)
            cv2.waitKey(1)
            depth = self._latest_obs["depth"].squeeze(0)
            depth = np.clip(depth, 0, 1500)
            depth = (depth / depth.max() * 255).astype(np.uint8)
            cv2.imshow("depth", depth)
            cv2.waitKey(1)
        else:
            super().render(mode=mode)  # just raise an exception

    def get_world_state(self) -> dict[str, dict[str, np.ndarray]]:
        return self._latest_world_state

    def _step(self, action: np.ndarray, gaze: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        assert isinstance(action, np.ndarray) and action.shape == (16,)

        # if self.last_step_time and time.time() - self.last_step_time > 1e-1:
        #     print("[TongSystem] Warning: step() function was called more than 0.1 second after the last observation.")

        self._set_cmd(action, reset=False)

        next_obs = self._get_obs()
        reward = 0
        done = False
        info = None

        self.last_step_time = time.time()

        # Log
        self.logger.stack(self.last_step_time, next_obs, action, gaze)

        # Conversion of state w.r.t self.state_type
        if self.state_type == "pos":
            next_obs["state"][:16] = self.solve_fk(next_obs["state"][:16])

        return next_obs, reward, done, info

    def _get_obs(self) -> dict[str, np.ndarray]:
        [posN, posL, posR, velN, velL, velR, fsL, fsR, SbSResultL, SbSResultR, SbSResultD], world_state = self.socket.receive()

        obs = {}
        obs["state"] = np.concatenate((posL, posR, posN, velL, velR, velN, fsL, fsR))  # (82=16+16+50,)

        SbSResultL = cv2.resize(SbSResultL[:, :, :3], (self.image_width, self.image_height))
        SbSResultR = cv2.resize(SbSResultR[:, :, :3], (self.image_width, self.image_height))
        SbSResultL = cv2.cvtColor(SbSResultL, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        SbSResultR = cv2.cvtColor(SbSResultR, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        obs["image"] = np.concatenate((SbSResultL, SbSResultR), axis=2).transpose(2, 0, 1) / 255.0  # (C=6, H, W), [0, 1]

        SbSResultD = cv2.resize(SbSResultD[:, :, 0], (self.image_width, self.image_height))
        obs["depth"] = SbSResultD.reshape(1, *SbSResultD.shape)  # (C=1, H, W), [0mm, inf), depth image from left camera view

        self._latest_obs = deepcopy(obs)
        self._latest_world_state = deepcopy(world_state)

        return obs

    def _set_cmd(self, action: np.ndarray, reset: bool):
        """
        action: joint angle of next step
        """
        assert isinstance(action, np.ndarray) and action.shape == (16,)

        self._latest_action = action.copy()

        cmdL, cmdR, cmdN = action[:7], action[7:14], action[14:]
        self.socket.send(self.ur5_manager.return_cmd_params, cmdN, cmdL, cmdR, reset)
        time.sleep(self.dt)  # Robot state changing

    def save_log(self):
        self.logger.save()

    def reset_log(self):
        self.logger.reset()

    def joint_angles(self, action: np.ndarray, last_state: np.ndarray) -> np.ndarray:
        """
        Input
            action: action of (self.state_type, self.action_type)
            last_state: observed joint angles of last step
        Output
            joint angles
        """
        assert action.shape == last_state.shape

        if self.action_type == "state":  # a_t = s_t+1
            if self.state_type == "pos":
                return self.solve_ik(action, last_state)
            elif self.state_type == "joint_angle":
                return action
            else:
                raise NotImplementedError
        elif self.action_type == "state_diff":  # a_t = s_t+1 - s_t
            if self.state_type == "pos":
                last_f_state = self.solve_fk(last_state)
                next_f_state = last_f_state + action
                return self.solve_ik(next_f_state, last_state)
            elif self.state_type == "joint_angle":
                return last_state + action
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def solve_fk(self, state: np.ndarray) -> np.ndarray:
        return np.concatenate((self.ur5_manager.solve_fk(state[:6]), [state[6]], self.ur5_manager.solve_fk(state[7:13]), state[13:]))

    def solve_ik(self, f_state: np.ndarray, ref_f_state: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                self.ur5_manager.solve_ik(f_state[:6], ref_f_state[:6]),
                [f_state[6]],
                self.ur5_manager.solve_ik(f_state[7:13], ref_f_state[7:13]),
                f_state[13:],
            )
        )


class TongSystem(AbstTongSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,))  # (right_arm + left_arm + head:7+7+2)

    def step(self, action: np.ndarray, gaze: np.ndarray = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        assert isinstance(action, np.ndarray) and action.shape == self.action_space.shape

        last_state = self._latest_obs["state"][:16]
        action = self.joint_angles(action, last_state)

        return self._step(action, gaze)


class TongSystemFixedNeck(AbstTongSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,))  # (right_arm + left_arm + head:7+7+2)

    def step(self, action: np.ndarray, gaze: np.ndarray = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        assert isinstance(action, np.ndarray) and action.shape == self.action_space.shape

        last_state = self._latest_obs["state"][:14]
        action = self.joint_angles(action, last_state)

        fixed_pose = self.init_poses[-1]
        action = np.concatenate((action, fixed_pose[14:]))

        return self._step(action, gaze)


class AbstTongSystemUseOnlySingleArm(AbstTongSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))  # (left_arm:7)

    def solve_fk(self, state: np.ndarray) -> np.ndarray:
        if state.shape[0] == 7:
            return np.concatenate((self.ur5_manager.solve_fk(state[:6]), [state[6]]))
        else:
            return super().solve_fk(state)

    def solve_ik(self, f_state: np.ndarray, ref_f_state: np.ndarray) -> np.ndarray:
        if f_state.shape[0] == 7:
            return np.concatenate((self.ur5_manager.solve_ik(f_state[:6], ref_f_state[:6]), [f_state[6]]))
        else:
            return super().solve_ik(f_state, ref_f_state)


class TongSystemUseOnlyLeftArm(AbstTongSystemUseOnlySingleArm):
    def step(self, action: np.ndarray, gaze: np.ndarray = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        assert isinstance(action, np.ndarray) and action.shape == self.action_space.shape

        last_state = self._latest_obs["state"][:7]
        action = self.joint_angles(action, last_state)

        fixed_pose = self.init_poses[-1]
        action = np.concatenate((action, fixed_pose[7:]))

        return self._step(action, gaze)


class TongSystemUseOnlyRightArm(AbstTongSystemUseOnlySingleArm):
    def step(self, action: np.ndarray, gaze: np.ndarray = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        assert isinstance(action, np.ndarray) and action.shape == self.action_space.shape

        last_state = self._latest_obs["state"][7:14]
        action = self.joint_angles(action, last_state)

        fixed_pose = self.init_poses[-1]
        action = np.concatenate((fixed_pose[:7], action, fixed_pose[14:]))

        return self._step(action, gaze)
