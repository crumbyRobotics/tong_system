import os
import numpy as np
from typing import Optional
from datetime import datetime
import h5py

from .ur5 import UR5Manager


class Logger:
    def __init__(self, ur5_manager: UR5Manager, log_dir: str):
        self.ur5_manager = ur5_manager
        self.log_dir = os.path.abspath(os.path.expanduser(log_dir))

        self.reset()

    def stack(self, time: float, obs: dict[str, np.ndarray], action: np.ndarray, gaze: Optional[np.ndarray] = None):
        self.ds["time"].append(time)

        self.ds["head_state"].append(obs["state"][14:16])
        self.ds["left_state"].append(obs["state"][:7])
        self.ds["right_state"].append(obs["state"][7:14])

        left_f_state = np.concatenate((self.ur5_manager.solve_fk(obs["state"][:6]), [obs["state"][6]]), axis=0)
        right_f_state = np.concatenate((self.ur5_manager.solve_fk(obs["state"][7:13]), [obs["state"][13]]), axis=0)
        self.ds["left_f_state"].append(left_f_state)
        self.ds["right_f_state"].append(right_f_state)

        self.ds["head_hstate"].append(action[14:16])
        self.ds["left_hstate"].append(action[:7])
        self.ds["right_hstate"].append(action[7:14])
        left_f_hstate = np.concatenate((self.ur5_manager.solve_fk(action[:6]), [action[6]]), axis=0)
        right_f_hstate = np.concatenate((self.ur5_manager.solve_fk(action[7:13]), [action[13]]), axis=0)
        self.ds["left_f_hstate"].append(left_f_hstate)
        self.ds["right_f_hstate"].append(right_f_hstate)

        left_image = np.transpose(obs["image"][:3] * 255, (1, 2, 0)).astype(np.uint8)  # (H, W, 3), uint8
        right_image = np.transpose(obs["image"][:3] * 255, (1, 2, 0)).astype(np.uint8)  # (H, W, 3), uint8

        self.ds["left_img"].append(left_image)
        self.ds["right_img"].append(right_image)
        self.ds["depth_img"].append(obs["depth"].transpose((1, 2, 0)))  # (H, W, 1), uint16

        self.ds["left_sensor"].append(obs["state"][-50:-25])
        self.ds["right_sensor"].append(obs["state"][-25:])

        if gaze is not None:
            self.ds["gaze"].append(gaze.astype(np.int64))

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f.h5")
        path = os.path.join(self.log_dir, filename)
        with h5py.File(path, "w") as f:
            for key in self.ds:
                if "img" in key:
                    chunk_size = np.asarray(self.ds[key]).shape[1:]
                    chunk_size = (1, *chunk_size)
                    if len(chunk_size) > 1:
                        f.create_dataset(key, data=np.asarray(self.ds[key]), compression="lzf", chunks=chunk_size)
                    else:
                        f.create_dataset(key, data=np.asarray(self.ds[key]), compression="lzf")
                else:
                    f.create_dataset(key, data=np.asarray(self.ds[key]), compression="lzf")

        print(f"ds saved to {path}")
        self.reset()

    def reset(self):
        self.ds = {
            "time": [],
            "conf_img": [],
            "left_img": [],
            "right_img": [],
            "depth_img": [],
            "left_state": [],
            "right_state": [],
            "head_state": [],
            "left_hstate": [],
            "right_hstate": [],
            "head_hstate": [],
            "left_f_state": [],
            "right_f_state": [],
            "left_f_hstate": [],
            "right_f_hstate": [],
            "left_sensor": [],
            "right_sensor": [],
            "left_hsensor": [],
            "right_hsensor": [],
            "gaze": [],
            "macro": [],
        }
