import os
import numpy as np
from io import TextIOWrapper

from .ikfastpy import ikfastpy
from .kinematics import euler2r, r2euler


class UR5Manager:
    def __init__(self, conf_dir: str):
        self._read_params(conf_dir)
        self.ur5_kin = ikfastpy.PyKinematics()
        self.n_joints = self.ur5_kin.getDOF()

    def _open_params(self, file: TextIOWrapper) -> np.ndarray:
        while True:
            line = file.readline()
            if not line:
                break
            if line.startswith("max_vel_a_division:"):
                p0 = float(line.split(":")[-1])
            elif line.startswith("p_err_division:"):
                p1 = float(line.split(":")[-1])
            elif line.startswith("time_sec_division:"):
                p2 = float(line.split(":")[-1])
            elif line.startswith("wait_time_division:"):
                p3 = float(line.split(":")[-1])
            elif line.startswith("max_vel_a_gripper:"):
                p4 = float(line.split(":")[-1])
            elif line.startswith("p_err_gripper:"):
                p5 = float(line.split(":")[-1])
            elif line.startswith("time_sec_gripper:"):
                p6 = float(line.split(":")[-1])
            elif line.startswith("wait_time_gripper:"):
                p7 = float(line.split(":")[-1])
            elif line.startswith("max_vel_a_move:"):
                p8 = float(line.split(":")[-1])
            elif line.startswith("p_err_move:"):
                p9 = float(line.split(":")[-1])
            # elif line.startswith('gp_err_move:'): p10 = float(line.split(':')[-1])
            elif line.startswith("time_sec_move:"):
                p10 = float(line.split(":")[-1])
            elif line.startswith("wait_time_move:"):
                p11 = float(line.split(":")[-1])
            elif line.startswith("pan:"):
                self.pan = float(line.split(":")[-1])
            elif line.startswith("tilt:"):
                self.tilt = float(line.split(":")[-1])
        # cmd_params = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]).astype(np.float64)
        cmd_params = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]).astype(np.float64)
        return cmd_params

    def _read_params(self, conf_dir: str):
        with open(os.path.join(conf_dir, "controller_params.txt"), "r") as f:
            self.cmd_params = self._open_params(f)

        with open(os.path.join(conf_dir, "return_controller_params.txt"), "r") as f:
            self.return_cmd_params = self._open_params(f)

    def solve_fk(self, pos: np.ndarray) -> np.ndarray:
        """
        input: (6,)
        output: (6,)
        """
        ee_pose = self.ur5_kin.forward(pos[:6])
        ee_pose = np.asarray(ee_pose).reshape(3, 4)
        R = ee_pose[:, :3]
        xyz = ee_pose[:, 3]
        euler = r2euler(R)
        n_pos = np.concatenate((xyz, euler), axis=0).tolist()
        # n_pos.append(pos[6])
        return np.asarray(n_pos)

    def solve_ik(self, cmd: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """
        input: (6,), (6,)
        output: (6,)
        """
        pos = cmd[0:3]  # (3,)
        theta = cmd[3:6]
        ref_joint = ref[:6]
        R = euler2r(theta)  # (3,3)
        ee_pose = np.concatenate((R, pos.reshape(3, 1)), axis=1)  # (3,4)
        joint_configs = self.ur5_kin.inverse(ee_pose.reshape(-1).tolist())
        # Exception when ik faiulre
        if joint_configs == []:
            print("warning: no IK solution found!")
            return ref

        n_solutions = int(len(joint_configs) / self.n_joints)
        joint_configs = np.asarray(joint_configs).reshape(n_solutions, self.n_joints)
        joint_diff = joint_configs - ref_joint.reshape(1, -1)
        joint_diff_mod_pi = (joint_diff + np.pi) % (2 * np.pi) - np.pi
        args_mse_joint = np.argmin(np.linalg.norm(joint_diff_mod_pi, axis=1))
        out = joint_configs[args_mse_joint]
        # out = np.concatenate((out, np.array([cmd[6]])), axis=0)
        return out
