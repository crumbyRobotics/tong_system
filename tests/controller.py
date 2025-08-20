import numpy as np

import tongsystem
from tongsystem.env import AbstTongSystemUseOnlySingleArm


class TongPickAndPlaceController:
    def __init__(self, env: AbstTongSystemUseOnlySingleArm, state_type: str, action_type: str, control_arm: str):
        self.state_type = state_type
        self.action_type = action_type
        assert state_type in ["pos", "joint_angle"] and action_type in ["state", "state_diff"]

        self.control_left = control_arm == "left"

        self.env = env

        self.transform = tongsystem.left_ee2world if self.control_left else tongsystem.right_ee2world
        self.inv_transform = np.linalg.pinv(self.transform)

    def reset(self, init_robot_state: np.ndarray, init_box_pos: np.ndarray, goal_box_pos: np.ndarray):
        self.mode = 0  # 0: Reach, 1: grasp, 2: up

        init_robot_state = init_robot_state[:7] if self.control_left else init_robot_state[7:14]  # (7,)
        if self.state_type == "joint_angle":
            self.init_f_state = self.env.solve_fk(init_robot_state)
        else:
            self.init_f_state = init_robot_state

        self.init_box_pos = init_box_pos  # (3,)
        self.goal_box_pos = goal_box_pos  # (3,)

        if self.control_left:
            self.global_reaching_robot_hand_pos = self.init_box_pos + np.array([0.0, -0.04 + np.random.uniform(-0.03, 0.03), 0.0])
        else:
            self.global_reaching_robot_hand_pos = self.init_box_pos + np.array([0.0, 0.04 + np.random.uniform(-0.03, 0.03), 0.0])

        self.lift_up_box_pos = (self.init_box_pos + goal_box_pos) / 2.0 + np.array(
            [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), 0.15 + np.random.uniform(-0.07, 0.02)]
        )

        self.prev_gripper_angle = self.init_f_state[6]

    def control(self, robot_state: np.ndarray, box_pos: np.ndarray, thresh: list = [1e-2, 1.5e-2, 2e-2]) -> tuple[np.ndarray, bool]:
        end = False

        robot_state = robot_state[:7] if self.control_left else robot_state[7:14]  # (7,)
        if self.state_type == "joint_angle":
            f_state = self.env.solve_fk(robot_state)
        else:
            f_state = robot_state
        target_f_state = f_state.copy()

        def transform(trans_mat, pos):
            pos = pos.reshape((3, 1))
            pos = np.concatenate((pos, np.array([[1]])), axis=0)
            return np.matmul(trans_mat, pos).reshape(-1)[:3]

        global_f_state = transform(self.inv_transform, f_state[:3])
        global_robot_hand_pos = global_f_state[:3]

        if self.mode == 0:
            # Reaching a box
            target_global_robot_hand_pos = global_robot_hand_pos + (self.global_reaching_robot_hand_pos - global_robot_hand_pos) * 0.1

            err = np.linalg.norm((global_robot_hand_pos - self.global_reaching_robot_hand_pos))
            if err < thresh[0]:
                self.mode = 1

        elif self.mode == 1:
            # Grasping the box
            target_global_robot_hand_pos = global_robot_hand_pos + (self.global_reaching_robot_hand_pos - global_robot_hand_pos) * 0.1
            target_f_state[6] = f_state[6] + (-2 if self.control_left else 2)  # close gripper with 2 deg per step
            if abs(f_state[6]) < 30 and abs(f_state[6] - self.prev_gripper_angle) < 1e-2:
                self.mode = 2

        elif self.mode == 2:
            # Lifting up the box
            target_global_robot_hand_pos = global_robot_hand_pos + (self.lift_up_box_pos - box_pos) * 0.1
            target_f_state[6] = 10 if self.control_left else -10

            err = np.linalg.norm((box_pos - self.lift_up_box_pos))
            if err < thresh[1]:
                self.mode = 3

        elif self.mode == 3:
            # Moving the box to the goal
            if np.linalg.norm((self.goal_box_pos - box_pos)) > 2e-2:
                target_global_robot_hand_pos = global_robot_hand_pos + (self.goal_box_pos - box_pos) * 0.1
            else:
                target_global_robot_hand_pos = (
                    global_robot_hand_pos + (self.goal_box_pos - box_pos) * 0.1 / np.linalg.norm((self.goal_box_pos - box_pos)) * 2e-2
                )
            target_f_state[6] = 10 if self.control_left else -10

            err = np.linalg.norm((box_pos - self.goal_box_pos))
            if err < thresh[2]:
                self.mode = 4

        elif self.mode == 4:
            # Releasing the box
            target_global_robot_hand_pos = global_robot_hand_pos
            if abs(f_state[6]) < 30:
                target_f_state[6] = f_state[6] + (2 if self.control_left else -2)  # open gripper with 2 deg per step
            else:
                end = True

        target_robot_hand_pos = transform(self.transform, target_global_robot_hand_pos)  # (3,)
        target_f_state[0:3] = target_robot_hand_pos
        target_f_state[3:6] = self.init_f_state[3:6]

        target = robot_state.copy()
        if self.state_type == "pos":
            target[:7] = target_f_state
        else:
            target[:7] = self.env.solve_ik(target_f_state, robot_state)

        self.prev_gripper_angle = f_state[6]

        if self.action_type == "state_diff":
            return target - robot_state, end  # ndarray: (7,)

        return target, end
