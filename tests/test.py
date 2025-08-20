import os
import numpy as np

import tongsystem


def main(log_dir: str):
    # fmt: off
    init_poses = [
        [-0.7261853, -1.27618531, 0.957000000, -1.10618531, -1.22618531, 0.636000000, 20.0021325, 1.04400000, -1.66818531, -2.25618531, -0.091185307, 0.98400000, -1.53018531, -20.0009621, 0.0, 0.0],
        [-0.9250245, -1.11701072, 1.67551608, 4.25860337, -1.18682389, 1.43116999, 10.0, 0.97738438, -1.97222205, -1.83259571, -0.9424778, 1.13446401, -1.3962634, -10.0, 0.0, -0.9],
    ]
    # fmt: on

    env = tongsystem.TongSystemUseOnlyLeftArm(
        state_type="pos",
        action_type="state",
        image_width=320,
        image_height=180,
        init_poses=init_poses,
        log_dir=log_dir,
    )

    keyboard = tongsystem.KeyboardManager()

    MAX_STEP = 10000
    abort_flag = False
    print("[Info] Loop starts")
    while True:
        obs = env.reset()

        for _ in range(MAX_STEP):
            action = obs["state"][:7]  # np.zeros_like(obs["state"][:7])
            obs, _, _, _ = env.step(action)
            # obs, _, _, _ = env.freeze_step()

            env.render(mode="human")

            key = keyboard.get_key()
            if key == "a":
                abort_flag = True
                break
            elif key == "b":
                break

        # env.save_log()

        if abort_flag:
            break

    print("[Info] Finish")


if __name__ == "__main__":
    main(log_dir="logs")
