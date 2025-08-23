import os
import random
import numpy as np

import tongsystem

from controller import TongPickAndPlaceController


def main(num_demos: int, max_step: int, viewer: bool, create_demos: bool, log_dir: str):
    """
    test TongEnvs and creating demonstrations
    """
    # Fix seed
    random.seed(0)
    np.random.seed(0)

    # Create environment
    init_poses = [
        [-0.8, -1.5, 1.5, -1.5, -1.5, 1.5, 30, 0.8, -1.5, -1.5, -1.5, 1.5, -1.5, -30, 0, -0.9],
    ]

    state_type = "pos"
    action_type = "state"
    env = tongsystem.TongSystemUseOnlyLeftArm(
        state_type,
        action_type,
        image_width=320,
        image_height=180,
        init_poses=init_poses,
        log_dir=log_dir,
    )

    controller = TongPickAndPlaceController(env.ur5_manager.ur5_kin, state_type, action_type, control_arm="left")

    keyboard = tongsystem.KeyboardManager()

    abort_flag = False
    saved_demo_count = 0
    print("[Info] Loop starts")
    for episode in range(10000):
        obs = env.reset()
        world_state = env.get_world_state()

        goal_box_pos_rand = world_state["box2"]["position"] + np.array(
            [np.random.uniform(-0.01, 0.01), np.random.uniform(-0.005, 0.005), np.random.uniform(-0.02, 0.02)]
        )
        goal_box_pos_rand[2] += 0.07
        goal_box_pos_rand[1] -= 0.015
        controller.reset(obs["state"][:16], world_state["box1"]["position"], goal_box_pos_rand)

        end_count = 0
        success = False
        for step in range(max_step):
            # keyboard input
            key = keyboard.get_key()
            if key == "a":
                abort_flag = True
                break
            elif key == "b":
                break

            action, end = controller.control(obs["state"][:16], world_state["box1"]["position"], [0.01, 0.03, 0.02])

            if controller.mode < 2 and world_state["box1"]["position"][2] < 0.73:
                dummy_gaze = np.floor(world_state["box1"]["pixel"] * np.array([env.image_width, env.image_height, env.image_width, env.image_height]))
            else:
                dummy_gaze = np.floor(world_state["box2"]["pixel"] * np.array([env.image_width, env.image_height, env.image_width, env.image_height]))
            dummy_gaze += np.random.randint(-5, 6, size=dummy_gaze.shape)

            next_obs, _, _, _ = env.step(action[:7], dummy_gaze)

            world_state = env.get_world_state()

            if viewer:
                env.render(mode="human")

            obs = next_obs

            if end:
                end_count += 1
            if end_count > 10:  # wait for 10 step to close
                if world_state["box1"]["position"][2] > 0.73:
                    success = True
                break

        print(f"episode: {episode}, steps: {step}, success: {success}")
        if create_demos and success:
            env.save_log()
            saved_demo_count += 1
            print(f"Saved Demo: {saved_demo_count}")

        if abort_flag or saved_demo_count == num_demos:
            break
    print("[Info] Finish")


if __name__ == "__main__":
    num_demos = 110
    max_step = 250
    use_viewer = True
    sync_realtime = False
    create_demos = True
    log_dir = "./logs"

    main(num_demos, max_step, use_viewer, create_demos, log_dir)
