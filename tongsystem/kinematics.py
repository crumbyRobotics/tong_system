import numpy as np


left_ee2world = np.array(
    [
        [-1 / np.sqrt(2), 0, 1 / np.sqrt(2), -1.003 / np.sqrt(2)],
        [1 / np.sqrt(2), 0, 1 / np.sqrt(2), -1.003 / np.sqrt(2)],
        [0, 1, 0, -0.2],
        [0, 0, 0, 1],
    ]
)
right_ee2world = np.array(
    [
        [1 / np.sqrt(2), 0, -1 / np.sqrt(2), 1.003 / np.sqrt(2)],
        [1 / np.sqrt(2), 0, 1 / np.sqrt(2), -1.003 / np.sqrt(2)],
        [0, -1, 0, -0.2],
        [0, 0, 0, 1],
    ]
)

camera2left_ee = np.array(
    [
        [-0.02177536, 0.03698473, -0.99907856, -0.22352613],
        [-0.03215057, -0.99882456, -0.03627459, 0.30907208],
        [-0.9992458, 0.03133106, 0.02293885, -0.18261354],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
camera2right_ee = np.array(
    [
        [0.02638825, -0.06179807, 0.99773977, 0.22079462],
        [-0.03824855, -0.9974189, -0.0607666, 0.32936743],
        [0.99891977, -0.03655858, -0.02868382, -0.22785084],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

camera2left_ee_sim = np.array(
    [
        [0.000, 0.114, -0.9934, -0.2534],
        [0.000, -0.9934, -0.1144, 0.31953],
        [-1.000, 0.0000, 0.0000, -0.1715],
        [0.000, 0.0000, 0.0000, 1.00000],
    ]
)
camera2right_ee_sim = np.array(
    [
        [0.000, -0.114, 0.9934, 0.2534],
        [0.000, -0.9934, -0.114, 0.3195],
        [1.000, 0.000, 0.000, -0.2285],
        [0.000, 0.000, 0.000, 1.0000],
    ]
)


def r2euler(r: np.ndarray) -> np.ndarray:
    sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(r[2, 1], r[2, 2])
        y = np.arctan2(-r[2, 0], sy)
        z = np.arctan2(r[1, 0], r[0, 0])
    else:
        x = np.arctan2(-r[1, 2], r[1, 1])
        y = np.arctan2(-r[2, 0], sy)
        z = 0.0
    euler = np.array([x, y, z])
    return euler


def euler2r(theta: np.ndarray) -> np.ndarray:
    # https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    r_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )

    r_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )

    r_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    r = np.dot(r_z, np.dot(r_y, r_x))
    return r
