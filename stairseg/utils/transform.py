import numpy as np

translation_mat = lambda dx, dy, dz: np.array((
    (1, 0, 0, dx),
    (0, 1, 0, dy),
    (0, 0, 1, dz),
    (0, 0, 0, 1),
), dtype=np.float64)


def rotation_mat(x, y, z) -> np.ndarray:
    along_z = np.array((
        (np.cos(z), -np.sin(z), 0, 0),
        (np.sin(z), np.cos(z), 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ), dtype=np.float64)
    along_y = np.array((
        (np.cos(y), 0, -np.sin(y), 0),
        (0, 1, 0, 0),
        (np.sin(y), 0, np.cos(y), 0),
        (0, 0, 0, 1),
    ), dtype=np.float64)
    along_x = np.array((
        (1, 0, 0, 0),
        (0, np.cos(x), -np.sin(x), 0),
        (0, np.sin(x), np.cos(x), 0),
        (0, 0, 0, 1),
    ), dtype=np.float64)

    return along_z @ along_y @ along_x
