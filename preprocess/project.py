# %%
import os
from pathlib import Path
import numpy as np
import cv2
import re
from plyfile import PlyData, PlyElement
from IPython import embed

from stairseg.utils.transform import translation_mat, rotation_mat

RGB_DIR = '../data/stairs/data/mono/rgb'
DEPTH_DIR = '../data/stairs/data/mono/depth'
MASK_DIR = '../data/annotations/SegmentationClass'
SAMPLING_LOG_PATH = '../data/stairs/log/sampling_log.yaml'

parse_rgb = lambda path: cv2.imread(path, cv2.IMREAD_COLOR)
parse_depth = lambda path: cv2.imread(path, cv2.IMREAD_UNCHANGED)
parse_mask = lambda path: (cv2.imread(path, cv2.IMREAD_COLOR) != [0, 0, 0]).any(axis=2)


def parse_log(path):
    with open(path) as f:
        text = f.read()
    indices_raw = re.findall(r"index:\s(.+)\n", text)
    fovs = re.findall(r"horiz_fov,\svert_fov:\s([\d\.]+),\s([\d\.]+)", text)
    focals = re.findall(r"focal_x,\sfocal_y:\s([\d\.]+),\s([\d\.]+)", text)
    rots = re.findall(r"parameter:\scamera_rot\s+val:\s\'\[(.*?)\]\'", text)
    rots = [i.split() for i in rots]
    coords = re.findall(r"parameter:\scamera_coord\s+val:\s\'\[(.*?)\]\'", text)
    coords = [i.split() for i in coords]
    count = len(coords)

    starts_idx = [int(indices_raw[i + 1]) for i in range(len(indices_raw)) if indices_raw[i] == 'start-up'] + [count]
    step_counts = [starts_idx[i + 1] - starts_idx[i] for i in range(len(starts_idx) - 1)]

    def meta_f(l):
        multi_level = [[l[idx] for _ in range(count)] for idx, count in enumerate(step_counts)]
        single_level = []
        for li in multi_level:
            single_level += li
        return single_level

    fovs = meta_f(fovs)
    fovs = ((float(i), float(j)) for i, j in fovs)
    focals = meta_f(focals)
    focals = ((float(i), float(j)) for i, j in focals)
    rots = ((float(i), float(j), float(k)) for i, j, k in rots)
    coords = ((float(i), float(j), float(k)) for i, j, k in coords)
    return zip(fovs, focals, rots, coords)


def construct_vertex_ply(coords: np.ndarray, rgb: np.ndarray, label: np.ndarray, binary=True) -> PlyData:
    assert len(coords.shape) == 3 and len(rgb.shape) == 3 and len(label.shape) == 2
    assert coords.shape[0] == rgb.shape[0] and rgb.shape[0] == label.shape[0]
    assert coords.shape[1] == rgb.shape[1] and rgb.shape[1] == label.shape[1]
    assert coords.shape[2] == 3 and rgb.shape[2] == 3
    vertex = np.zeros(
        coords.shape[0] * coords.shape[1],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'), ('label', 'u1')],
    )
    vertex['x'] = coords[:, :, 0].flatten()
    vertex['y'] = coords[:, :, 1].flatten()
    vertex['z'] = coords[:, :, 2].flatten()
    vertex['red'] = rgb[:, :, 0].flatten()
    vertex['green'] = rgb[:, :, 1].flatten()
    vertex['blue'] = rgb[:, :, 2].flatten()
    vertex['alpha'] = np.ones_like(rgb[:, :, 0].flatten()) * 255
    vertex['label'] = label.flatten()
    el = PlyElement.describe(vertex, 'vertex')
    return PlyData([el], text=not binary)


def project(rgb, depth, mask, fov, cam_rot, cam_coord) -> PlyData:
    height, width, _ = rgb.shape
    horizontal_ratio = np.array([(2. * i / width - 1) for i in range(width)])  # [-1, 1]
    horizontal_ratio_mat = np.stack([horizontal_ratio for _ in range(height)], axis=1).T
    vertical_ratio = np.array([(2. * i / height - 1) for i in list(range(height))[::-1]]).T  # [-1, 1]
    vertical_ratio_mat = np.stack([vertical_ratio for _ in range(width)], axis=0).T
    x_mat = horizontal_ratio_mat * np.tan(fov[0] / 2 / 180 * np.pi) * depth
    z_mat = vertical_ratio_mat * np.tan(fov[1] / 2 / 180 * np.pi) * depth
    y_mat = depth
    aff_coords = np.stack([x_mat, y_mat, z_mat, np.ones_like(x_mat)],
                          axis=2) @ translation_mat(*[-i for i in cam_coord]) @ rotation_mat(*[-i / 180 * np.pi for i in cam_rot])

    coords = aff_coords[:, :, :3]
    embed()
    construct_vertex_ply(coords, rgb, mask).write('test.ply')
    # with open('test.txt', 'w') as f:
    #     for coord, rgb, mask_bit in zip(coords.reshape(-1, 3), rgb.reshape(-1, 3), mask.flatten()):
    #         construct_vertex_ply(coord, rgb, mask_bit)


if __name__ == '__main__':
    sorted_ls = lambda path: [path + '/' + i for i in sorted(os.listdir(path))]
    for rgb_path, depth_path, mask_path, (fov, _, rot,
                                          coord) in zip(sorted_ls(RGB_DIR), sorted_ls(DEPTH_DIR), sorted_ls(MASK_DIR), parse_log(SAMPLING_LOG_PATH)):
        rgb = parse_rgb(rgb_path)
        depth = parse_depth(depth_path)
        mask = parse_mask(mask_path)
        project(rgb, depth, mask, fov, rot, coord)
        break

# %%

### DEBUG ONLY ###
import numpy as np
import cv2
from stairseg.utils.vis import arr2Image

PFM_PATH = '../data/stairs/data/mono/depth/000.pfm'

im = cv2.imread(PFM_PATH, cv2.IMREAD_UNCHANGED)

# %%
