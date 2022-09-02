# %%
import os
from pathlib import Path
import numpy as np
import cv2
import re
from plyfile import PlyData

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


def project(rgb, depth, mask, fov, cam_rot, cam_coord) -> PlyData:
    # ((fov_h, fov_v), (focal_h, focal_v), (cam_rot_x, cam_rot_y, cam_rot_z), (cam_coord_x, cam_coord_y, cam_coord_z)) = params
    # fov_h, fov_v, focal_h, focal_v, cam_rot_x, cam_rot_y, cam_rot_z, cam_coord_x, cam_coord_y, cam_coord_z = float(fov_h), float(fov_v), float(
    #     focal_h), float(focal_v), float(cam_rot_x), float(cam_rot_y), float(cam_rot_z), float(cam_coord_x), float(cam_coord_y), float(cam_coord_z)
    height, width, _ = rgb.shape
    horizontal_ratio = np.array([(2. * i / width - 1) for i in range(width)])  # [-1, 1]
    horizontal_ratio_mat = np.stack([horizontal_ratio for _ in range(height)], axis=1).T
    vertical_ratio = np.array([(2. * i / height - 1) for i in list(range(height))[::-1]]).T  # [-1, 1]
    vertical_ratio_mat = np.stack([vertical_ratio for _ in range(width)], axis=0).T
    x_mat = horizontal_ratio_mat * np.tan(fov[0] / 2) * depth
    z_mat = horizontal_ratio_mat * np.tan(fov[1] / 2) * depth
    y_mat = depth
    # with open('test.txt', 'w') as f:
    #     for x, y, z in zip(x_mat.flatten(), y_mat.flatten(), z_mat.flatten()):
    #         print(f"{x} {y} {z}", file=f)


if __name__ == '__main__':
    sorted_ls = lambda path: [path + '/' + i for i in sorted(os.listdir(path))]
    for rgb_path, depth_path, mask_path, (fov, _, rot, coord) in zip(sorted_ls(RGB_DIR), sorted_ls(DEPTH_DIR), sorted_ls(MASK_DIR), parse_log(SAMPLING_LOG_PATH)):
        rgb = parse_rgb(rgb_path)
        depth = parse_depth(depth_path)
        mask = parse_mask(mask_path)
        project(rgb, depth, mask, fov, rot, coord)

# %%

### DEBUG ONLY ###
import numpy as np
import cv2
from stairseg.utils.vis import arr2Image

PFM_PATH = '../data/stairs/data/mono/depth/000.pfm'

im = cv2.imread(PFM_PATH, cv2.IMREAD_UNCHANGED)

# %%
