from typing import List
import numpy as np
from plyfile import PlyData


def load_plyfile_to_array(
    ply_path,
    coords=['x', 'y', 'z'],
    features=['red', 'green', 'blue'],
    label=['label'],
) -> List[np.ndarray]:
    """
    TODO add docstring
    TODO add mesh information
    """
    plydata = PlyData.read(ply_path)
    component_names = [i.name for i in plydata.elements]
    assert 'vertex' in component_names

    def parse(fields):
        if fields is not None:
            if len(fields) > 1:
                ret = np.stack([plydata['vertex'][field] for field in fields], axis=-1)
            else:
                ret = np.array(plydata['vertex'][fields[0]])
        else:
            ret = None
        return ret

    ret_coords = parse(coords)
    ret_feats = parse(features)
    ret_labels = parse(label)
    return ret_coords, ret_feats, ret_labels
