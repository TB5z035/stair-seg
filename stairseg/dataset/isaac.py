from cProfile import label
import os
from pathlib import Path

from torch.utils.data import Dataset
import MinkowskiEngine as ME

from ..utils.io import load_plyfile_to_array


class IssacSimDataset(Dataset):

    def __init__(self, plyfile_dir, quantization_size=0.02) -> None:
        self.plyfile_dir = Path(plyfile_dir)
        self.plyfile_list = sorted(os.listdir(plyfile_dir)) # TODO use file list to split train/val 
        self.quantization_size = quantization_size
        super().__init__()

    def __getitem__(self, index):
        coords, feats, labels = load_plyfile_to_array(self.plyfile_dir / self.plyfile_list[index])

        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=coords,
            features=feats,
            labels=labels,
            quantization_size=self.quantization_size,
        )
        return discrete_coords, unique_feats, unique_labels

    def __len__(self):
        return len(self.plyfile_list)
