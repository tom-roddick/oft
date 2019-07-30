import os
import torch
import pandas
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from .. import utils

KITTI_CLASS_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                     'Cyclist', 'Tram', 'Misc', 'DontCare']

class KittiObjectDataset(Dataset):

    def __init__(self, kitti_root, split='train', grid_size=(80., 80.), 
                 grid_res=0.5, y_offset=1.74):
        
        # Get the root directory containing object detection data
        kitti_split = 'testing' if split == 'test' else 'training'
        self.root = os.path.join(kitti_root, 'object', kitti_split)

        # Read split indices from file
        split_file = os.path.dirname(__file__) + '/splits/{}.txt'.format(split)
        self.indices = read_split(split_file)

        # Make grid
        self.grid = utils.make_grid(
            grid_size, (-grid_size[0]/2., y_offset, 0.), grid_res)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        idx = self.indices[index]

        # Load image
        img_file = os.path.join(self.root, 'image_2/{:06d}.png'.format(idx))
        image = Image.open(img_file)

        # Load calibration matrix
        calib_file = os.path.join(self.root, 'calib/{:06d}.txt'.format(idx))
        calib = read_kitti_calib(calib_file)

        # Load annotations
        label_file = os.path.join(self.root, 'label_2/{:06d}.txt'.format(idx))
        objects = read_kitti_objects(label_file)

        return idx, image, calib, objects, self.grid

def read_split(filename):
    """
    Read a list of indices to a subset of the KITTI training or testing sets
    """
    with open(filename) as f:
        return [int(val) for val in f]

def read_kitti_calib(filename):
    """Read the camera 2 calibration matrix from a text file"""

    with open(filename) as f:
        for line in f:
            data = line.split(' ')
            if data[0] == 'P2:':
                calib = torch.tensor([float(x) for x in data[1:13]])
                return calib.view(3, 4)
    
    raise Exception(
        'Could not find entry for P2 in calib file {}'.format(filename))

def read_kitti_objects(filename):

    objects = list()
    with open(filename, 'r') as fp:
        
        # Each line represents a single object
        for line in fp:
            objdata = line.split(' ')
            if not (14 <= len(objdata) <= 15):
                raise IOError('Invalid KITTI object file {}'.format(filename))

            # Parse object data
            objects.append(utils.ObjectData(
                classname=objdata[0],
                dimensions=[
                    float(objdata[10]), float(objdata[8]), float(objdata[9])],
                position=[float(p) for p in objdata[11:14]],
                angle=float(objdata[14]),
                score=float(objdata[15]) if len(objdata) == 16 else 1.
            ))
    return objects
