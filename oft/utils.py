import time
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor
from collections import namedtuple, defaultdict, Counter

ObjectData = namedtuple('ObjectData', 
    ['classname', 'position', 'dimensions', 'angle', 'score'])

class MetricDict(defaultdict):

    def __init__(self):
        super().__init__(float)
        self.count = defaultdict(int)
    
    def __add__(self, other):
        for key, value in other.items():
            self[key] += value
            self.count[key] += 1
        return self
    
    @property
    def mean(self):
        return { key: self[key] / self.count[key] for key in self.keys()}


class Timer(object):

    def __init__(self):
        self.total = 0
        self.runs = 0
        self.t = 0
    
    def reset(self):
        self.total = 0
        self.runs = 0
        self.t = 0
    
    def start(self):
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        self.t = time.perf_counter()
    
    def stop(self):
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        self.total += time.perf_counter() - self.t
        self.runs += 1
    
    @property
    def mean(self):
        val = self.total / self.runs 
        self.reset()
        return val


def rotate(vector, angle):
    """
    Rotate a vector around the y-axis
    """
    sinA, cosA = torch.sin(angle), torch.cos(angle)
    xvals =  cosA * vector[..., 0] + sinA * vector[..., 2]
    yvals = vector[..., 1]
    zvals = -sinA * vector[..., 0] + cosA * vector[..., 2]
    return torch.stack([xvals, yvals, zvals], dim=-1)


def perspective(matrix, vector):
    """
    Applies perspective projection to a vector using projection matrix
    """
    vector = vector.unsqueeze(-1)
    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]


def make_grid(grid_size, grid_offset, grid_res):
    """
    Constructs an array representing the corners of an orthographic grid 
    """
    depth, width = grid_size
    xoff, yoff, zoff = grid_offset

    xcoords = torch.arange(0., width, grid_res) + xoff
    zcoords = torch.arange(0., depth, grid_res) + zoff

    zz, xx = torch.meshgrid(zcoords, xcoords)
    return torch.stack([xx, torch.full_like(xx, yoff), zz], dim=-1)


def gaussian_kernel(sigma=1., trunc=2.):

    width = round(trunc * sigma)
    x = torch.arange(-width, width+1).float() / sigma
    kernel1d = torch.exp(-0.5 * x ** 2)
    kernel2d = kernel1d.view(1, -1) * kernel1d.view(-1, 1)

    return kernel2d / kernel2d.sum()


def bbox_corners(obj):
    """
    Return the 2D
    """

    # Get corners of bounding box in object space
    offsets = torch.tensor([
        [-.5,  0., -.5],    # Back-left lower
        [ .5,  0., -.5],    # Front-left lower
        [-.5,  0.,  .5],    # Back-right lower
        [ .5,  0.,  .5],    # Front-right lower
        [-.5, -1., -.5],    # Back-left upper
        [ .5, -1., -.5],    # Front-left upper
        [-.5, -1.,  .5],    # Back-right upper
        [ .5, -1.,  .5],    # Front-right upper
    ])
    corners = offsets * torch.tensor(obj.dimensions)
    # corners = corners[:, [2, 0, 1]]

    # Apply y-axis rotation
    corners = rotate(corners, torch.tensor(obj.angle))

    # Apply translation
    corners = corners + torch.tensor(obj.position)
    return corners



def collate(batch):

    idxs, images, calibs, objects, grids = zip(*batch)

    # Crop images to the same dimensions
    minw = min(img.size[0] for img in images)
    minh = min(img.size[1] for img in images)
    images = [img.crop((0, 0, minw, minh)) for img in images]

    # Create a vector of indices
    idxs = torch.LongTensor(idxs)

    # Stack images and calibration matrices along the batch dimension
    images = torch.stack([to_tensor(img) for img in images])
    calibs = torch.stack(calibs)
    grids = torch.stack(grids)

    return idxs, images, calibs, objects, grids


def convert_figure(fig, tight=True):
    # Converts a matplotlib figure into a numpy array

    # Draw plot
    if tight:
        fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Convert figure to numpy via a string array
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    






