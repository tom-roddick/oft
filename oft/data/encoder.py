import math
import torch
from .. import utils

class ObjectEncoder(object):

    def __init__(self, classnames=['Car'], pos_std=[.5, .36, .5], 
                 log_dim_mean=[[0.42, 0.48, 1.35]], 
                 log_dim_std=[[.085, .067, .115]], sigma=1.):
        
        self.classnames = classnames
        self.nclass = len(classnames)
        self.pos_std = pos_std
        self.log_dim_mean = log_dim_mean
        self.log_dim_std = log_dim_std
        self.sigma = sigma
        
    
    def encode_batch(self, objects, grids):

        # Encode batch element by element
        batch_encoded = [self.encode(objs, grid) for objs, grid 
                         in zip(objects, grids)]
        
        # Transpose batch
        return [torch.stack(t) for t in zip(*batch_encoded)]


    def encode(self, objects, grid):
        
        # Filter objects by class name
        objects = [obj for obj in objects if obj.classname in self.classnames]

        # Skip empty examples
        if len(objects) == 0:
            return self._encode_empty(grid)
        
        # Construct tensor representation of objects
        classids = torch.tensor([self.classnames.index(obj.classname) 
                                for obj in objects], device=grid.device)
        positions = grid.new([obj.position for obj in objects])
        dimensions = grid.new([obj.dimensions for obj in objects])
        angles = grid.new([obj.angle for obj in objects])

        # Assign objects to locations on the grid
        _, indices = self._assign_to_grid(
            classids, positions, dimensions, angles, grid)
        mask = indices >= 0

        # Encode object heatmaps
        heatmaps = self._encode_heatmaps(classids, positions, grid)
        
        # Encode positions, dimensions and angles
        pos_offsets = self._encode_positions(positions, indices, grid)
        dim_offsets = self._encode_dimensions(classids, dimensions, indices)
        ang_offsets = self._encode_angles(angles, indices)

        return heatmaps, pos_offsets, dim_offsets, ang_offsets, mask   
    

    def _assign_to_grid(self, classids, positions, dimensions, angles, grid):

        # Compute grid centers
        centers = (grid[1:, 1:, :] + grid[:-1, :-1, :]) / 2.

        # Transform grid into object coordinate systems
        local_grid = utils.rotate(centers - positions.view(-1, 1, 1, 3), 
            -angles.view(-1, 1, 1)) / dimensions.view(-1, 1, 1, 3)
        
        # Find all grid cells which lie within each object
        inside = (local_grid[..., [0, 2]].abs() <= 0.5).all(dim=-1)
        
        # Expand the mask in the class dimension NxDxW * NxC => NxCxDxW
        class_mask = classids.view(-1, 1) == torch.arange(
            len(self.classnames)).type_as(classids)
        class_inside = inside.unsqueeze(1) & class_mask[:, :, None, None]

        # Return positive locations and the id of the corresponding instance 
        labels, indices = torch.max(class_inside, dim=0)
        return labels, indices
    

    def _encode_heatmaps(self, classids, positions, grid):

        centers = (grid[1:, 1:, [0, 2]] + grid[:-1, :-1, [0, 2]]) / 2.
        positions = positions.view(-1, 1, 1, 3)[..., [0, 2]]

        # Compute per-object heatmaps
        sqr_dists = (positions - centers).pow(2).sum(dim=-1) 
        obj_heatmaps = torch.exp(-0.5 * sqr_dists / self.sigma ** 2)

        heatmaps = obj_heatmaps.new_zeros(self.nclass, *obj_heatmaps.size()[1:])
        for i in range(self.nclass):
            mask = classids == i
            if mask.any():
                heatmaps[i] = torch.max(obj_heatmaps[mask], dim=0)[0]

        return heatmaps

    

    def _encode_positions(self, positions, indices, grid):

        # Compute the center of each grid cell
        centers = (grid[1:, 1:] + grid[:-1, :-1]) / 2.

        # Encode positions into the grid
        C, D, W = indices.size()
        positions = positions.index_select(0, indices.view(-1)).view(C, D, W, 3)

        # Compute relative offsets and normalize
        pos_offsets = (positions - centers) / grid.new_tensor(self.pos_std)
        return pos_offsets.permute(0, 3, 1, 2)
    
    def _encode_dimensions(self, classids, dimensions, indices):
        
        # Convert mean and std to tensors
        log_dim_mean = dimensions.new_tensor(self.log_dim_mean)[classids]
        log_dim_std = dimensions.new_tensor(self.log_dim_std)[classids]

        # Compute normalized log scale offset
        dim_offsets = (torch.log(dimensions) - log_dim_mean) / log_dim_std

        # Encode dimensions to grid
        C, D, W = indices.size()
        dim_offsets = dim_offsets.index_select(0, indices.view(-1))
        return dim_offsets.view(C, D, W, 3).permute(0, 3, 1, 2)
    

    def _encode_angles(self, angles, indices):

        # Compute rotation vector
        sin = torch.sin(angles)[indices]
        cos = torch.cos(angles)[indices]
        return torch.stack([cos, sin], dim=1)
    

    def _encode_empty(self, grid):
        depth, width, _ = grid.size()

        # Generate empty tensors
        heatmaps = grid.new_zeros((self.nclass, depth-1, width-1))
        pos_offsets = grid.new_zeros((self.nclass, 3, depth-1, width-1))
        dim_offsets = grid.new_zeros((self.nclass, 3, depth-1, width-1))
        ang_offsets = grid.new_zeros((self.nclass, 2, depth-1, width-1))
        mask = grid.new_zeros((self.nclass, depth-1, width-1)).byte()
        
        return heatmaps, pos_offsets, dim_offsets, ang_offsets, mask

