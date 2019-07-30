import matplotlib.pyplot as plt
import matplotlib.colors as colors
from .bbox import draw_bbox2d

def vis_score(score, grid, cmap='cividis', ax=None):
    score = score.cpu().float().detach().numpy()
    grid = grid.cpu().detach().numpy()

    # Create a new axis if one is not provided
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    
    # Plot scores
    ax.clear()
    ax.pcolormesh(grid[..., 0], grid[..., 2], score, cmap=cmap, vmin=0, vmax=1)
    ax.set_aspect('equal')

    # Format axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')

    return ax


def vis_uncertainty(logvar, objects, grid, cmap='cividis_r', ax=None):
    var = logvar.cpu().float().detach().numpy()
    grid = grid.cpu().detach().numpy()

    # Create a new axis if one is not provided
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    
    # Plot scores
    ax.clear()
    ax.pcolormesh(grid[..., 0], grid[..., 2], var, cmap=cmap)
    ax.set_aspect('equal')
    
    # Draw object positions
    draw_bbox2d(objects, ax=ax)

    # Format axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    return ax

