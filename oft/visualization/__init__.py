import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .encoded import vis_score, vis_uncertainty
from .bbox import visualize_objects

# def vis_score(scores, labels, ax=None):
#     scores = scores.sigmoid().cpu().detach().numpy()
#     labels = labels.cpu().float().detach().numpy()
#     # grid = 0.5 * (grid[1:, 1:, [0, 2]] + grid[:-1, :-1, [0, 2]]).cpu().detach().numpy()

#     # norm = Normalize(0., 1.)
#     # pos_cmap = ScalarMappable(norm, cmap='YlOrRd').to_rgba(scores)
#     # neg_cmap = ScalarMappable(norm, cmap='PuBu').to_rgba(1-scores)

#     # colors = pos_cmap * labels[..., None] + neg_cmap * (1-labels)[..., None]

#     if ax is None:
#         fig = plt.figure()
#         ax = plt.gca()
#     ax.imshow(labels - scores, vmin=-1, vmax=1, cmap='coolwarm')
#     return ax

    