import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import transforms

def draw_bbox2d(objects, color='k', ax=None):

    limits = ax.axis()

    for obj in objects:
        x, _, z = obj.position
        l, _, w = obj.dimensions

        # Setup transform 
        t = transforms.Affine2D().rotate(obj.angle + math.pi/2)
        t = t.translate(x, z) + ax.transData

        # Draw 2D object bounding box
        rect = Rectangle((-w/2, -l/2), w, l, edgecolor=color, transform=t, fill=False)
        ax.add_patch(rect)

        # Draw dot indicating object center
        center = Circle((x, z), 0.25, facecolor='k')
        ax.add_patch(center)

    ax.axis(limits)
    return ax
