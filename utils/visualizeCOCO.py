# import numpy as np
# import cv2
# import random
# import os
# import json
# import glob
# import sys
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg

class VisImgInitializer:
    def __init__(self, img, scale=1.0, rgb=True, maximum=255):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)
        self.ret = self.random_color(rgb=True, maximum=255)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax

    def random_color(rgb=False, maximum=255):
        """
        Args:
            rgb (bool): whether to return RGB colors or BGR colors.
            maximum (int): either 255 or 1

        Returns:
            ndarray: a vector of 3 numbers
        """
        _COLORS = np.array(
        [0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000])
        idx = np.random.randint(0, len(_COLORS))
        ret = _COLORS[idx] * maximum
        if not rgb:
            ret = ret[::-1]
    return ret

class VISUALIZER:
    def __init__(self, img, bboxes, scale=1.0):
        self.img = img
        self.output = VisImgInitializer(self.img, scale=scale, rgb=True, maximum=255)
        self.scale = scale
        num_instances = len(bboxes)
        self.assigned_colors = [random_color for _ in range(num_instances)]
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
    def overlay_instances():
        for bbox in bboxes:
            self.draw_box(bbox, alpha=0.5, edge_color="g", line_style="-"):

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing xmin, ymin, xmax, ymax
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output


# Directory to the xml files
root_dir = '/data/Projects/accessmath-icfhr2018/AccessMathVOC'
train_lectures = ['lecture_01']

for lecture in train_lectures:
    # Directory to the image files
    img_dir = os.path.join(root_dir, lecture, 'JPEGImages')
    coco_path = os.path.join(root_dir, lecture, 'annotation.json')

    # Output img file path
    output_path = os.path.join(root_dir, lecture, 'JPEGImages_vis')
    os.makedirs(output_path, exist_ok=True)
    xml2coco(xml_dir, output_jsonpath, img_dir)


class VISUALIZER:
    def __init__(self, img, bboxes, scale=1.0):

output_dir = '/data/Datasets/FCI/Encap/train_set_2/images_vis_v1'
os.makedirs(output_dir, exist_ok=True)

img_dir = '/data/Datasets/FCI/Encap/train_set_2/images'
json_file = '/data/Datasets/FCI/Encap/train_set_2/encap10_v1.json'
# ann_utils = Ann_utils(json_file)
# print(ann_utils.data['categories'])
# print(len(ann_utils.data['images']))


# cat_dict = {1: 'Insufficient', 2: 'Encap', 3: 'Contam', 4: 'contam_or_pass', \
#     5: 'Voids', 6: 'voids_or_bh', 7: 'voids_or_pass', 8: 'Pass', 9: 'Crack', \
#     10: 'Blowhole', 11: 'origin', 12: 'Pass_notsure', 13: 'contam_smallsize', \
#     14: 'Beading', 15: 'Crazing', 16: 'Smearing', 17: 'TailsNDrips', 18: 'Wicking', 19: 'Uncure'}


# Each registered dataset is associated with some meta data. Access the registered data
metadata = MetadataCatalog.get("dataset")
print('metadata', metadata)
dataset_dicts = DatasetCatalog.get("dataset")

class_names = metadata.get("thing_classes", None)
print('class_names', class_names)

# verify the data is loaded correctly
for i, d in enumerate(dataset_dicts):
    print(d["file_name"])
    file_name = os.path.basename(d["file_name"])

    img = cv2.imread(d["file_name"])
    # print(i, d["file_name"])
    # print(file_name)
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)


    cv2.imwrite(output_dir+'/'+file_name[:-4]+'_annotation.jpg', vis.get_image()[:, :, ::-1])
    cv2.imwrite(f'{output_dir}/{file_name}', img)
