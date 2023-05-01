# import numpy as np
# import cv2
# import random
# import os
import json
from json_utils import Ann_utils
# import glob
# import sys
# import matplotlib.figure as mplfigure
# from matplotlib.backends.backend_agg import FigureCanvasAgg


json_file = '/data/Projects/accessmath-icfhr2018/AccessMathVOC/lecture_01/annotation.json'
ann_util = Ann_utils(json_file)

ann_util.json_info(verbose=True)