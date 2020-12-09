import os
import time
import argparse
import pandas as pd
from scipy.stats import zscore
from scipy import stats
import numpy as np
import cv2 as cv
import pycuda.autoinit  # This is needed for initializing CUDA driver
import matplotlib.pyplot as plt
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from AitasConcrete import AitasConcrete

WINDOW_NAME = 'TrtYOLOv4'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
            '-n', '--num_points', type=int, default=50,
            help='Number of interpolated points.')
    parser.add_argument(
            '-c', '--category_num', type=int, default=1,
            help='number of object categories [80]')
    parser.add_argument('-d', '--display_mode', type=bool, default=False,
                        help=('to turn display_mode on: --display_mode True'))
    parser.add_argument(
            '-lat', '--latitude', type=float, required=True,
            help=('Please provide latitude in decimal degrees'))
    parser.add_argument('-lon', '--longitude', type=float, required=True,
                        help=('Please provide longitude in decimal degrees'))
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # todo: should parse camera gps as a 2, 1 numpy array

    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    # load video
    path_to_vid = args.video
    stream = cv.VideoCapture(path_to_vid)
    camera_latitude = args.latitude
    camera_longitude = args.longitude
    camera_gps = np.array([camera_latitude, camera_longitude])
    
    num_points = args.num_points
    

    # testing
    temp = AitasConcrete(stream, camera_gps)
    temp.process_video(num_points)

    # temp.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
