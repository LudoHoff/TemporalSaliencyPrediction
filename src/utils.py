import fnmatch
import os
import random
import cv2

import scipy.io as sio
import minpy.numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.image import imread
from tqdm import tqdm
from scipy import ndimage
from scipy.spatial import distance

IMAGE_PATH = '../data/images/'
FIXATION_PATH = '../data/fixations/'
GIF_PATH = '../data/gifs/'

TRAIN_PATH = 'train/'
VAL_PATH = 'val/'

GIF_SAMPLES = 2
W = 640
H = 480
TIMESPAN = 5000
MAX_PIXEL_DISTANCE = 800
ESTIMATED_TIMESTAMP_WEIGHT = 0.006

def get_filenames(path):
    return [file for file in sorted(os.listdir(path)) if fnmatch.fnmatch(file, 'COCO_*')]


def get_saliency_volumes(filenames,
                         path_prefix=TRAIN_PATH,
                         etw=ESTIMATED_TIMESTAMP_WEIGHT, progress_bar=True):
    saliency_volumes = []
    filenames = tqdm(filenames) if progress_bar else filenames

    for filename in filenames:
        # 1. Extracting data from .mat files
        mat = sio.loadmat(FIXATION_PATH + path_prefix + filename)
        gaze = mat["gaze"]

        locations = []
        timestamps = []
        fixations = []

        for i in range(len(gaze)):
            locations.append(mat["gaze"][i][0][0])
            timestamps.append(mat["gaze"][i][0][1])
            fixations.append(mat["gaze"][i][0][2])

        # 2. Matching fixations with timestamps
        saliency_volume = []
        for i, observer in enumerate(fixations):
            fix_timestamps = []
            fix_time = TIMESPAN / (len(observer) + 1)
            est_timestamp = 0

            for fixation in observer:
                distances = distance.cdist([fixation], locations[i], 'euclidean')[0][..., np.newaxis]
                time_diffs = abs(timestamps[i] - est_timestamp)
                min_idx = int(np.argmin(etw * time_diffs + distances)[0])

                fix_timestamps.append([min(timestamps[i][min_idx][0], TIMESPAN), fixation.tolist()])
                est_timestamp += fix_time

            if (len(observer) > 0):
                saliency_volume.append(fix_timestamps)

        saliency_volumes.append(saliency_volume)

    return saliency_volumes

def get_heat_image(image):
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * image), cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)

RATIO = 0.9
def format_image(heatmap, image, max_value):
    extended_map = heatmap / max_value
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    factors = np.clip(2 * extended_map, 0, 1)
    hsv[:,:,1] = np.uint8(factors * hsv[:,:,1])
    hsv[:,:,2] = np.uint8((RATIO * factors + (1 - RATIO)) * hsv[:,:,2])

    return get_heat_image(extended_map[:,:,np.newaxis]), cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def animate(maps, image, normalized=False):
    fig = plt.figure(figsize=(13, 6))
    max_value = np.max(maps);
    formatted_images = []

    for map in maps:
        heatmap, image_heatmap = format_image(map, image, np.max(map) if normalized else max_value)
        im = plt.imshow(np.concatenate((heatmap, image_heatmap), 1), animated=True)
        formatted_images.append([im])

    return animation.ArtistAnimation(fig, formatted_images, interval=1, blit=True, repeat_delay=1000)
