import pylab
import fnmatch
import os
import random
import cv2

import numpy as np
import skimage.io as io
import scipy.io as sio
import matplotlib.pyplot as plt
import ipywidgets as wg
import scipy.interpolate as interp

from matplotlib.image import imread
from tqdm import tqdm
from scipy import ndimage
from scipy.spatial import distance
from scipy.stats import gaussian_kde

IMAGE_PATH = '../data/images/train/'
FIXATION_PATH = '../data/fixations/train/'
MAP_PATH = '../data/maps/train/'
GT_SLICE_PATH = '../data/3-sliced_maps/'
MDSEM_GT_SLICE_PATH = '../data/mdsem_3-sliced_maps/'
FIX_SLICE_PATH = '../data/3-sliced_fixations/'

TRAINING_SIZE = 10000
MDSEM_SIZE = 500
W = 640
H = 480
REL_SAMPLES = 10000

MAX_PIXEL_DISTANCE = np.linalg.norm([W, H])
ESTIMATED_TIMESTAMP_WEIGHT = 0.006

def get_filenames(path):
    return [file for file in sorted(os.listdir(path)) if fnmatch.fnmatch(file, 'COCO_*.*')]

def get_saliency_volumes(filenames=get_filenames(FIXATION_PATH), etw=ESTIMATED_TIMESTAMP_WEIGHT,
                         progress_bar=True):
    saliency_volumes = []
    errors = []

    filenames = tqdm(filenames) if progress_bar else filenames

    for filename in filenames:
        # 1. Extracting data from .mat files
        mat = sio.loadmat(FIXATION_PATH + filename)
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
            fix_time = 5000 / (len(observer) + 1)
            est_timestamp = 0

            for fixation in observer:
                distances = distance.cdist([fixation], locations[i], 'euclidean')[0][..., np.newaxis]
                time_diffs = abs(timestamps[i] - est_timestamp)
                min_idx = np.argmin(etw * time_diffs + distances)

                fix_timestamps.append([min(timestamps[i][min_idx][0], 5000), fixation.tolist()])
                errors.append(distances[min_idx])
                est_timestamp += fix_time

            if (len(observer) > 0):
                saliency_volume.append(fix_timestamps)

        saliency_volumes.append(saliency_volume)

    return saliency_volumes, np.mean(errors) / MAX_PIXEL_DISTANCE

print("Computing saliency volumes...")
indices = random.sample(range(TRAINING_SIZE), REL_SAMPLES)
saliency_volumes, _ = get_saliency_volumes(np.array(get_filenames(FIXATION_PATH))[indices])

print("Loading ground truths...")
ground_truths = []
for filename in tqdm(np.array(get_filenames(MAP_PATH))[indices]):
    ground_truths.append(imread(MAP_PATH + filename))


print("Processing data...")
x = np.array([], dtype='float')
y = np.array([], dtype='float')

for i in tqdm(range(REL_SAMPLES)):
    ground_truth = ground_truths[i]
    fix_timestamps = np.array(sorted([fixation for fix_timestamps in saliency_volumes[i]
                        for fixation in fix_timestamps], key=lambda x: (x[0])))

    x = np.concatenate((x, fix_timestamps[:,0].astype(dtype='float')))
    y = np.concatenate((y, np.array([ground_truth[y - 1, x - 1] for [timestamp, [x, y]] in fix_timestamps], dtype='float')))


print("Generating heatmap...")
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

plt.scatter(x, y, c=z, s=50, edgecolor='')
plt.savefig('heatmap.png', bbox_inches='tight')
