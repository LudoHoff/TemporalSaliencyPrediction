import fnmatch
import os
import random
import torch
import cv2

from tqdm import tqdm
from scipy.spatial import distance
from math import pi, sqrt, exp

import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

W = 640
H = 480
TIMESPAN = 5000
MAX_PIXEL_DISTANCE = 800
ESTIMATED_TIMESTAMP_WEIGHT = 0.006
RATIO = 0.9

FIXATION_PATH = '../data/fixations/'
FIX_MAP_PATH = '../data/fixation_maps/'
SAL_VOL_PATH = '../data/saliency_volumes_'

def get_filenames(path):
    return [file for file in sorted(os.listdir(path)) if fnmatch.fnmatch(file, 'COCO_*')]

def parse_fixations(filenames,
                    path_prefix,
                    etw=ESTIMATED_TIMESTAMP_WEIGHT, progress_bar=True):
    fixation_volumes = []
    filenames = tqdm(filenames) if progress_bar else filenames

    for filename in filenames:
        # 1. Extracting data from .mat files
        mat = sio.loadmat(path_prefix + filename + '.mat')
        gaze = mat["gaze"]

        locations = []
        timestamps = []
        fixations = []

        for i in range(len(gaze)):
            locations.append(mat["gaze"][i][0][0])
            timestamps.append(mat["gaze"][i][0][1])
            fixations.append(mat["gaze"][i][0][2])

        # 2. Matching fixations with timestamps
        fixation_volume = []
        for i, observer in enumerate(fixations):
            fix_timestamps = []
            fix_time = TIMESPAN / (len(observer) + 1)
            est_timestamp = 0

            for fixation in observer:
                distances = distance.cdist([fixation], locations[i], 'euclidean')[0][..., np.newaxis]
                time_diffs = abs(timestamps[i] - est_timestamp)
                min_idx = (etw * time_diffs + distances).argmin()

                fix_timestamps.append([min(timestamps[i][min_idx][0], TIMESPAN), fixation.tolist()])
                est_timestamp += fix_time

            if (len(observer) > 0):
                fixation_volume.append(fix_timestamps)

        fixation_volumes.append(fixation_volume)

    return fixation_volumes

def gauss(n, sigma):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

class GaussianBlur1D(nn.Module):
    def __init__(self, time_slices):
        super(GaussianBlur1D, self).__init__()
        sigma = 2 * time_slices / 25
        self.size = 2 * int(4 * sigma + 0.5) + 1
        kernel = gauss(self.size, sigma)
        kernel = torch.cuda.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        pad = int(self.size/2)
        temp = F.conv1d(x, self.weight.view(1, 1, -1, 1, 1), padding=pad)
        return temp[:,:,:,pad:-pad,pad:-pad]

class GaussianBlur2D(nn.Module):
    def __init__(self):
        super(GaussianBlur2D, self).__init__()
        self.size = 201
        kernel = gauss(self.size, 25)
        kernel = torch.cuda.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        pad = int(self.size/2)
        temp = F.conv1d(x.unsqueeze(0).unsqueeze(0), self.weight.view(1, 1, 1, -1, 1), padding=pad)
        temp = temp[:,:,pad:-pad,:,pad:-pad]
        temp = F.conv1d(temp, self.weight.view(1, 1, 1, 1, -1), padding=pad)
        return temp[:,:,pad:-pad,pad:-pad]

def get_saliency_volume(fixation_volume, conv1D, conv2D, time_slices):
    fixation_map = torch.cuda.FloatTensor(time_slices,H,W).fill_(0)

    for ts, coords in fixation_volume:
        for (x, y) in coords:
            fixation_map[ts-1,y-1,x-1] = 1

    saliency_volume = conv2D.forward(fixation_map)
    saliency_volume = conv1D.forward(saliency_volume)
    return saliency_volume / saliency_volume.max()

def get_heat_image(image):
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * image), cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)

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

    return animation.ArtistAnimation(fig, formatted_images, interval=200, blit=True, repeat_delay=1000)
