from utils import *

<<<<<<< HEAD
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.image import imread
from tqdm import tqdm
from scipy import ndimage
from scipy.spatial import distance

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

IMAGE_PATH = '../data/images/'
FIXATION_PATH = '../data/fixations/'
GIF_PATH = '../data/gifs/'

TRAIN_PATH = 'train/'
VAL_PATH = 'val/'

GIF_SAMPLES = 300
W = 640
H = 480
TIMESPAN = 5000
MAX_PIXEL_DISTANCE = np.linalg.norm([W, H])
ESTIMATED_TIMESTAMP_WEIGHT = 0.006

################################################################################

def get_filenames(path):
    return [file for file in sorted(os.listdir(path)) if fnmatch.fnmatch(file, 'COCO_*')]


def get_saliency_volumes(filenames,
                         path_prefix=TRAIN_PATH,
                         etw=ESTIMATED_TIMESTAMP_WEIGHT, progress_bar=True):
    saliency_volumes = []
    errors = []

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
                min_idx = np.argmin(etw * time_diffs + distances)

                fix_timestamps.append([min(timestamps[i][min_idx][0], TIMESPAN), fixation.tolist()])
                errors.append(distances[min_idx])
                est_timestamp += fix_time

            if (len(observer) > 0):
                saliency_volume.append(fix_timestamps)

        saliency_volumes.append(saliency_volume)

    return saliency_volumes, np.mean(errors) / MAX_PIXEL_DISTANCE

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

################################################################################

filenames = np.random.choice(get_filenames(FIXATION_PATH + TRAIN_PATH), GIF_SAMPLES, replace=False)
images = []

print("Reading images...")
for filename in tqdm(filenames):
    images.append(imread(IMAGE_PATH + TRAIN_PATH + filename[:-3] + 'jpg'))
=======
filenames = get_filenames(FIXATION_PATH + TRAIN_PATH)[0:2]
>>>>>>> 344751b3447e975ec2ccab4979704b3911ceea04

print("Parsing fixations...")
saliency_volumes, _ = get_saliency_volumes(filenames, progress_bar=True)

print("Generating saliency volumes...")
temporal_maps = cp.zeros((len(saliency_volumes),25,H,W))
for i, saliency_volume in enumerate(tqdm(saliency_volumes)):
    fix_timestamps = cp.array(sorted([fixation for fix_timestamps in saliency_volume
                                      for fixation in fix_timestamps], key=lambda x: (x[0])))
    fix_timestamps = [(int(ts / 200), (x, y)) for (ts, (x, y)) in fix_timestamps]

    for ts, (x, y) in fix_timestamps:
        temporal_maps[i,ts-1,y-1,x-1] = 1

    for ts in cp.unique([ts for ts, _ in fix_timestamps]):
        temporal_maps[i,ts-1] = ndimage.filters.gaussian_filter(temporal_maps[i,ts-1], 25)

    for x in range(W):
        for y in range(H):
            temporal_maps[i,:,y,x] = ndimage.gaussian_filter1d(temporal_maps[i,:,y,x], 2, 0)

    temporal_maps[i] /= temporal_maps[i].max()

cp.save('../data/saliency_volumes_train.cpy', temporal_maps)
test = cp.load('../data/saliency_volumes_train.cpy')
print(test.shape, test.sum())
