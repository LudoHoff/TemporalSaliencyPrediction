from utils import *
#from scipy import ndimage
import gputools as gpu

from math import pi, sqrt, exp

def gauss(n, sigma):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

filenames = get_filenames(FIXATION_PATH + TRAIN_PATH)[0:2]

print("Parsing fixations...")
saliency_volumes = get_saliency_volumes(filenames, progress_bar=True)

print("Generating saliency volumes...")
temporal_maps = np.zeros((len(filenames),25,H,W))

kernel1D = np.array(gauss(17, 2))
kernel2D = np.array(gauss(101, 25))

for i, saliency_volume in enumerate(saliency_volumes):
    fix_timestamps = sorted([fixation for fix_timestamps in saliency_volume
                                      for fixation in fix_timestamps], key=lambda x: x[0])
    fix_timestamps = [(int(ts / 200), (x, y)) for (ts, (x, y)) in fix_timestamps]

    for ts, (x, y) in fix_timestamps:
        temporal_maps[i,ts-1,y-1,x-1] = 1

    for ts in np.unique([ts for ts, _ in fix_timestamps]):
        temporal_maps[i,ts-1] = gpu.convolve_sep2(temporal_maps[i,ts-1], kernel2D, kernel2D)
        #ndimage.filters.gaussian_filter(temporal_maps[i,ts-1], 25)

    for x in range(W):
        for y in range(H):
            temporal_maps[i,:,y,x] = gpu.convolve(temporal_maps[i,:,y,x], kernel1D)
            #ndimage.gaussian_filter1d(temporal_maps[i,:,y,x], 2, 0)

    temporal_maps[i] /= temporal_maps[i].max()

np.save('../data/saliency_volumes_train.npy', temporal_maps)
