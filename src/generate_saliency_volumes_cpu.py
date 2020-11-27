from scipy import ndimage
from utils import *

filenames = get_filenames(FIXATION_PATH + TRAIN_PATH)[0:10]

print("Parsing fixations...")
saliency_volumes = get_saliency_volumes(filenames, progress_bar=True)

print("Generating saliency volumes...")
temporal_maps = np.zeros((len(saliency_volumes),25,H,W))

for i, saliency_volume in enumerate(tqdm(saliency_volumes)):
    fix_timestamps = sorted([fixation for fix_timestamps in saliency_volume
                                      for fixation in fix_timestamps], key=lambda x: x[0])
    fix_timestamps = [(int(ts / 200), (x, y)) for (ts, (x, y)) in fix_timestamps]

    for ts, (x, y) in fix_timestamps:
        temporal_maps[i,ts-1,y-1,x-1] = 1
        
    temporal_maps[i] = ndimage.gaussian_filter1d(temporal_maps[i], 25, 1)
    temporal_maps[i] = ndimage.gaussian_filter1d(temporal_maps[i], 25, 2)
    temporal_maps[i] = ndimage.gaussian_filter1d(temporal_maps[i], 2, 0)
    temporal_maps[i] /= temporal_maps[i].max()

np.save('../data/saliency_volumes_train.npy', temporal_maps)