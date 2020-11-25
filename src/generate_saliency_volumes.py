from utils import *
import cusignal

filenames = get_filenames(FIXATION_PATH + TRAIN_PATH)[0:2]

images = []

print("Reading images...")
for filename in tqdm(filenames):
    images.append(imread(IMAGE_PATH + TRAIN_PATH + filename[:-3] + 'jpg'))

print("Parsing fixations...")
saliency_volumes = get_saliency_volumes(filenames, progress_bar=True)

print("Generating saliency volumes...")
temporal_maps = np.zeros((len(saliency_volumes),25,H,W))
kernel2D = np.outer(cusignal.gaussian(201, 25), cusignal.gaussian(201, 25))
kernel1D = cusignal.gaussian(17, 2)

for i, saliency_volume in enumerate(saliency_volumes):
    fix_timestamps = sorted([fixation for fix_timestamps in saliency_volume
                                      for fixation in fix_timestamps], key=lambda x: x[0])
    fix_timestamps = [(int(ts / 200), (x, y)) for (ts, (x, y)) in fix_timestamps]

    for ts, (x, y) in fix_timestamps:
        temporal_maps[i,ts-1,y-1,x-1] = 1

    for ts in np.unique([ts for ts, _ in fix_timestamps]):
        #temporal_maps[i,ts-1] = ndimage.filters.gaussian_filter(temporal_maps[i,ts-1], 25)
        temporal_maps[i,ts-1] = cusignal.convolve(temporal_maps[i,ts-1], kernel2D, mode='same')

    for x in range(W):
        for y in range(H):
            #temporal_maps[i,:,y,x] = ndimage.gaussian_filter1d(temporal_maps[i,:,y,x], 2, 0)
            temporal_maps[i,:,y,x] = cusignal.convolve(temporal_maps[i,:,y,x], kernel1D, mode='same')

    temporal_maps[i] /= temporal_maps[i].max()
    ani = animate(temporal_maps[i], images[i], False)
    ani.save(GIF_PATH + TRAIN_PATH + filenames[i][:-3] + 'gif', writer=animation.PillowWriter(fps=10))

np.save('../data/saliency_volumes_train.npy', temporal_maps)
