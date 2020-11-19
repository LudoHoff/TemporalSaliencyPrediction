from utils import *

filenames = cp.random.choice(get_filenames(FIXATION_PATH + TRAIN_PATH), GIF_SAMPLES, replace=False)
images = []

print("Reading images...")
for filename in tqdm(filenames):
    images.append(imread(IMAGE_PATH + TRAIN_PATH + filename[:-3] + 'jpg'))

print("Parsing fixations...")
saliency_volumes, _ = get_saliency_volumes(filenames, progress_bar=True)

print("Generating saliency volumes...")
for i, saliency_volume in tqdm(enumerate(saliency_volumes)):
    fix_timestamps = cp.array(sorted([fixation for fix_timestamps in saliency_volume
                                      for fixation in fix_timestamps], key=lambda x: (x[0])))
    fix_timestamps = [(int(ts / 100), (x, y)) for (ts, (x, y)) in fix_timestamps]
    temporal_map = cp.zeros((50,H,W))

    for ts, (x, y) in fix_timestamps:
        temporal_map[ts-1,y-1,x-1] = 1

    for ts in cp.unique([ts for ts, _ in fix_timestamps]):
        temporal_map[ts-1] = ndimage.filters.gaussian_filter(temporal_map[ts-1], 30)

    for x in range(W):
        for y in range(H):
            #temporal_map[:,y,x] = signal.convolve(temporal_map[:,y,x], gauss(20, 1), mode='same', method='auto')
            temporal_map[:,y,x] = ndimage.gaussian_filter1d(temporal_map[:,y,x], 2, 0)

    temporal_map /= temporal_map.max()
    ani = animate(temporal_map, images[i], False)
    ani.save(GIF_PATH + TRAIN_PATH + filenames[i][:-3] + 'gif', writer=animation.PillowWriter(fps=10))
