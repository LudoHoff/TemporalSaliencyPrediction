from helpers import *
from operator import itemgetter
from itertools import groupby
import cv2

def generate_fixation_files(path, time_slices):
    print('Parsing fixations of ' + path + '...')
    filenames = [nm.split(".")[0] for nm in os.listdir(FIXATION_PATH + path)]
    fixation_volumes = parse_fixations(filenames, FIXATION_PATH + path)

    write_path = SAL_VOL_PATH + str(time_slices)
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    write_path = write_path + '/' + path
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    conv1D = GaussianBlur1D(time_slices).cuda()
    conv2D = GaussianBlur2D().cuda()

    print('Generating saliency volumes of ' + path + '...')
    for i, fixation_volume in enumerate(tqdm(fixation_volumes)):
        fix_timestamps = sorted([fixation for fix_timestamps in fixation_volume
                                        for fixation in fix_timestamps], key=lambda x: x[0])
        fix_timestamps = np.array([(int(ts * time_slices / TIMESPAN), (x, y)) for (ts, (x, y)) in fix_timestamps])

        # Saving fixation map
        fix_map = np.zeros(shape=(W,H))
        for _, coords in fix_timestamps:
            fix_map[coords[0] - 1, coords[1] - 1] = 1
        
        fix_map = fix_map.T
        cv2.imwrite(FIX_MAP_PATH + path + filenames[i] + '.png', 255 * fix_map / fix_map.max())

        # Saving fixation list with timestamps
        compressed = np.array([(key, list(v[1] for v in valuesiter))
                            for key,valuesiter in groupby(fix_timestamps, key=itemgetter(0))])

        saliency_volume = get_saliency_volume(compressed, conv1D, conv2D, time_slices)
        saliency_volume = np.swapaxes(saliency_volume.squeeze(0).squeeze(0).detach().cpu().numpy(), 0, -1)
        saliency_volume = np.swapaxes(cv2.resize(saliency_volume, (256,256)), 0, -1)

        for j, saliency_slice in enumerate(saliency_volume):
            cv2.imwrite(write_path + filenames[i] + '_' + str(j) + '.png', 255 * saliency_slice / saliency_slice.max())
        

generate_fixation_files('train/', 5)
generate_fixation_files('val/', 5)
