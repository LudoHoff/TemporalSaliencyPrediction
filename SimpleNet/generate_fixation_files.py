from helpers import *
from operator import itemgetter
from itertools import groupby
import cv2

def generate_fixation_files(path):
    filenames = [nm.split(".")[0] for nm in os.listdir(FIXATION_PATH + path)]
    print('Parsing fixations of ' + path + '...')
    fixation_volumes = parse_fixations(filenames, FIXATION_PATH + path)

    print('Saving parsed fixations of ' + path + '...')
    for i, fixation_volume in enumerate(tqdm(fixation_volumes)):
        fix_timestamps = sorted([fixation for fix_timestamps in fixation_volume
                                        for fixation in fix_timestamps], key=lambda x: x[0])
        fix_timestamps = np.array([(int(ts * 10 / TIMESPAN), (x, y)) for (ts, (x, y)) in fix_timestamps])

        # Saving fixation map
        fix_map = np.zeros(shape=(W,H))
        for _, coords in fix_timestamps:
            fix_map[coords[0] - 1, coords[1] - 1] = 1
        
        fix_map = fix_map.T
        cv2.imwrite(FIX_MAP_PATH + path + filenames[i] + '.png', 255 * fix_map / fix_map.max())

        # Saving fixation list with timestamps
        compressed = np.array([(key, list(v[1] for v in valuesiter))
                            for key,valuesiter in groupby(fix_timestamps, key=itemgetter(0))])
        np.save(PARS_FIX_PATH + path + filenames[i] + '.npy', compressed)

generate_fixation_files('train/')
generate_fixation_files('val/')
