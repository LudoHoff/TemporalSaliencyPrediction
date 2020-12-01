from helpers import *
import cv2

def generate_fixation_files(path, time_slices):
    filenames = [nm.split(".")[0] for nm in os.listdir(PARS_FIX_PATH + path)]
    
    write_path = SAL_VOL_PATH + str(time_slices)
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    write_path = write_path + '/' + path
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    conv1D = GaussianBlur1D(time_slices).cuda()
    conv2D = GaussianBlur2D().cuda()

    for filename in tqdm(filenames):
        fixation_volume = np.load(PARS_FIX_PATH + path  + filename + '.npy', allow_pickle=True)
        saliency_volume = get_saliency_volume(fixation_volume, conv1D, conv2D, time_slices)
        saliency_volume = np.swapaxes(saliency_volume.squeeze(0).squeeze(0).detach().cpu().numpy(), 0, -1)
        saliency_volume = np.swapaxes(cv2.resize(saliency_volume, (256,256)), 0, -1)

        for i, saliency_slice in enumerate(saliency_volume):
            cv2.imwrite(write_path + filename + '_' + str(i) + '.png', 255 * saliency_slice / saliency_slice.max())

generate_fixation_files('train/', 10)
generate_fixation_files('val/', 10)
