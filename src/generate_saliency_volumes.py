from utils import *
#from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi, sqrt, exp

def gauss(n, sigma):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]


class GaussianBlur1D(nn.Module):
    def __init__(self):
        super(GaussianBlur1D, self).__init__()
        self.size = 17
        kernel = gauss(self.size, 2)
        kernel = torch.cuda.HalfTensor(kernel).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        return F.conv1d(x.unsqueeze(1), self.weight, padding=int(self.size/2))


class GaussianBlur2D(nn.Module):
    def __init__(self):
        super(GaussianBlur2D, self).__init__()
        self.size = 101
        kernel = np.outer(gauss(self.size, 25))
        kernel = torch.from_numpy(kernel, dtype=torch.float16).unsqueeze(0).unsqueeze(0).cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        return F.conv2d(x.unsqueeze(1), self.weight, padding=int(self.size/2))


filenames = get_filenames(FIXATION_PATH + TRAIN_PATH)[0:2]

print("Parsing fixations...")
saliency_volumes = get_saliency_volumes(filenames, progress_bar=True)

print("Generating saliency volumes...")
temporal_maps = torch.zeros([len(filenames),25,H,W], dtype=torch.float16).cuda()

conv1D = GaussianBlur1D().cuda()
conv2D = GaussianBlur2D().cuda()

for i, saliency_volume in enumerate(tqdm(saliency_volumes)):
    fix_timestamps = sorted([fixation for fix_timestamps in saliency_volume
                                      for fixation in fix_timestamps], key=lambda x: x[0])
    fix_timestamps = [(int(ts / 200), (x, y)) for (ts, (x, y)) in fix_timestamps]

    for ts, (x, y) in fix_timestamps:
        temporal_maps[i,ts-1,y-1,x-1] = 1

    for ts in np.unique([ts for ts, _ in fix_timestamps]):
        temporal_maps[i,ts-1] = conv1D.forward(temporal_maps[i,ts-1])
        #temporal_maps[i,ts-1] = (temporal_maps[i,ts-1], kernel2D, kernel2D)
        #ndimage.filters.gaussian_filter(temporal_maps[i,ts-1], 25)

    for x in range(W):
        for y in range(H):
            temporal_maps[i,:,y,x] = conv2D(temporal_maps[i,:,y,x])
            #temporal_maps[i,:,y,x] = gpu.convolve(temporal_maps[i,:,y,x], kernel1D)
            #ndimage.gaussian_filter1d(temporal_maps[i,:,y,x], 2, 0)

    temporal_maps[i] /= temporal_maps[i].max()

np.save('../data/saliency_volumes_train.npy', temporal_maps)
