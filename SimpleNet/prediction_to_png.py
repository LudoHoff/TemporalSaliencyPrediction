from tqdm import tqdm
import cv2
import os
from matplotlib.image import imread

directory = '../data/predictions/'
directory2 = '../data/submission/'

for filename in tqdm(sorted(os.listdir(directory))):
    if filename.endswith(".jpg"):
        img = imread(directory + filename)[:,:,0]
        name = filename[:-4] + '.png'
        cv2.imwrite(directory2 + name, img)
    else:
        continue
