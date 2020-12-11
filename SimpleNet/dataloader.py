from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch
import os, cv2
from helpers import *

class SaliconDataset(DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, img_ids, exten='.png'):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.img_ids = img_ids
        self.exten = exten
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        gt_path = os.path.join(self.gt_dir, img_id + self.exten)
        fix_path = os.path.join(self.fix_dir, img_id + self.exten)

        img = Image.open(img_path).convert('RGB')
        img = self.img_transform(img)

        gt = np.array(Image.open(gt_path).convert('L'))
        gt = gt.astype('float')
        gt = cv2.resize(gt, (256,256))
        if np.max(gt) > 1.0:
            gt = gt / 255.0

        fixations = np.array(Image.open(fix_path).convert('L'))
        fixations = fixations.astype('float')
        fixations = (fixations > 0.5).astype('float')

        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)

    def __len__(self):		
         return len(self.img_ids)

class SaliconVolDataset(DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, vol_dir, img_ids, time_slices, exten='.png'):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.vol_dir = vol_dir
        self.img_ids = img_ids
        self.time_slices = time_slices
        self.exten = exten
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])

        self.conv1D = GaussianBlur1D(time_slices).cuda()
        self.conv2D = GaussianBlur2D().cuda()

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        gt_path = os.path.join(self.gt_dir, img_id + self.exten)
        fix_path = os.path.join(self.fix_dir, img_id + self.exten)

        img = Image.open(img_path).convert('RGB')
        img = self.img_transform(img)

        gt = np.array(Image.open(gt_path).convert('L'))
        gt = gt.astype('float')
        gt = cv2.resize(gt, (256,256))
        if np.max(gt) > 1.0:
            gt = gt / 255.0

        saliency_volume = np.zeros((self.time_slices, 256, 256))
        for i in range(self.time_slices):
            vol_path = os.path.join(self.vol_dir, img_id + '_' + str(i) + self.exten)
            saliency_volume[i] = cv2.imread(vol_path, cv2.IMREAD_GRAYSCALE)
        saliency_volume.astype('float')

        fixations = np.array(Image.open(fix_path).convert('L'))
        fixations = fixations.astype('float')
        fixations = (fixations > 0.5).astype('float')

        print(saliency_volume.shape)
        print(saliency_volume.min())
        print(saliency_volume.max())
        print()

        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        return img, torch.FloatTensor(gt), torch.FloatTensor(saliency_volume), torch.FloatTensor(fixations)

    def __len__(self):
        return len(self.img_ids)

class TestLoader(DataLoader):
    def __init__(self, img_dir, img_ids):
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        img = Image.open(img_path).convert('RGB')
        sz = img.size
        img = self.img_transform(img)
        return img, img_id, sz

    def __len__(self):
        return len(self.img_ids)

class MITDataset(DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, img_ids, exten='.png', val=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.img_ids = img_ids
        self.val = val
        self.exten = exten
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + '.png')
        gt_path = os.path.join(self.gt_dir, img_id + self.exten)
        fix_path = os.path.join(self.fix_dir, img_id + self.exten)

        img = Image.open(img_path).convert('RGB')

        gt = np.array(Image.open(gt_path).convert('L'))
        gt = gt.astype('float')
        gt = cv2.resize(gt, (256,256))

        fixations = np.array(Image.open(fix_path).convert('L'))
        fixations = fixations.astype('float')

        img = self.img_transform(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0
        fixations = (fixations > 0.5).astype('float')

        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        if self.val:
            return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)
        else:
            return img, torch.FloatTensor(gt), torch.FloatTensor(gt)


    def __len__(self):
        return len(self.img_ids)
