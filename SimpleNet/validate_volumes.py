import argparse
import os
import torch
import cv2

import numpy as np
import torch.nn as nn

from collections import OrderedDict
from dataloader import SaliconVolDataset
from tqdm import tqdm
from utils import *
from model import PNASVolModel, VolModel
from loss import *
from matplotlib.image import imread

parser = argparse.ArgumentParser()

parser.add_argument('--model_val_path',default="../models/model.pt", type=str)
parser.add_argument('--time_slices',default=5, type=int)
parser.add_argument('--normalize',default=False, type=str)
parser.add_argument('--samples',default=25, type=int)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--dataset_dir',default="../data/", type=str)
parser.add_argument('--model',default="PNASVol", type=str)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model == 'PNASVol':
    model = PNASVolModel(time_slices=args.time_slices)
elif args.model == 'Vol':
    model = VolModel(device, time_slices=args.time_slices)

model = nn.DataParallel(model)
state_dict = torch.load(args.model_val_path)
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if 'module' not in k:
        k = 'module.' + k
    else:
        k = k.replace('features.module.', 'module.features.')
    new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model = model.to(device)

val_img_dir = args.dataset_dir + "images/val/"
val_vol_dir = args.dataset_dir + "saliency_volumes_" + str(args.time_slices) + "/val/"
val_fix_dir = args.dataset_dir + "fixation_volumes_" + str(args.time_slices) + "/val/"
val_pred_dir = args.dataset_dir + "volumes/"

val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]
val_dataset = SaliconVolDataset(val_img_dir, val_vol_dir, val_fix_dir, val_img_ids, args.time_slices)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

with torch.no_grad():
    model.eval()
    os.makedirs(val_pred_dir, exist_ok=True)

    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()

    for idx, (img, gt_vol, avg_vol) in enumerate(tqdm(val_loader)):
        img = img.to(device)
        gt_vol = gt_vol.to(device)
        pred_vol = model(img).squeeze(0)
        gt_vol = gt_vol.squeeze(0)

        if args.normalize:
            avg_vol = avg_vol.to(device).squeeze(0)
            pred_vol = pred_vol * 2 + avg_vol - 1

        #pred_vol /= pred_vol.max()

        cc_loss.update(cc(pred_vol, gt_vol))
        kldiv_loss.update(kldiv(pred_vol, gt_vol))
        nss_loss.update(nss(pred_vol, gt_vol))
        sim_loss.update(similarity(pred_vol, gt_vol))

        if idx < args.samples:
            pred_vol = np.swapaxes(pred_vol.detach().cpu().numpy(), 0, -1)
            pred_vol = np.swapaxes(cv2.resize(pred_vol, (H, W)), 0, -1)

            gt_vol = np.swapaxes(gt_vol.detach().cpu().numpy(), 0, -1)
            gt_vol = np.swapaxes(cv2.resize(gt_vol, (H, W)), 0, -1)

            img_path = os.path.join(val_img_dir, val_img_ids[idx] + '.jpg')
            img = imread(img_path)

            anim = animate(gt_vol, pred_vol, img)
            anim.save(val_pred_dir + val_img_ids[idx] + '.gif', writer=animation.PillowWriter(fps=2))

            plt.close('all')

    print('KLDIV : {:.5f}, CC : {:.5f}, SIM : {:.5f}, NSS : {:.5f}'.format(kldiv_loss.avg, cc_loss.avg, sim_loss.avg, nss_loss.avg))

        
        


