import argparse
import os
import torch
from torch.autograd import Variable
from collections import OrderedDict
from torch.utils.data import DataLoader
import numpy as np, cv2
import torch.nn as nn
from dataloader import SaliconVolDataset
from tqdm import tqdm
from utils import *
from helpers import *
from model import PNASVolModel
from loss import *
from matplotlib.image import imread

parser = argparse.ArgumentParser()

parser.add_argument('--model_val_path',default="model.pt", type=str)
parser.add_argument('--time_slices',default=5, type=int)
parser.add_argument('--samples',default=50, type=int)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--dataset_dir',default="../data/", type=str)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PNASVolModel(args.time_slices)
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
val_pred_dir = args.dataset_dir + "volumes/predictions/"
val_vol_gt_dir = args.dataset_dir + "volumes/ground_truths/"

val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]
val_dataset = SaliconVolDataset(val_img_dir, val_vol_dir, val_img_ids, args.time_slices)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

with torch.no_grad():
    model.eval()
    os.makedirs(val_pred_dir, exist_ok=True)
    os.makedirs(val_vol_gt_dir, exist_ok=True)

    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()
    
    for idx, (img, gt_vol) in enumerate(tqdm(val_loader)):
        img = img.to(device)
        gt_vol = gt_vol.to(device)
        pred_vol = model(img)

        for i in range(pred_vol.size()[0]):
            pred_map = pred_vol[i]
            blur_map = pred_map.cpu().squeeze(0).clone().numpy()
            blur_map = blur(blur_map).to(device)

            cc_loss.update(cc(blur_map, gt_vol[i]))
            kldiv_loss.update(kldiv(blur_map, gt_vol[i]))
            nss_loss.update(nss(blur_map, gt_vol[i]))
            sim_loss.update(similarity(blur_map, gt_vol[i]))

        if idx < args.samples:
            pred_vol = np.swapaxes(pred_vol.squeeze(0).detach().cpu().numpy(), 0, -1)
            pred_vol = np.swapaxes(cv2.resize(pred_vol, (H, W)), 0, -1)

            vol = np.swapaxes(vol.squeeze(0).detach().cpu().numpy(), 0, -1)
            vol = np.swapaxes(cv2.resize(vol, (H, W)), 0, -1)
            
            img_path = os.path.join(val_img_dir, val_img_ids[idx] + '.jpg')
            img = imread(img_path)

            anim1 = animate(pred_vol, img, False)
            anim1.save(val_pred_dir + val_img_ids[idx] + '.gif', writer=animation.PillowWriter(fps=2))

            anim2 = animate(vol, img, False)
            anim2.save(val_vol_gt_dir + val_img_ids[idx] + '.gif', writer=animation.PillowWriter(fps=2))

            plt.close('all')

    print('KLDIV : {:.5f}, CC : {:.5f}, SIM : {:.5f}'.format(kl_avg.item() / (idx + 1), cc_avg.item() / (idx + 1), sim_avg.item() / (idx + 1)))

        
        


