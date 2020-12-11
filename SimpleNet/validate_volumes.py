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
from helpers import animate
from model import PNASVolModel

parser = argparse.ArgumentParser()

parser.add_argument('--model_val_path',default="model.pt", type=str)
parser.add_argument('--results_dir',default="volume_predictions_10/val/", type=str)
parser.add_argument('--time_slices',default=10, type=int)
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
val_gt_dir = args.dataset_dir + "maps/val/"
val_fix_dir = args.dataset_dir + "fixation_maps/val/"
val_vol_dir = args.dataset_dir + "saliency_volumes_" + str(args.time_slices) + "/val/"
val_pred_dir = args.dataset_dir + args.results_dir

val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]
val_dataset = SaliconVolDataset(val_img_dir, val_gt_dir, val_fix_dir, val_vol_dir, val_img_ids, args.time_slices)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

with torch.no_grad():
    model.eval()
    os.makedirs(val_pred_dir, exist_ok=True)

    kl_avg = AverageMeter()
    cc_avg = AverageMeter()
    sim_avg = AverageMeter()
    
    for i, (img, gt, vol, fixations) in enumerate(tqdm(val_loader)):
        img = img.to(device)
        
        pred_vol, _ = model(img)
        pred_vol = np.swapaxes(pred_vol.squeeze(0).detach().cpu().numpy(), 0, -1)
        pred_vol = np.swapaxes(cv2.resize(pred_vol, img.squeeze(0).squeeze(0).size()), 0, -1)
        
        kl = torch.FloatTensor([0.0]).cuda()
        cc = torch.FloatTensor([0.0]).cuda()
        sim = torch.FloatTensor([0.0]).cuda()

        for i in range(args.time_slices):
            pred_map = pred_vol[i]
            gt = vol[i]

            kl += kldiv(pred_map, gt)
            cc += cc(pred_map, gt)
            sim += similarity(pred_map, gt)
        
        kl_avg.update(kl / args.time_slices)
        cc_avg.update(cc / args.time_slices)
        sim_avg.update(sim / args.time_slices)

        if i < args.samples:
            anim1 = animate(pred_vol, img, False)
            anim1.save(str(i) + '_predicted_volume.gif', writer=animation.PillowWriter(fps=10))

            anim2 = animate(vol, img, False)
            anim2.save(str(i) + '_ground_truth_volume.gif', writer=animation.PillowWriter(fps=10))

    print('KLDIV : {:.5f}, CC : {:.5f}, SIM : {:.5f}'.format(kl_avg.avg, cc_avg.avg, sim_avg.avg))

        
        


