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
        gt = gt.to(device)
        vol = vol.to(device)
        pred_vol, _ = model(img)
        
        kl_loss = torch.FloatTensor([0.0]).to(device)
        cc_loss = torch.FloatTensor([0.0]).to(device)
        sim_loss = torch.FloatTensor([0.0]).to(device)

        for slice in range(args.time_slices):
            pred_map = pred_vol[0,slice].unsqueeze(0)
            gt = vol[0,slice].unsqueeze(0)

            kl_loss += kldiv(pred_map, gt)
            cc_loss += cc(pred_map, gt)
            sim_loss += similarity(pred_map, gt)
        
        kl_avg.update(kl_loss / args.time_slices)
        cc_avg.update(cc_loss / args.time_slices)
        sim_avg.update(sim_loss / args.time_slices)

        if i < args.samples:
            pred_vol = np.swapaxes(pred_vol.squeeze(0).detach().cpu().numpy(), 0, -1)
            pred_vol = np.swapaxes(cv2.resize(pred_vol, (H, W)), 0, -1)

            vol = np.swapaxes(vol.squeeze(0).detach().cpu().numpy(), 0, -1)
            vol = np.swapaxes(cv2.resize(vol, (H, W)), 0, -1)
            
            img_path = os.path.join(val_img_dir, val_img_ids[i] + '.jpg')
            img = imread(img_path)

            anim1 = animate(pred_vol, img, False)
            anim1.save(args.dataset_dir + args.results_dir + str(i) + '_predicted_volume.gif', writer=animation.PillowWriter(fps=10))

            anim2 = animate(vol, img, False)
            anim2.save(args.dataset_dir + args.results_dir + str(i) + '_ground_truth_volume.gif', writer=animation.PillowWriter(fps=10))

            plt.close('all')

    print('KLDIV : {:.5f}, CC : {:.5f}, SIM : {:.5f}'.format(kl_avg.avg, cc_avg.avg, sim_avg.avg))

        
        


