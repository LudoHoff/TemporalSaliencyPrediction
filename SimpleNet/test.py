import argparse
import os
import torch
import sys
import time
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from dataloader import TestLoader, SaliconDataset
from loss import *
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--val_img_dir',default="../data/images/test/", type=str)
parser.add_argument('--time_slices',default=5, type=int)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--enc_model',default="pnas", type=str)
parser.add_argument('--results_dir',default="../data/predictions/", type=str)
parser.add_argument('--validate',default=0, type=int)
parser.add_argument('--save_results',default=1, type=int)
parser.add_argument('--dataset_dir',default="../data/", type=str)

# Path of the saved model
parser.add_argument('--model_val_path',default="model.pt", type=str)
# If the model type is pnas_boosted, specify the path of the pre-trained pnas model here
parser.add_argument('--model_path',default="", type=str)
# If the model type is pnas_boosted, specify the path of the pre-trained pnasvol model here
parser.add_argument('--model_vol_path',default="", type=str)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.enc_model == "pnas":
    print("PNAS Model")
    from model import PNASModel
    model = PNASModel()

elif args.enc_model == "pnas_boosted":
    print("PNAS Boosted Model")
    from model import PNASBoostedModel
    model = PNASBoostedModel(device, args.model_path, args.model_vol_path, args.time_slices)

elif args.enc_model == "densenet":
    print("DenseNet Model")
    from model import DenseModel
    model = DenseModel()

elif args.enc_model == "resnet":
    print("ResNet Model")
    from model import ResNetModel
    model = ResNetModel()

elif args.enc_model == "vgg":
    print("VGG Model")
    from model import VGGModel
    model = VGGModel()

elif args.enc_model == "mobilenet":
    print("Mobile NetV2")
    from model import MobileNetV2
    model = MobileNetV2()

if args.enc_model!="mobilenet":
    model = nn.DataParallel(model)


state_dict = torch.load(args.model_val_path)
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if 'module' not in k.split('.')[0]:
        k = 'module.' + k
    else:
        k = k.replace('features.module.', 'module.features.')
    new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model = model.to(device)

val_img_ids = os.listdir(args.val_img_dir)
val_dataset = TestLoader(args.val_img_dir, val_img_ids)
vis_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

def validate(model, loader, device, args):
    model.eval()
    tic = time.time()
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()

    for (img, gt, fixations) in tqdm(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        pred_map = model(img)

        # Blurring
        blur_map = pred_map.cpu().squeeze(0).clone().numpy()
        blur_map = blur(blur_map).unsqueeze(0).to(device)

        cc_loss.update(cc(blur_map, gt))
        kldiv_loss.update(kldiv(blur_map, gt))
        nss_loss.update(nss(blur_map, gt))
        sim_loss.update(similarity(blur_map, gt))

    print('CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}  time:{:3f} minutes'.format(cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, (time.time()-tic)/60))
    sys.stdout.flush()

    return cc_loss.avg

if args.validate:
	val_img_dir = args.dataset_dir + "images/val/"
	val_gt_dir = args.dataset_dir + "maps/val/"
	val_fix_dir = args.dataset_dir + "fixation_maps/val/"

	val_img_ids = sorted([nm.split(".")[0] for nm in os.listdir(val_img_dir)])

	val_dataset = SaliconDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)
	with torch.no_grad():
		validate(model, val_loader, device, args)
if args.save_results:
	visualize_model(model, vis_loader, device, args)
