import matplotlib
import argparse
import glob, os
import pickle
import torch
import wandb
import time
import sys
import cv2

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal as Norm
from multiprocessing import set_start_method
from scipy.stats import multivariate_normal
from torchvision import transforms, utils
from dataloader import SaliconVolDataset
from dataloader import SaliconDataset
from torch.utils.data import DataLoader
from utils import blur, AverageMeter
from torch.autograd import Variable
from model import PNASVolModel
from PIL import Image
from helpers import *
from loss import *

matplotlib.use('Agg')

if __name__ == '__main__':
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    wandb.init(project="saliency")

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_epochs',default=40, type=int)
    parser.add_argument('--lr',default=1e-4, type=float)
    parser.add_argument('--kldiv',default=True, type=bool)
    parser.add_argument('--cc',default=True, type=bool)
    parser.add_argument('--nss',default=False, type=bool)
    parser.add_argument('--sim',default=False, type=bool)
    parser.add_argument('--nss_emlnet',default=False, type=bool)
    parser.add_argument('--nss_norm',default=False, type=bool)
    parser.add_argument('--l1',default=False, type=bool)
    parser.add_argument('--lr_sched',default=False, type=bool)
    parser.add_argument('--dilation',default=False, type=bool)
    parser.add_argument('--optim',default="Adam", type=str)

    parser.add_argument('--load_weight',default=1, type=int)
    parser.add_argument('--kldiv_coeff',default=1.0, type=float)
    parser.add_argument('--step_size',default=5, type=int)
    parser.add_argument('--cc_coeff',default=-1.0, type=float)
    parser.add_argument('--sim_coeff',default=-1.0, type=float)
    parser.add_argument('--nss_coeff',default=1.0, type=float)
    parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
    parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
    parser.add_argument('--l1_coeff',default=1.0, type=float)
    parser.add_argument('--loss_gt_coeff',default=1.0, type=float)
    parser.add_argument('--loss_vol_coeff',default=0.25, type=float)
    parser.add_argument('--train_enc',default=1, type=int)

    parser.add_argument('--dataset_dir',default="../data/", type=str)
    parser.add_argument('--batch_size',default=32, type=int)
    parser.add_argument('--log_interval',default=60, type=int)
    parser.add_argument('--no_workers',default=4, type=int)
    parser.add_argument('--time_slices',default=10, type=int)
    parser.add_argument('--model_val_path',default="model.pt", type=str)

    args = parser.parse_args()

    train_img_dir = args.dataset_dir + "images/train/"
    train_gt_dir = args.dataset_dir + "maps/train/"
    train_fix_dir = args.dataset_dir + "fixation_maps/train/"
    train_vol_dir = args.dataset_dir + "saliency_volumes_" + str(args.time_slices) + "/train/"

    val_img_dir = args.dataset_dir + "images/val/"
    val_gt_dir = args.dataset_dir + "maps/val/"
    val_fix_dir = args.dataset_dir + "fixation_maps/val/"
    val_vol_dir = args.dataset_dir + "saliency_volumes_" + str(args.time_slices) + "/val/"

    print("PNAS with saliency volume Model")
    model = PNASVolModel(train_enc=bool(args.train_enc), load_weight=args.load_weight, time_slices=args.time_slices)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    wandb.watch(model)

    train_img_ids = [nm.split(".")[0] for nm in os.listdir(train_img_dir)]
    val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]

    train_dataset = SaliconVolDataset(train_img_dir, train_gt_dir, train_fix_dir, train_vol_dir, train_img_ids, args.time_slices)
    val_dataset = SaliconDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

    def loss_func(pred_map, gt, fixations, args):
        loss = torch.FloatTensor([0.0]).cuda()
        criterion = nn.L1Loss()
        if args.kldiv:
            loss += args.kldiv_coeff * kldiv(pred_map, gt)
        if args.cc:
            loss += args.cc_coeff * cc(pred_map, gt)
        if args.nss:
            loss += args.nss_coeff * nss(pred_map, fixations)
        if args.l1:
            loss += args.l1_coeff * criterion(pred_map, gt)
        if args.sim:
            loss += args.sim_coeff * similarity(pred_map, gt)
        return loss

    def vol_loss_func(pred_vol, vol, args):
        loss = torch.FloatTensor([0.0]).cuda()
        criterion = nn.L1Loss()

        for i in range(pred_vol.size()[0]):
            pred_map = pred_vol[i]
            gt = vol[i]
            if args.kldiv:
                loss += args.kldiv_coeff * kldiv(pred_map, gt)
            if args.cc:
                loss += args.cc_coeff * cc(pred_map, gt)
            if args.l1:
                loss += args.l1_coeff * criterion(pred_map, gt)
            if args.sim:
                loss += args.sim_coeff * similarity(pred_map, gt)
        return loss / pred_vol.size()[0]

    def train(model, optimizer, loader, epoch, device, args):
        model.train()
        tic = time.time()

        total_loss = 0.0
        cur_loss = 0.0
        vol_loss = 0.0
        gt_loss = 0.0

        for idx, (img, gt, vol, fixations) in enumerate(loader):
            img = img.to(device)
            gt = gt.to(device)
            vol = vol.to(device)
            fixations = fixations.to(device)

            optimizer.zero_grad()
            pred_vol, pred_map = model(img)
            assert pred_vol.size() == vol.size()
            assert pred_map.size() == gt.size()
            loss_gt = loss_func(pred_map, gt, fixations, args)
            loss_vol = vol_loss_func(pred_vol, vol, args)
            loss = args.loss_gt_coeff * loss_gt + args.loss_vol_coeff * loss_vol
            loss.backward()

            total_loss += loss.item()
            cur_loss += loss.item()
            vol_loss += loss_vol.item()
            gt_loss += loss_gt.item()

            optimizer.step()
            if idx%args.log_interval==(args.log_interval-1):
                print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60))
                wandb.log({"loss": cur_loss/args.log_interval})
                wandb.log({"vol loss": vol_loss/args.log_interval})
                wandb.log({"gt loss": gt_loss/args.log_interval})

                cur_loss = 0.0
                vol_loss = 0.0
                gt_loss = 0.0
                sys.stdout.flush()

        print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
        sys.stdout.flush()

        return total_loss/len(loader)

    def validate(model, loader, epoch, device, args):
        model.eval()
        tic = time.time()
        total_loss = 0.0
        cc_loss = AverageMeter()
        kldiv_loss = AverageMeter()
        nss_loss = AverageMeter()
        sim_loss = AverageMeter()

        for (img, gt, fixations) in loader:
            img = img.to(device)
            gt = gt.to(device)
            fixations = fixations.to(device)

            _, pred_map = model(img)

            # Blurring
            blur_map = pred_map.cpu().squeeze(0).clone().numpy()
            blur_map = blur(blur_map).unsqueeze(0).to(device)

            cc_loss.update(cc(blur_map, gt))
            kldiv_loss.update(kldiv(blur_map, gt))
            nss_loss.update(nss(blur_map, gt))
            sim_loss.update(similarity(blur_map, gt))

        print('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}  time:{:3f} minutes'.format(epoch, cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, (time.time()-tic)/60))
        wandb.log({"CC": cc_loss.avg, 'KLDIV': kldiv_loss.avg, 'NSS': nss_loss.avg, 'SIM': sim_loss.avg})
        sys.stdout.flush()

        return cc_loss.avg

    params = list(filter(lambda p: p.requires_grad, model.parameters()))

    if args.optim=="Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
    if args.optim=="Adagrad":
        optimizer = torch.optim.Adagrad(params, lr=args.lr)
    if args.optim=="SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    if args.lr_sched:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    print(device)

    for epoch in range(0, args.no_epochs):
        loss = train(model, optimizer, train_loader, epoch, device, args)

        with torch.no_grad():
            cc_loss = validate(model, val_loader, epoch, device, args)
            if epoch == 0 :
                best_loss = cc_loss
            if best_loss <= cc_loss:
                best_loss = cc_loss
                print('[{:2d},  save, {}]'.format(epoch, args.model_val_path))
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), args.model_val_path)
                else:
                    torch.save(model.state_dict(), args.model_val_path)
            print()

        if args.lr_sched:
            scheduler.step()
