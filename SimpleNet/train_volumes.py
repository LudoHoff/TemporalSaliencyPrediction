import matplotlib
import argparse
import os
import torch
import wandb
import time
import sys

import torch.nn as nn

from multiprocessing import set_start_method
from dataloader import SaliconVolDataset
from model import PNASVolModel, VolModel
from utils import *
from loss import *

matplotlib.use('Agg')

if __name__ == '__main__':
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    wandb.init(project="saliency")

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_epochs',default=10, type=int)
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
    parser.add_argument('--model',default="PNASVol", type=str)

    parser.add_argument('--normalize',default=False, type=str)
    parser.add_argument('--load_weight',default=1, type=int)
    parser.add_argument('--kldiv_coeff',default=1.0, type=float)
    parser.add_argument('--step_size',default=5, type=int)
    parser.add_argument('--cc_coeff',default=-1.0, type=float)
    parser.add_argument('--sim_coeff',default=-1.0, type=float)
    parser.add_argument('--nss_coeff',default=1.0, type=float)
    parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
    parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
    parser.add_argument('--l1_coeff',default=1.0, type=float)
    parser.add_argument('--loss_coeff',default=0.0, type=float)
    parser.add_argument('--loss_rescaling',default=None, type=str)
    parser.add_argument('--loss_rescaling_delay',default=0, type=int)
    parser.add_argument('--train_enc',default=1, type=int)

    parser.add_argument('--dataset_dir',default="../data/", type=str)
    parser.add_argument('--batch_size',default=32, type=int)
    parser.add_argument('--log_interval',default=60, type=int)
    parser.add_argument('--no_workers',default=4, type=int)
    parser.add_argument('--time_slices',default=5, type=int)
    parser.add_argument('--model_val_path',default="../models/model.pt", type=str)

    args = parser.parse_args()

    train_img_dir = args.dataset_dir + "images/train/"
    train_vol_dir = args.dataset_dir + "saliency_volumes_" + str(args.time_slices) + "/train/"

    val_img_dir = args.dataset_dir + "images/val/"
    val_vol_dir = args.dataset_dir + "saliency_volumes_" + str(args.time_slices) + "/val/"

    print("PNAS with saliency volume Model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def filter_params(params):
        return list(filter(lambda p: p.requires_grad, params))

    if args.model == 'PNASVol':
        model = PNASVolModel(train_enc=bool(args.train_enc), load_weight=args.load_weight, time_slices=args.time_slices)
        params = filter_params(model.parameters())
    elif args.model == 'Vol':
        model = VolModel(device, train_enc=bool(args.train_enc), load_weight=args.load_weight, time_slices=args.time_slices)
        models_params = [model.__dict__['model_' + str(i)].parameters() for i in range(args.time_slices)]
        models_params = [param for model_params in models_params for param in model_params]
        params = filter_params(model.parameters()) + filter_params(models_params)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    wandb.watch(model)

    train_img_ids = [nm.split(".")[0] for nm in os.listdir(train_img_dir)]
    val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]

    train_dataset = SaliconVolDataset(train_img_dir, train_vol_dir, train_img_ids, args.time_slices)
    val_dataset = SaliconVolDataset(val_img_dir, val_vol_dir, val_img_ids, args.time_slices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

    def loss_func(pred_vol, gt_vol, epoch, args):
        losses = torch.zeros(args.time_slices).cuda()
        criterion = nn.L1Loss()

        for i in range(args.time_slices):
            pred_map = pred_vol[:,i]
            gt = gt_vol[:,i]
            if args.kldiv:
                losses[i] += args.kldiv_coeff * kldiv(pred_map, gt)
            if args.cc:
                losses[i] += args.cc_coeff * cc(pred_map, gt)
            if args.l1:
                losses[i] += args.l1_coeff * criterion(pred_map, gt)
            if args.sim:
                losses[i] += args.sim_coeff * similarity(pred_map, gt)

        if epoch >= args.loss_rescaling_delay:
            if args.loss_rescaling == 'min_max':
                min_loss = torch.min(losses)
                max_loss = torch.max(losses)
                losses = losses * (1 + args.loss_coeff * (losses - min_loss) / (max_loss - min_loss))
            elif args.loss_rescaling == 'power':
                losses = losses * (losses / torch.min(losses)) ** args.loss_coeff
            elif args.loss_rescaling == 'max':
                losses = losses * (losses / torch.max(losses)) ** args.loss_coeff
            elif args.loss_rescaling == 'diff':
                min_loss = torch.min(losses)
                losses = losses + args.loss_coeff * (losses - min_loss)
            elif args.loss_rescaling == 'std':
                losses = losses + args.loss_coeff * losses.std()
            elif args.loss_rescaling == 'mean_diff':
                losses = losses + (losses - losses.mean()).clip(min=0) ** args.loss_coeff

        return torch.sum(losses) / args.time_slices
    
    def train(model, optimizer, loader, epoch, device, args):
        model.train()
        tic = time.time()

        total_loss = 0.0
        cur_loss = 0.0

        for idx, (img, gt_vol, avg_vol) in enumerate(loader):
            img = img.to(device)
            gt_vol = gt_vol.to(device)
            avg_vol = avg_vol.to(device)

            optimizer.zero_grad()
            pred_vol = model(img)
            pred_vol = pred_vol / pred_vol.max()
            
            if args.normalize:
                gt_vol = (gt_vol - avg_vol + 1) / 2
            
            assert pred_vol.size() == gt_vol.size()
            assert pred_vol.size() == avg_vol.size()
            loss = loss_func(pred_vol, gt_vol, epoch, args)
            loss.backward()
            total_loss += loss.item()
            cur_loss += loss.item()

            optimizer.step()
            if idx%args.log_interval==(args.log_interval-1):
                print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60))
                wandb.log({"loss": cur_loss/args.log_interval})
                cur_loss = 0.0
                sys.stdout.flush()

        print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
        sys.stdout.flush()

        return total_loss/len(loader)

    def validate(model, loader, epoch, device, args):
        model.eval()
        tic = time.time()
        cc_loss = AverageMeter()
        kldiv_loss = AverageMeter()
        nss_loss = AverageMeter()
        sim_loss = AverageMeter()

        for img, gt_vol, avg_vol in loader:
            img = img.to(device)
            gt_vol = gt_vol.to(device)
            avg_vol = avg_vol.to(device)

            pred_vol = model(img)
            if args.normalize:
                pred_vol = pred_vol * 2 + avg_vol - 1
                
            pred_vol /= pred_vol.max()

            # Blurring
            for i in range(pred_vol.size()[0]):
                pred_map = pred_vol[i]

                cc_loss.update(cc(pred_map, gt_vol[i]))
                kldiv_loss.update(kldiv(pred_map, gt_vol[i]))
                nss_loss.update(nss(pred_map, gt_vol[i])) # replace GT by fixations
                sim_loss.update(similarity(pred_map, gt_vol[i]))

        print('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}  time:{:3f} minutes'.format(epoch, cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, (time.time()-tic)/60))
        wandb.log({"CC": cc_loss.avg, 'KLDIV': kldiv_loss.avg, 'NSS': nss_loss.avg, 'SIM': sim_loss.avg})
        sys.stdout.flush()

        return cc_loss.avg

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
            
            if epoch >= args.loss_rescaling_delay:
                if epoch == args.loss_rescaling_delay:
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
