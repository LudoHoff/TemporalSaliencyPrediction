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
    wandb.define_metric("AVGs/CC_avg", summary="max")

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
    parser.add_argument('--loss_rescaling_alternation',default=False, type=bool)
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
    train_fix_dir = args.dataset_dir + "fixation_volumes_" + str(args.time_slices) + "/train/"

    val_img_dir = args.dataset_dir + "images/val/"
    val_vol_dir = args.dataset_dir + "saliency_volumes_" + str(args.time_slices) + "/val/"
    val_fix_dir = args.dataset_dir + "fixation_volumes_" + str(args.time_slices) + "/val/"

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

    train_dataset = SaliconVolDataset(train_img_dir, train_vol_dir, train_fix_dir, train_img_ids, args.time_slices)
    val_dataset = SaliconVolDataset(val_img_dir, val_vol_dir, val_fix_dir, val_img_ids, args.time_slices)

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

        if epoch >= args.loss_rescaling_delay and (not args.loss_rescaling_alternation or epoch % 2 == 1):
            loss_type = args.loss_rescaling
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
            else:
                loss_type = 'classic'
        else:
            loss_type = 'classic'

        return loss_type, torch.sum(losses) / args.time_slices
    
    def train(model, optimizer, loader, epoch, device, args):
        model.train()
        tic = time.time()

        total_loss = 0.0
        cur_loss = 0.0

        for idx, (img, gt_vol, _) in enumerate(loader):
            img = img.to(device)
            gt_vol = gt_vol.to(device)

            optimizer.zero_grad()
            pred_vol = model(img)
            #pred_vol = pred_vol / pred_vol.max()
            
            assert pred_vol.size() == gt_vol.size()
            loss_type, loss = loss_func(pred_vol, gt_vol, epoch, args)

            if idx == 0:
                print('[{:2d}] loss type : '.format(epoch) + loss_type)

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
        cc_loss = []
        kldiv_loss = []
        nss_loss = []
        sim_loss = []

        for _ in range(args.time_slices):
            cc_loss.append(AverageMeter())
            kldiv_loss.append(AverageMeter())
            nss_loss.append(AverageMeter())
            sim_loss.append(AverageMeter())

        for img, gt_vol, fix_vol in tqdm(loader):
            img = img.to(device)
            gt_vol = gt_vol.to(device)
            fix_vol = fix_vol.to(device)

            pred_vol = model(img)                
            #pred_vol /= pred_vol.max()

            for i in range(pred_vol.size()[1]):
                cc_loss[i].update(cc(pred_vol[:,i], gt_vol[:,i]))
                kldiv_loss[i].update(kldiv(pred_vol[:,i], gt_vol[:,i]))
                nss_loss[i].update(nss(pred_vol[:,i], fix_vol[:,i]))
                sim_loss[i].update(similarity(pred_vol[:,i], gt_vol[:,i]))

        for i in range(args.time_slices):
            cc_loss[i] = cc_loss[i].avg.item()
            kldiv_loss[i] = kldiv_loss[i].avg.item()
            nss_loss[i] = nss_loss[i].avg.item()
            sim_loss[i] = sim_loss[i].avg.item()

        avg_cc = np.mean(cc_loss)
        avg_kl = np.mean(kldiv_loss)
        avg_nss = np.mean(nss_loss)
        avg_sim = np.mean(sim_loss)

        std_cc = np.std(cc_loss)
        std_kl = np.std(kldiv_loss)
        std_nss = np.std(nss_loss)
        std_sim = np.std(sim_loss)

        for i in range(args.time_slices):
            print('[{:1d},   val] SLICE_{:2d}'.format(epoch, i) +
            '   CC: ' + get_colored_value(cc_loss[i], avg_cc) + 
            ', KLDIV: ' + get_colored_value(kldiv_loss[i], avg_kl, False) + 
            ', NSS: ' + get_colored_value(nss_loss[i], avg_nss) + 
            ', SIM: ' + get_colored_value(sim_loss[i], avg_sim) + 
            ' time: {:3f} minutes'.format((time.time()-tic)/60))
            wandb.log({f"CC/CC_{i}": cc_loss[i], f"KLDIV/KLDIV_{i}": kldiv_loss[i], f"NSS/NSS_{i}": nss_loss[i], f"SIM/SIM_{i}": sim_loss[i]}, commit=False)

        print('[{:2d},   val] STDs       CC: {:.5f}, KLDIV: {:.5f}, NSS: {:.5f}, SIM: {:.5f} time: {:3f} minutes'.format(epoch, std_cc, std_kl, std_nss, std_sim, (time.time()-tic)/60))
        print('[{:2d},   val] AVGs       CC: {:.5f}, KLDIV: {:.5f}, NSS: {:.5f}, SIM: {:.5f} time: {:3f} minutes'.format(epoch, avg_cc, avg_kl, avg_nss, avg_sim, (time.time()-tic)/60))
        
        wandb.log({"AVGs/CC_avg": avg_cc, 'AVGs/KLDIV_avg': avg_kl, 'AVGs/NSS_avg': avg_nss, 'AVGs/SIM_avg': avg_sim}, commit=False)
        wandb.log({"STDs/CC_std": std_cc, 'STDs/KLDIV_std': std_kl, 'STDs/NSS_std': std_nss, 'STDs/SIM_std': std_sim})
        sys.stdout.flush()

        return avg_cc

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
