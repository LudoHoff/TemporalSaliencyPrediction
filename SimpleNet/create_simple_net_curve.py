import wandb

wandb.init(project="saliency")

for i in range(40):
    wandb.log({"CC": 0.907, 'KLDIV': 0.201, 'NSS': 1.960, 'SIM': 0.793})