import random

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import time
import os
import sys
from os.path import join
from torch.utils.tensorboard import SummaryWriter

from dataset.load_dataloader_cde import get_dataloader_cde
from train.STG_NCDE_train import print_model_parameters, Trainer
from models.STG_NCDE import *
from utils.loss import scaler_Loss

args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default="PEMS-BAY", type=str)
args.add_argument('--root', default="./data", type=str)
args.add_argument('--val_ratio', default=0.2, type=float)
args.add_argument('--test_ratio', default=0.2, type=float)
args.add_argument('--y_offsets', default=12, type=int)
args.add_argument('--x_offsets', default=12, type=int)
args.add_argument('--input_dim', default=2, type=int)
args.add_argument('--output_dim', default=1, type=int)
args.add_argument('--weight_decay', default=1e-3, type=eval)
args.add_argument('--default_graph', default=True, type=eval)
args.add_argument('--missing_test', default=False, type=bool)
args.add_argument('--missing_rate', default=0, type=int)
args.add_argument('--cheb_k', default=2, type=int)
args.add_argument('--solver', default="rk4", type=str)
# model
args.add_argument('--hid_dim', default=64, type=int)
args.add_argument('--hid_hid_dim', default=64, type=int)
args.add_argument('--embed_dim', default=10, type=int)
args.add_argument('--num_layers', default=2, type=int)

# train
args.add_argument('--batch_size', default=64, type=int)
args.add_argument('--epochs', default=100, type=int)
args.add_argument('--lr_init', default=0.001, type=float)
args.add_argument('--lr_decay', default=True, type=eval)
args.add_argument('--lr_decay_rate', default=0.1, type=float)
args.add_argument('--lr_decay_steps', default="[5, 20,40]", type=eval)
args.add_argument('--early_stop', default=True, type=eval)
args.add_argument('--early_stop_patience', default=15, type=int)
args.add_argument('--grad_norm', default=False, type=eval)
args.add_argument('--max_grad_norm', default=5, type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)

# test
args.add_argument('--mae_thresh', default=None, type=eval)
args.add_argument('--mape_thresh', default=0., type=float)
# log
args.add_argument('--path', default='./logs', type=str)
args.add_argument('--log_step', default=20, type=int)
args.add_argument('--comment', default="", type=str)
args.add_argument('--tensorboard', default=False, type=bool)
args.add_argument('--cuda', default=3, type=int)
args = args.parse_known_args()[0]
torch.cuda.cudnn_enabled = False
torch.backends.cudnn.deterministic = True
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

args.model_name = "stgncde"
device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
save_name = time.strftime(
    "%m-%d-%Hh%Mm") + args.comment + "_" + args.dataset + "_" + args.model_name + "_" + "embed{" + str(
    args.embed_dim) + "}" + "hid{" + str(args.hid_dim) + "}" + "hidhid{" + str(args.hid_hid_dim) + "}" + "lyrs{" + str(
    args.num_layers) + "}" + "lr{" + str(args.lr_init) + "}" + "wd{" + str(args.weight_decay) + "}"
log_dir = join(args.path, args.dataset, save_name)
if os.path.exists(log_dir):
    print('has model save path')
else:
    os.makedirs(log_dir)

w: SummaryWriter = SummaryWriter(log_dir)
train_loader, val_loader, test_loader, scaler, times, num_nodes = get_dataloader_cde(args.dataset, root=args.root,
                                                                                     x_offsets=args.x_offsets,
                                                                                     y_offsets=args.y_offsets,
                                                                                     batch_size=args.batch_size,
                                                                                     test_ratio=args.test_ratio,
                                                                                     val_ratio=args.val_ratio,
                                                                                     device=device,
                                                                                     missing_test=args.missing_test,
                                                                                     missing_rate=args.missing_rate)

model, vector_field_f, vector_field_g = make_model(num_nodes, args.input_dim, args.hid_dim, args.hid_hid_dim,
                                                   args.output_dim, args.embed_dim,
                                                   args.num_layers, args.cheb_k, args.x_offsets, args.solver,
                                                   args.default_graph)
model = model.to(device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

loss = scaler_Loss(scaler)
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init,
                             weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                    milestones=args.lr_decay_steps,
                                                    gamma=args.lr_decay_rate)

trainer = Trainer(model, args.model_name, loss, optimizer, train_loader, val_loader,
                  test_loader, scaler, lr_scheduler, device, times, w, log_dir, args.tensorboard, args.grad_norm,
                  args.max_grad_norm, args.log_step, args.lr_decay, args.epochs, args.early_stop,
                  args.early_stop_patience, args.mae_thresh, args.mape_thresh,
                  args.dataset)
trainer.train()
