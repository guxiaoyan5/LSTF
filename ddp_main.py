import math
import random
import torch
import torch.nn as nn
import argparse
import time
import os
from os.path import join
import torch.distributed as dist
from dataset.load_dataset import get_data_loader, load_distance, get_data_loader_ddp
from models.LDSTGNN import *
from train.train import Trainer, print_model_parameters, DDPTrainer
from utils.adjacency import *
from utils.loss import scaler_Loss

args = argparse.ArgumentParser(description='arguments')
# data
args.add_argument('--dataset', default="PEMS03", type=str)
args.add_argument('--root', default="data", type=str)
args.add_argument('--val_ratio', default=0.2, type=float)
args.add_argument('--test_ratio', default=0.2, type=float)
args.add_argument('--y_offsets', default=12, type=int)
args.add_argument('--x_offsets', default=12, type=int)
args.add_argument('--input_dim', default=2, type=int)
args.add_argument('--output_dim', default=1, type=int)
args.add_argument('--num_workers', default=0, type=int)
args.add_argument('--pin_memory', default=False, type=bool)
# train
args.add_argument('--batch_size', default=64, type=int)
args.add_argument('--epochs', default=100, type=int)
args.add_argument('--lr_init', default=0.001, type=float)
args.add_argument('--lr_decay', default=True, type=eval)
args.add_argument('--gamma', default=0.96, type=float)
args.add_argument('--early_stop', default=True, type=eval)
args.add_argument('--early_stop_patience', default=20, type=int)
# args.add_argument('--use_curriculum_learning', action='store_true', default=False)
args.add_argument('--use_curriculum_learning', default=False)
args.add_argument('--grad_norm', default=True, type=eval)
args.add_argument('--max_grad_norm', default=5, type=int)
args.add_argument('--weight_decay', default=0.00001, type=eval)
args.add_argument('--seed', default=2, type=int)
args.add_argument('--l', default=0.98, type=float)
# model
args.add_argument('--nb_block', default=4, type=int)
args.add_argument('--K', default=3, type=int)
args.add_argument('--n_heads', default=3, type=int)
args.add_argument('--nb_filter', default=24, type=int)
args.add_argument('--d_model', default=80, type=int)
args.add_argument('--d_k', default=16, type=int)
args.add_argument('--d_v', default=16, type=int)
args.add_argument('--decay_r', default=0.1, type=float)
args.add_argument('--decay_interval', default=10, type=int)
args.add_argument('--final_r', default=0.5, type=float)
args.add_argument('--init_r', default=0.5, type=float)
args.add_argument('--dropout', default=0.3, type=float)
# test
args.add_argument('--mae_thresh', default=None, type=eval)
args.add_argument('--mape_thresh', default=0.5, type=float)

# log
args.add_argument('--path', default='./logs', type=str)
args.add_argument('--log_step', default=20, type=int)
args.add_argument('--comment', default="", type=str)
args.add_argument('--model_name', default="LDSTGNN-NOTAT", type=str)
args.add_argument('--local_rank', default=-1, type=int)

args = args.parse_known_args()[0]
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.cudnn_enabled = False
torch.backends.cudnn.deterministic = True
torch.distributed.init_process_group(backend="nccl")
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
save_name = time.strftime(
    "%m-%d-%Hh%Mm") + args.comment + "_" + args.dataset + "_" + args.model_name + "nb_block" + str(args.nb_block) + "_K" \
            + str(args.K) + "_nb_filter" + str(args.nb_filter) + "_d_model" \
            + str(args.d_model) + "_d_k" + str(args.d_k) + "_d_v" + str(args.d_v)
log_dir = join(args.path, args.dataset, save_name)

if os.path.exists(log_dir):
    print('has model save path')
elif dist.get_rank() == 0:
    os.makedirs(log_dir)

train_loader, val_loader, test_loader, scaler, num_nodes = get_data_loader_ddp(args.dataset, root=args.root,
                                                                               x_offsets=args.x_offsets,
                                                                               y_offsets=args.y_offsets,
                                                                               batch_size=args.batch_size,
                                                                               test_ratio=args.test_ratio,
                                                                               val_ratio=args.val_ratio,
                                                                               add_time_in_day=True)
model = LDSTGNN(1, args.nb_block, args.input_dim, args.K, args.nb_filter, 1,
                args.y_offsets, args.x_offsets, num_nodes, args.d_model, args.d_k, args.d_v, args.n_heads,
                args.dropout).to(
    device)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 设置多个gpu的BN同步
model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[args.local_rank],
                                                  output_device=args.local_rank,
                                                  find_unused_parameters=True,
                                                  broadcast_buffers=False)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)
# loss = nn.L1Loss()
loss = scaler_Loss(scaler, args.mae_thresh)
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, amsgrad=False,
                             weight_decay=args.weight_decay)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=args.T_0,
#                                                                     T_mult=args.T_mult)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)

trainer = DDPTrainer(model, args.model_name, loss, optimizer, train_loader, val_loader,
                     test_loader, scaler, lr_scheduler, device, log_dir, args.grad_norm,
                     args.max_grad_norm, args.log_step, args.lr_decay, args.epochs, args.early_stop,
                     args.early_stop_patience, args.mae_thresh, args.mape_thresh, args.dataset, args.decay_interval,
                     args.decay_r, args.init_r, args.final_r, args.l,
                     use_curriculum_learning=args.use_curriculum_learning,
                     args=args, num_node=num_nodes)
trainer.train()
# trainer.test(trainer.model, trainer.test_loader, trainer.scaler, trainer.logger,
#              "logs/METR-LA/09-07-09h58m_METR-LA_model_lr{0.01}wd{0.001}/best_model.pth", trainer.device,
#              trainer.log_dir, trainer.dataset, trainer.mae_thresh, trainer.mape_thresh, trainer.adj)
