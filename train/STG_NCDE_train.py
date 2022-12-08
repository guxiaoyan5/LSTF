import numpy
import torch
import math
import os
import time
import copy
import numpy as np

from utils.logger import get_logger
from utils.metrics import All_Metrics


def print_model_parameters(model, only_num=True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')
    return total_num


class Trainer(object):
    def __init__(self, model, model_name, loss, optimizer, train_loader, val_loader,
                 test_loader, scaler, lr_scheduler, device, times, w, log_dir, tensorboard, grad_norm,
                 max_grad_norm, log_step, lr_decay, epochs, early_stop, early_stop_patience, mae_thresh, mape_thresh,
                 dataset):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.val_per_epoch = len(val_loader)
        self.log_dir = log_dir
        self.best_path = os.path.join(log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(log_dir, 'loss.png')
        # log
        if os.path.isdir(log_dir) is False:
            os.makedirs(log_dir, exist_ok=True)
        self.logger = get_logger(log_dir, name=model_name, debug=False)
        self.logger.info('Experiment log path in: {}'.format(log_dir))
        total_param = print_model_parameters(model, only_num=False)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)
        self.logger.info(self.model)
        self.logger.info("Total params: {}".format(str(total_param)))
        self.device = device
        self.times = times.to(self.device, dtype=torch.float)
        self.w = w
        self.tensorboard = tensorboard
        self.grad_norm = grad_norm
        self.max_grad_norm = max_grad_norm
        self.log_step = log_step
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.mae_thresh = mae_thresh
        self.mape_thresh = mape_thresh
        self.dataset = dataset

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (valid_coeffs, label) in enumerate(self.val_loader):
                # valid_coeffs = valid_coeffs.to(self.device)
                # label = label.to(self.device)
                output = self.model(valid_coeffs, self.times)
                loss = self.loss(output, label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        # if self.tensorboard:
        #     self.w.add_scalar(f'valid/loss', val_loss, epoch)
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (train_coeffs, label) in enumerate(self.train_loader):
            # train_coeffs = train_coeffs.to(self.device)
            # label = label.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(train_coeffs, self.times)
            loss = self.loss(output, label)
            loss.backward()

            # add max grad clipping
            if self.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            # log information
            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))
        # if self.tensorboard:
        #     self.w.add_scalar(f'train/loss', train_epoch_loss, epoch)

        # learning rate decay
        if self.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            train_epoch_loss = self.train_epoch(epoch)
            val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            if self.early_stop:
                if not_improved_count == self.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop_patience))
                    break
            if best_state:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
        np.save(self.log_dir + '/{}_train_loss.npy'.format(self.dataset), numpy.array(train_loss_list))
        np.save(self.log_dir + '/{}_val_loss.npy'.format(self.dataset), numpy.array(val_loss_list))
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        torch.save(best_model, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)
        self.model.load_state_dict(best_model)
        self.test(self.model, self.test_loader, self.scaler, self.logger, None, self.times, self.device,
                  self.log_dir, self.dataset, self.mae_thresh, self.mape_thresh)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, data_loader, scaler, logger, path, times, device, log_dir, dataset, mae_thresh, mape_thresh):
        if path is not None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            model.load_state_dict(state_dict)
            model.to(device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (test_coeffs, label) in enumerate(data_loader):
                # test_coeffs = test_coeffs.to(device)
                # label = label.to(device)
                output = model(test_coeffs, times)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0)).cpu().numpy()
        # y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0)).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
        np.save(log_dir + '/{}_true.npy'.format(dataset), y_true)
        np.save(log_dir + '/{}_pred.npy'.format(dataset), y_pred)
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                          mae_thresh, mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape * 100))
        mae, rmse, mape = All_Metrics(y_pred, y_true, mae_thresh, mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
            mae, rmse, mape * 100))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
